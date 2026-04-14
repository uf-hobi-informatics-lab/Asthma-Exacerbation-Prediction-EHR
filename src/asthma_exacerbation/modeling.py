from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier

from .metrics import binary_j_statistic_metrics


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    search_space: dict


def _build_model_specs(random_state: int) -> dict[str, ModelSpec]:
    return {
        "xgboost": ModelSpec(
            name="XGBoost",
            estimator=XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=random_state,
            ),
            search_space={
                "max_depth": Integer(3, 9),
                "min_child_weight": Real(0.1, 5.0, prior="log-uniform"),
                "gamma": Real(1e-3, 10.0, prior="log-uniform"),
                "subsample": Real(0.5, 0.9),
                "reg_alpha": Real(1e-3, 10.0, prior="log-uniform"),
                "reg_lambda": Real(1e-3, 10.0, prior="log-uniform"),
                "n_estimators": Integer(50, 400),
                "learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
            },
        ),
        "mlp": ModelSpec(
            name="MLP",
            estimator=MLPClassifier(random_state=random_state),
            search_space={
                "hidden_layer_sizes": Integer(10, 150),
                "activation": Categorical(["relu", "tanh", "logistic"]),
                "solver": Categorical(["adam", "sgd"]),
                "alpha": Real(1e-5, 1e-1, prior="log-uniform"),
                "learning_rate": Categorical(["constant", "invscaling", "adaptive"]),
                "learning_rate_init": Real(1e-4, 1e-1, prior="log-uniform"),
                "max_iter": Integer(200, 700),
            },
        ),
        "random_forest": ModelSpec(
            name="Random Forest",
            estimator=RandomForestClassifier(random_state=random_state),
            search_space={
                "n_estimators": Integer(50, 250),
                "max_depth": Integer(3, 15),
                "min_samples_split": Integer(2, 10),
                "min_samples_leaf": Integer(1, 4),
                "max_features": Categorical(["sqrt", "log2", None]),
                "bootstrap": Categorical([True, False]),
            },
        ),
        "lasso_logistic": ModelSpec(
            name="Lasso Regression",
            estimator=LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=1000,
                random_state=random_state,
            ),
            search_space={
                "C": Real(1e-3, 10.0, prior="log-uniform"),
                "tol": Real(1e-5, 1e-3, prior="log-uniform"),
                "max_iter": Integer(100, 1000),
            },
        ),
        "logistic_regression": ModelSpec(
            name="Logistic Regression",
            estimator=LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=random_state,
            ),
            search_space={
                "C": Real(1e-5, 10.0, prior="log-uniform"),
                "solver": Categorical(["newton-cg", "lbfgs", "sag", "saga"]),
                "penalty": Categorical(["l2"]),
            },
        ),
    }


def _sampling_modes(random_state: int) -> dict[str, object | None]:
    return {
        "baseline": None,
        "undersampling": RandomUnderSampler(random_state=random_state),
        "oversampling": RandomOverSampler(random_state=random_state),
    }


def _fit_with_optional_search(
    estimator,
    search_space: dict,
    train_x,
    train_y,
    do_search: bool,
    bayes_iterations: int,
    random_state: int,
    cached_best_params: dict | None,
):
    if do_search:
        search = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_space,
            n_iter=bayes_iterations,
            scoring="neg_log_loss",
            cv=5,
            n_jobs=-1,
            refit=True,
            random_state=random_state,
            verbose=0,
        )
        search.fit(train_x, train_y)
        fitted_estimator = clone(estimator).set_params(**search.best_params_)
        fitted_estimator.fit(train_x, train_y)
        return fitted_estimator, dict(search.best_params_)

    if cached_best_params is None:
        raise RuntimeError("Missing cached parameters for this fold. Run fold 1 first.")

    fitted_estimator = clone(estimator).set_params(**cached_best_params)
    fitted_estimator.fit(train_x, train_y)
    return fitted_estimator, cached_best_params


def run_cross_validated_training(
    features: pd.DataFrame,
    labels: pd.Series,
    output_dir: str | Path,
    selected_models: list[str] | None = None,
    selected_sampling: list[str] | None = None,
    n_splits: int = 5,
    bayes_iterations: int = 50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_specs = _build_model_specs(random_state=random_state)
    sampling_modes = _sampling_modes(random_state=random_state)

    if selected_models is None:
        selected_models = list(model_specs.keys())
    if selected_sampling is None:
        selected_sampling = list(sampling_modes.keys())

    unknown_models = sorted(set(selected_models) - set(model_specs))
    if unknown_models:
        raise ValueError(f"Unknown model names: {unknown_models}")

    unknown_sampling = sorted(set(selected_sampling) - set(sampling_modes))
    if unknown_sampling:
        raise ValueError(f"Unknown sampling strategies: {unknown_sampling}")

    split_iterator = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    best_params_store: dict[tuple[str, str], dict] = {}
    per_fold_results: list[pd.DataFrame] = []

    for fold_index, (train_idx, test_idx) in enumerate(split_iterator.split(features, labels), start=1):
        train_x = features.iloc[train_idx]
        test_x = features.iloc[test_idx]
        train_y = labels.iloc[train_idx]
        test_y = labels.iloc[test_idx]

        fold_dir = output_path / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_rows = []
        for sampling_name in selected_sampling:
            sampler = sampling_modes[sampling_name]
            sampled_x = train_x
            sampled_y = train_y
            if sampler is not None:
                sampled_x, sampled_y = sampler.fit_resample(train_x, train_y)

            for model_key in selected_models:
                spec = model_specs[model_key]
                cache_key = (sampling_name, model_key)
                fitted_model, best_params = _fit_with_optional_search(
                    estimator=spec.estimator,
                    search_space=spec.search_space,
                    train_x=sampled_x,
                    train_y=sampled_y,
                    do_search=(fold_index == 1),
                    bayes_iterations=bayes_iterations,
                    random_state=random_state,
                    cached_best_params=best_params_store.get(cache_key),
                )
                best_params_store[cache_key] = best_params

                predictions = fitted_model.predict(test_x)
                if hasattr(fitted_model, "predict_proba"):
                    probabilities = fitted_model.predict_proba(test_x)
                else:
                    decision = fitted_model.decision_function(test_x)
                    positive_scores = (decision - decision.min()) / (decision.max() - decision.min() + 1e-12)
                    probabilities = pd.DataFrame(
                        {
                            0: 1 - positive_scores,
                            1: positive_scores,
                        }
                    ).to_numpy()

                metrics = binary_j_statistic_metrics(
                    true_labels=test_y.to_numpy(),
                    predictions=predictions,
                    probabilities=probabilities,
                )
                metrics["model"] = spec.name
                metrics["sampling_strategy"] = sampling_name
                metrics["fold"] = fold_index
                fold_rows.append(metrics)

                model_filename = f"{model_key}_{sampling_name}.joblib"
                joblib.dump(fitted_model, fold_dir / model_filename)

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(fold_dir / "model_metrics.csv", index=False)
        per_fold_results.append(fold_df)

    all_results = pd.concat(per_fold_results, ignore_index=True)
    all_results.to_csv(output_path / "all_fold_metrics.csv", index=False)

    metric_columns = [
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auroc",
        "sensitivity",
        "specificity",
    ]
    grouped = all_results.groupby(["model", "sampling_strategy"])[metric_columns].agg(["mean", "std"])
    summary = pd.DataFrame(index=grouped.index)
    for metric_name in metric_columns:
        summary[metric_name] = (
            grouped[(metric_name, "mean")].map("{:.3f}".format)
            + " (± "
            + grouped[(metric_name, "std")].fillna(0).map("{:.3f}".format)
            + ")"
        )
    summary.to_csv(output_path / "cv_summary_metrics.csv")

    serializable_best_params = {
        f"{sampling_name}:{model_name}": params
        for (sampling_name, model_name), params in best_params_store.items()
    }
    with (output_path / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable_best_params, handle, indent=2)

    return all_results, summary
