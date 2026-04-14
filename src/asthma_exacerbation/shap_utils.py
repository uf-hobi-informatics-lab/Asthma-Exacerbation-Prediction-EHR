from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _select_explainer(model, feature_matrix: pd.DataFrame):
    tree_models = (XGBClassifier, RandomForestClassifier)
    if isinstance(model, tree_models):
        return shap.TreeExplainer(model)
    background = shap.sample(feature_matrix, min(len(feature_matrix), 200), random_state=42)
    return shap.Explainer(model, background)


def _normalize_shap_values(raw_values):
    if isinstance(raw_values, list):
        return raw_values[-1]

    if isinstance(raw_values, shap.Explanation):
        values = raw_values.values
        if values.ndim == 3:
            return shap.Explanation(
                values=values[:, :, -1],
                base_values=raw_values.base_values[:, -1],
                data=raw_values.data,
                feature_names=raw_values.feature_names,
            )
        return raw_values

    raise TypeError("Unsupported SHAP output format.")


def generate_shap_report(
    model_path: str | Path,
    feature_matrix: pd.DataFrame,
    output_dir: str | Path,
    top_k: int = 15,
    max_samples: int | None = 1000,
    random_state: int = 42,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    evaluation_features = feature_matrix.copy()
    if max_samples is not None and len(evaluation_features) > max_samples:
        evaluation_features = evaluation_features.sample(
            n=max_samples,
            random_state=random_state,
        ).sort_index()

    model = joblib.load(model_path)
    explainer = _select_explainer(model, evaluation_features)
    shap_values = _normalize_shap_values(explainer(evaluation_features))

    importance = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(-importance)[:top_k]
    top_shap_values = shap_values[:, top_indices]

    importance_df = pd.DataFrame(
        {
            "feature": np.array(top_shap_values.feature_names),
            "mean_abs_shap": importance[top_indices],
        }
    )
    importance_csv = output_path / "shap_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)

    figure_path = output_path / "shap_summary.png"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    plt.sca(axes[0])
    shap.plots.bar(top_shap_values, max_display=top_k, show=False)
    axes[0].set_xlabel("Mean(|SHAP value|)")

    plt.sca(axes[1])
    shap.plots.beeswarm(top_shap_values, max_display=top_k, show=False)
    axes[1].set_xlabel("SHAP value (impact on model output)")
    axes[1].yaxis.set_visible(False)

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    return figure_path, importance_csv
