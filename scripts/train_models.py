from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asthma_exacerbation.data_utils import load_dataset
from asthma_exacerbation.modeling import run_cross_validated_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train multimodal EHR models with 5-fold CV and Bayesian hyperparameter search.",
    )
    parser.add_argument("--features-csv", required=True, help="Path to the feature table CSV.")
    parser.add_argument("--labels-csv", required=True, help="Path to the label CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for saved models and metrics.")
    parser.add_argument("--id-column", default="deid_pat_id", help="Patient identifier column in the feature CSV.")
    parser.add_argument(
        "--label-id-column",
        default=None,
        help="Patient identifier column in the label CSV. Defaults to --id-column.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional binary label column in the label CSV. If omitted, the label CSV is treated as a positive-patient list.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        help="Optional feature columns to remove before training.",
    )
    parser.add_argument(
        "--scale-note-features",
        action="store_true",
        help="Apply z-score scaling to columns starting with the note prefix.",
    )
    parser.add_argument(
        "--note-prefix",
        default="Note_",
        help="Prefix used to identify note-derived features.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of models to run: xgboost mlp random_forest lasso_logistic logistic_regression",
    )
    parser.add_argument(
        "--sampling-strategies",
        nargs="*",
        default=None,
        help="Subset of sampling strategies to run: baseline undersampling oversampling",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument(
        "--bayes-iterations",
        type=int,
        default=100,
        help="Number of BayesSearchCV iterations for fold 1.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset = load_dataset(
        features_csv=args.features_csv,
        labels_csv=args.labels_csv,
        id_column=args.id_column,
        label_id_column=args.label_id_column,
        label_column=args.label_column,
        drop_columns=args.drop_columns,
        scale_note_features=args.scale_note_features,
        note_prefix=args.note_prefix,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    run_cross_validated_training(
        features=dataset.features,
        labels=dataset.labels,
        output_dir=output_dir,
        selected_models=args.models,
        selected_sampling=args.sampling_strategies,
        n_splits=args.n_splits,
        bayes_iterations=args.bayes_iterations,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
