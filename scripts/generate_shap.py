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
from asthma_exacerbation.shap_utils import generate_shap_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SHAP summary plots and ranked feature importance tables.",
    )
    parser.add_argument("--model-path", required=True, help="Path to a trained .joblib model.")
    parser.add_argument("--features-csv", required=True, help="Path to the feature table CSV.")
    parser.add_argument("--labels-csv", required=True, help="Path to the label CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for SHAP outputs.")
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
        help="Optional feature columns to remove before explanation.",
    )
    parser.add_argument(
        "--scale-note-features",
        action="store_true",
        help="Apply z-score scaling to columns starting with the note prefix.",
    )
    parser.add_argument("--note-prefix", default="Note_", help="Prefix used to identify note-derived features.")
    parser.add_argument("--top-k", type=int, default=15, help="Number of features to show.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of rows used for SHAP explanation.",
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

    figure_path, importance_csv = generate_shap_report(
        model_path=args.model_path,
        feature_matrix=dataset.features,
        output_dir=output_dir,
        top_k=args.top_k,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )

    with (output_dir / "shap_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "figure_path": str(figure_path),
                "importance_csv": str(importance_csv),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
