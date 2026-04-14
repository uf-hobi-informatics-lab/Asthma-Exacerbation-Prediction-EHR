from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    ids: pd.Series
    features: pd.DataFrame
    labels: pd.Series


def _sanitize_feature_names(columns: Iterable[str]) -> list[str]:
    sanitized = []
    for column in columns:
        new_name = str(column)
        new_name = new_name.replace("[", "(")
        new_name = new_name.replace("]", ")")
        new_name = new_name.replace("<", "")
        sanitized.append(new_name)
    return sanitized


def _attach_labels(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    id_column: str,
    label_id_column: str,
    label_column: str | None,
) -> pd.DataFrame:
    merged = features_df.copy()

    if label_column:
        label_frame = labels_df[[label_id_column, label_column]].drop_duplicates()
        label_frame = label_frame.rename(
            columns={
                label_id_column: id_column,
                label_column: "Label",
            }
        )
        merged = merged.merge(label_frame, on=id_column, how="left")
        merged["Label"] = merged["Label"].fillna(0).astype(int)
        return merged

    positive_ids = set(labels_df[label_id_column].dropna())
    merged["Label"] = merged[id_column].isin(positive_ids).astype(int)
    return merged


def load_dataset(
    features_csv: str | Path,
    labels_csv: str | Path,
    id_column: str = "deid_pat_id",
    label_id_column: str | None = None,
    label_column: str | None = None,
    drop_columns: list[str] | None = None,
    scale_note_features: bool = False,
    note_prefix: str = "Note_",
) -> DatasetBundle:
    features_path = Path(features_csv)
    labels_path = Path(labels_csv)

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    if id_column not in features_df.columns:
        raise ValueError(f"Missing id column '{id_column}' in {features_path}")

    label_id_column = label_id_column or id_column
    if label_id_column not in labels_df.columns:
        raise ValueError(f"Missing label id column '{label_id_column}' in {labels_path}")

    data = _attach_labels(
        features_df=features_df,
        labels_df=labels_df,
        id_column=id_column,
        label_id_column=label_id_column,
        label_column=label_column,
    )

    drop_columns = drop_columns or []
    removable = [column for column in drop_columns if column in data.columns]
    if removable:
        data = data.drop(columns=removable)

    note_columns = [column for column in data.columns if column.startswith(note_prefix)]
    if scale_note_features and note_columns:
        scaler = StandardScaler()
        data[note_columns] = scaler.fit_transform(data[note_columns])

    data = data.sort_values(by=id_column).reset_index(drop=True)

    feature_columns = [
        column
        for column in data.columns
        if column not in {id_column, "Label"}
    ]
    features = data[feature_columns].astype(float).copy()
    features.columns = _sanitize_feature_names(features.columns)

    return DatasetBundle(
        ids=data[id_column].copy(),
        features=features,
        labels=data["Label"].astype(int).copy(),
    )
