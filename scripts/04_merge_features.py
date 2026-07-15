"""Build core modality/scope datasets and cross-category datasets."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    CLEAN_BEHAVIOR_DIR, EXPERIMENT_DIR, CROSS_CATEGORY_DIR, OUTPUT_DIR,
    SUBJECT_ID_COLUMN, TARGETS, MAX_MISSING_FRACTION, PHENOTYPE_DIR,
)
from experiments import CORE_EXPERIMENTS, CROSS_CATEGORY_EXPERIMENTS


def normalize_id(value):
    if pd.isna(value):
        return np.nan
    return str(value).replace('sub-', '').replace('_', '').strip()


def load_imaging():
    path = OUTPUT_DIR / 'imaging_features.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. The user indicated this file already exists.')
    imaging = pd.read_csv(path, low_memory=False)
    id_col = 'participant_id' if 'participant_id' in imaging.columns else SUBJECT_ID_COLUMN
    imaging[SUBJECT_ID_COLUMN] = imaging[id_col].apply(normalize_id)
    imaging = imaging.drop(columns=[id_col], errors='ignore').drop_duplicates(SUBJECT_ID_COLUMN)
    imaging = imaging.rename(columns={c: f'imaging__{c}' for c in imaging.columns if c != SUBJECT_ID_COLUMN})
    return imaging


def load_demographics():
    path = OUTPUT_DIR / 'demographics_once.csv'
    if not path.exists():
        return pd.DataFrame(columns=[SUBJECT_ID_COLUMN])
    return pd.read_csv(path, low_memory=False).drop_duplicates(SUBJECT_ID_COLUMN)


def load_target(exp):
    path = CLEAN_BEHAVIOR_DIR / exp['target_file']
    raw = pd.read_csv(path, low_memory=False)
    prefixed = f"{Path(exp['target_file']).stem}__{exp['target_column']}"
    if prefixed not in raw.columns:
        # Target totals may have been excluded by a conservative column rule.
        original = pd.read_csv(PHENOTYPE_DIR / exp['target_file'], low_memory=False)
        original.columns = original.columns.str.strip()
        id_col = next(c for c in [SUBJECT_ID_COLUMN, 'src_subject_id', 'participant_id'] if c in original.columns)
        original[SUBJECT_ID_COLUMN] = original[id_col].apply(normalize_id)
        target = original[[SUBJECT_ID_COLUMN, exp['target_column']]].copy()
    else:
        target = raw[[SUBJECT_ID_COLUMN, prefixed]].rename(columns={prefixed: exp['target_column']})

    target['target'] = pd.to_numeric(target[exp['target_column']], errors='coerce')
    target = target[target['target'].between(exp['valid_min'], exp['valid_max'], inclusive='both')]
    return target[[SUBJECT_ID_COLUMN, 'target']].drop_duplicates(SUBJECT_ID_COLUMN)


def merge_behavior_files(files):
    merged = None
    for filename in files:
        path = CLEAN_BEHAVIOR_DIR / filename
        if not path.exists():
            print(f'WARNING: cleaned predictor file missing: {filename}')
            continue
        current = pd.read_csv(path, low_memory=False).drop_duplicates(SUBJECT_ID_COLUMN)
        if current.shape[1] <= 1:
            continue
        merged = current if merged is None else merged.merge(current, on=SUBJECT_ID_COLUMN, how='outer')
    return merged if merged is not None else pd.DataFrame(columns=[SUBJECT_ID_COLUMN])


def finalize_dataset(dataset):
    protected = {SUBJECT_ID_COLUMN, 'target'}
    predictor_cols = [c for c in dataset.columns if c not in protected]
    missing = dataset[predictor_cols].isna().mean() if predictor_cols else pd.Series(dtype=float)
    drop_missing = [c for c in predictor_cols if missing.get(c, 1.0) >= MAX_MISSING_FRACTION]
    dataset = dataset.drop(columns=drop_missing, errors='ignore')
    drop_constant = [c for c in dataset.columns if c not in protected and dataset[c].nunique(dropna=True) <= 1]
    dataset = dataset.drop(columns=drop_constant, errors='ignore')
    return dataset


def build_dataset(exp, imaging, demographics, include_imaging, include_behavior):
    # Inner join to imaging for every experiment so modalities use the same
    # participant cohort for a given target.
    dataset = load_target(exp).merge(imaging[[SUBJECT_ID_COLUMN]], on=SUBJECT_ID_COLUMN, how='inner')

    if include_behavior:
        behavior = merge_behavior_files(exp['behavior_files'])
        dataset = dataset.merge(behavior, on=SUBJECT_ID_COLUMN, how='left')
        if not demographics.empty:
            dataset = dataset.merge(demographics, on=SUBJECT_ID_COLUMN, how='left')

    if include_imaging:
        dataset = dataset.merge(imaging, on=SUBJECT_ID_COLUMN, how='left')

    dataset = finalize_dataset(dataset)
    if len(dataset) == 0 or dataset.shape[1] <= 2:
        raise ValueError(f'No usable data: rows={len(dataset)}, columns={dataset.shape[1]}')
    return dataset


def save_manifest(experiments, directory, imaging, demographics, cross=False):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp in experiments:
        name = exp['experiment_name']
        try:
            dataset = build_dataset(
                exp, imaging, demographics,
                include_imaging=False if cross else exp['include_imaging'],
                include_behavior=True if cross else exp['include_behavior'],
            )
            path = directory / f'{name}.csv'
            dataset.to_csv(path, index=False)
            row = {
                **{k: exp.get(k) for k in exp},
                'behavior_files': json.dumps(exp.get('behavior_files', [])),
                'n_rows': len(dataset),
                'n_predictors': dataset.shape[1] - 2,
                'status': 'ok',
                'error': '',
                'dataset_path': str(path),
            }
            print(f'Saved {name}: rows={len(dataset)}, predictors={dataset.shape[1]-2}')
        except Exception as exc:
            print(f'ERROR building {name}: {exc}')
            row = {**exp, 'behavior_files': json.dumps(exp.get('behavior_files', [])), 'n_rows': 0,
                   'n_predictors': 0, 'status': 'error', 'error': str(exc), 'dataset_path': ''}
        rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(directory / 'experiment_manifest.csv', index=False)
    return manifest


def main():
    imaging = load_imaging()
    demographics = load_demographics()
    print(f'Imaging participants: {len(imaging)}')

    core = save_manifest(CORE_EXPERIMENTS, EXPERIMENT_DIR, imaging, demographics, cross=False)
    cross = save_manifest(CROSS_CATEGORY_EXPERIMENTS, CROSS_CATEGORY_DIR, imaging, demographics, cross=True)
    print(f'\nCore datasets built: {(core.status == "ok").sum()}/{len(core)}')
    print(f'Cross-category datasets built: {(cross.status == "ok").sum()}/{len(cross)}')


if __name__ == '__main__':
    main()
