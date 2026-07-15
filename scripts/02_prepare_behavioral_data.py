"""Clean phenotype files individually and remove redundant demographics."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

from config import (
    PHENOTYPE_DIR, DEMOS_FILE, CLEAN_BEHAVIOR_DIR, OUTPUT_DIR,
    SUBJECT_ID_COLUMN, MAX_MISSING_FRACTION, MISSING_CODES,
    ALL_BEHAVIOR_FILES, CATEGORY_FILES, TARGETS,
)


def normalize_id(value):
    if pd.isna(value):
        return np.nan
    return str(value).replace('sub-', '').replace('_', '').strip()


def read_csv(path: Path):
    return pd.read_csv(path, low_memory=False).replace(MISSING_CODES, np.nan)


def find_id_column(df):
    for col in [SUBJECT_ID_COLUMN, 'src_subject_id', 'participant_id', 'subject_id']:
        if col in df.columns:
            return col
    raise ValueError(f'No subject ID column found in columns: {list(df.columns)[:20]}')


def standardize_id(df):
    out = df.copy()
    id_col = find_id_column(out)
    if id_col != SUBJECT_ID_COLUMN:
        out = out.rename(columns={id_col: SUBJECT_ID_COLUMN})
    out[SUBJECT_ID_COLUMN] = out[SUBJECT_ID_COLUMN].apply(normalize_id)
    return out


def is_admin_column(col):
    c = str(col).strip().lower()
    exact = {
        'src_subject_id', 'interview_date', 'interview_age', 'age', 'sex',
        'site', 'group', 'collection_id', 'dataset_id', 'respondent',
        'relationship', 'rel_to_proband', 'species', 'level',
    }
    if c in exact:
        return True
    if any(x in c for x in [
        'time_first_click', 'time_last_click', 'time_page_submit',
        'time_click_count', 'tot_timing', '_timing', 'first_click',
        'last_click', 'page_submit',
    ]):
        return True
    if c.startswith('qc_') or '_qc' in c or 'num_nonresp' in c or 'num_items' in c:
        return True
    if c.endswith('_complete') or c.endswith('_completed_by'):
        return True
    if any(x in c for x in ['note', 'comment', 'description']):
        return True
    return False


def looks_item_level(col, stem):
    c = str(col).lower()
    short_stem = stem.lower().replace('01', '')
    return bool(
        re.search(r'(^|_)q\d+[a-z]?$', c)
        or re.fullmatch(r'[a-z]+\d+[a-z]?', c)
        or re.search(r'_\d+[a-z]?$', c)
        or (short_stem and re.fullmatch(rf'{re.escape(short_stem)}_?\d+[a-z]?', c))
    )


def looks_summary(col):
    c = str(col).lower()
    terms = [
        'total', 'tot', 'score', 'sum', 'subscale', 'scale', 'ave', 'avg',
        'mean', 'median', 'std', 'percent', 'rate', 'accuracy', 'reaction',
        'error', 'omission', 'commission', 'correct', 'interference',
        'severity', 'positive', 'negative', 'general', 'depression', 'depress',
        'anxiety', 'stress', 'panic', 'mania', 'anhedonia', 'rumination',
        'brooding', 'reflection', 'trauma', 'avoidance', 'attachment', 'impuls',
        'reward', 'inhibition', 'activation', 'support', 'neuroticism',
        'extraversion', 'openness', 'agreeableness', 'conscientiousness',
        'harm_avoidance', 'novelty', 'persistence', 'pleasure', 'risk',
        'state', 'trait', 'global', 'clinical', 'function', 'symptom',
    ]
    return any(term in c for term in terms)


def clean_scale_file(path):
    stem = path.stem
    df = standardize_id(read_csv(path))
    keep = [SUBJECT_ID_COLUMN]

    for col in df.columns:
        if col == SUBJECT_ID_COLUMN or is_admin_column(col):
            continue
        if looks_item_level(col, stem) and not looks_summary(col):
            continue
        if looks_summary(col):
            keep.append(col)

    out = df[keep].drop_duplicates(subset=[SUBJECT_ID_COLUMN]).copy()
    out = out.rename(columns={c: f'{stem}__{c}' for c in out.columns if c != SUBJECT_ID_COLUMN})

    # Remove columns that are mostly missing or constant within this scale.
    predictors = [c for c in out.columns if c != SUBJECT_ID_COLUMN]
    missing = out[predictors].isna().mean() if predictors else pd.Series(dtype=float)
    drop = [c for c in predictors if missing.get(c, 1.0) >= MAX_MISSING_FRACTION]
    out = out.drop(columns=drop, errors='ignore')
    constant = [c for c in out.columns if c != SUBJECT_ID_COLUMN and out[c].nunique(dropna=True) <= 1]
    out = out.drop(columns=constant, errors='ignore')
    return out


def load_demographics():
    if not DEMOS_FILE.exists():
        print('WARNING: demos.csv not found; demographics will be omitted.')
        return pd.DataFrame(columns=[SUBJECT_ID_COLUMN])

    demos = standardize_id(read_csv(DEMOS_FILE))
    rename_candidates = {
        'Age': 'demographic__age',
        'age': 'demographic__age',
        'sex': 'demographic__sex',
        'Site': 'demographic__site',
        'site': 'demographic__site',
        'Group': 'demographic__group',
        'group': 'demographic__group',
    }
    selected = [SUBJECT_ID_COLUMN]
    rename = {}
    for source, destination in rename_candidates.items():
        if source in demos.columns and destination not in rename.values():
            selected.append(source)
            rename[source] = destination

    demos = demos[selected].rename(columns=rename).drop_duplicates(subset=[SUBJECT_ID_COLUMN])
    if 'demographic__age' in demos.columns:
        demos['demographic__age'] = pd.to_numeric(demos['demographic__age'], errors='coerce')
        demos.loc[~demos['demographic__age'].between(18, 100), 'demographic__age'] = np.nan
    return demos


def required_files():
    files = set(ALL_BEHAVIOR_FILES)
    for category_files in CATEGORY_FILES.values():
        files.update(category_files)
    for target in TARGETS.values():
        files.add(target['file'])
    return sorted(files)


def main():
    CLEAN_BEHAVIOR_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for filename in required_files():
        path = PHENOTYPE_DIR / filename
        if not path.exists():
            print(f'WARNING: missing {filename}')
            manifest.append({'file': filename, 'status': 'missing', 'n_rows': 0, 'n_predictors': 0})
            continue
        try:
            cleaned = clean_scale_file(path)
            cleaned.to_csv(CLEAN_BEHAVIOR_DIR / filename, index=False)
            n_predictors = max(0, cleaned.shape[1] - 1)
            print(f'Cleaned {filename}: rows={len(cleaned)}, predictors={n_predictors}')
            manifest.append({'file': filename, 'status': 'ok', 'n_rows': len(cleaned), 'n_predictors': n_predictors})
        except Exception as exc:
            print(f'ERROR cleaning {filename}: {exc}')
            manifest.append({'file': filename, 'status': 'error', 'n_rows': 0, 'n_predictors': 0, 'error': str(exc)})

    demographics = load_demographics()
    demographics.to_csv(OUTPUT_DIR / 'demographics_once.csv', index=False)
    print(f'Saved demographics once: {demographics.shape}')

    with open(CLEAN_BEHAVIOR_DIR / 'cleaning_manifest.json', 'w') as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == '__main__':
    main()
