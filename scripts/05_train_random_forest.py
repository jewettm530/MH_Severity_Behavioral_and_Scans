"""Train the 28 core modality and behavioral-scope Random Forest models."""
import json
import joblib
import pandas as pd

from config import EXPERIMENT_DIR, RESULTS_DIR, MODELS_DIR, PREDICTIONS_DIR
from model_utils import fit_and_score


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(EXPERIMENT_DIR / 'experiment_manifest.csv')
    results = []

    for _, row in manifest.iterrows():
        name = row['experiment_name']
        if row['status'] != 'ok':
            results.append({'experiment_name': name, 'status': 'skipped', 'error': row.get('error', '')})
            continue
        try:
            print(f'\n=== {name} ===')
            df = pd.read_csv(EXPERIMENT_DIR / f'{name}.csv', low_memory=False)
            pipeline, metrics, predictions = fit_and_score(df)
            joblib.dump(pipeline, MODELS_DIR / f'{name}.joblib')
            predictions['experiment_name'] = name
            predictions.to_csv(PREDICTIONS_DIR / f'{name}.csv', index=False)

            result = {
                'experiment_name': name,
                'target_key': row['target_key'],
                'target_label': row['target_label'],
                'input_type': row['input_type'],
                'behavior_scope': row['behavior_scope'],
                **metrics,
                'status': 'ok',
                'error': '',
            }
            print(json.dumps(result, indent=2))
            results.append(result)
        except Exception as exc:
            print(f'ERROR: {exc}')
            results.append({'experiment_name': name, 'status': 'error', 'error': str(exc)})

    output = pd.DataFrame(results)
    output.to_csv(RESULTS_DIR / 'model_metrics_comparison.csv', index=False)
    print('\nSaved:', RESULTS_DIR / 'model_metrics_comparison.csv')


if __name__ == '__main__':
    main()
