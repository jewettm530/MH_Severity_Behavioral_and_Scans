"""
Train behavioral cross-category ablation models and produce:
1. cross_category_metrics.csv -- all raw metrics and improvements.
2. cross_category_metric_improvements.html -- one readable output divided by target.
3. cross_category_summary.txt -- plain-language percentage explanations.
"""
import html
import json
import joblib
import numpy as np
import pandas as pd

from config import CROSS_CATEGORY_DIR, RESULTS_DIR, MODELS_DIR, PREDICTIONS_DIR
from model_utils import fit_and_score


def safe_relative_change(new, baseline):
    if pd.isna(new) or pd.isna(baseline) or baseline == 0:
        return np.nan
    return ((new - baseline) / abs(baseline)) * 100.0


def train_models():
    manifest = pd.read_csv(CROSS_CATEGORY_DIR / 'experiment_manifest.csv')
    results = []
    model_dir = MODELS_DIR / 'cross_category'
    prediction_dir = PREDICTIONS_DIR / 'cross_category'
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for _, row in manifest.iterrows():
        name = row['experiment_name']
        if row['status'] != 'ok':
            results.append({'experiment_name': name, 'status': 'skipped', 'error': row.get('error', '')})
            continue
        try:
            print(f'\n=== {name} ===')
            df = pd.read_csv(CROSS_CATEGORY_DIR / f'{name}.csv', low_memory=False)
            pipeline, metrics, predictions = fit_and_score(df)
            joblib.dump(pipeline, model_dir / f'{name}.joblib')
            predictions['experiment_name'] = name
            predictions.to_csv(prediction_dir / f'{name}.csv', index=False)

            result = {
                'experiment_name': name,
                'target_key': row['target_key'],
                'target_label': row['target_label'],
                'base_category': row['base_category'],
                'added_categories': row['added_categories'],
                'included_categories': row['included_categories'],
                'combination_label': row['combination_label'],
                **metrics,
                'status': 'ok',
                'error': '',
            }
            print(json.dumps(result, indent=2))
            results.append(result)
        except Exception as exc:
            print(f'ERROR: {exc}')
            results.append({'experiment_name': name, 'status': 'error', 'error': str(exc)})

    return pd.DataFrame(results)


def add_improvement_columns(metrics):
    metrics = metrics[metrics['status'] == 'ok'].copy()
    output_groups = []

    for target, group in metrics.groupby('target_key'):
        group = group.copy()
        baseline_rows = group[group['added_categories'].astype(str).isin(['[]', '', 'nan'])]
        if baseline_rows.empty:
            # The combination label with only one category is also the baseline.
            baseline_rows = group[group['combination_label'].str.count(r'\+') == 0]
        if baseline_rows.empty:
            raise ValueError(f'No baseline cross-category model found for {target}')
        baseline = baseline_rows.iloc[0]

        for metric in ['test_R2', 'test_MAE', 'test_RMSE', 'cv_R2_mean', 'cv_MAE_mean', 'cv_RMSE_mean']:
            group[f'baseline_{metric}'] = baseline[metric]

        group['delta_test_R2'] = group['test_R2'] - baseline['test_R2']
        group['relative_test_R2_change_percent'] = group['test_R2'].apply(
            lambda value: safe_relative_change(value, baseline['test_R2']))
        group['test_MAE_reduction_percent'] = (
            (baseline['test_MAE'] - group['test_MAE']) / baseline['test_MAE'] * 100.0
        )
        group['test_RMSE_reduction_percent'] = (
            (baseline['test_RMSE'] - group['test_RMSE']) / baseline['test_RMSE'] * 100.0
        )
        group['delta_cv_R2'] = group['cv_R2_mean'] - baseline['cv_R2_mean']
        group['cv_MAE_reduction_percent'] = (
            (baseline['cv_MAE_mean'] - group['cv_MAE_mean']) / baseline['cv_MAE_mean'] * 100.0
        )

        group['r2_table_value'] = group['delta_test_R2'].map(lambda x: f'{x:+.3f}')
        baseline_index = baseline.name
        if baseline_index in group.index:
            group.loc[baseline_index, 'r2_table_value'] = f"{baseline['test_R2']:.3f}"

        output_groups.append(group)

    return pd.concat(output_groups, ignore_index=True)


def html_table(metrics):
    sections = []
    for target, group in metrics.groupby('target_key', sort=False):
        group = group.sort_values(['added_categories', 'combination_label'])
        rows = []
        for _, row in group.iterrows():
            rows.append(
                '<tr>'
                f"<td>{html.escape(str(row['combination_label']))}</td>"
                f"<td>{row['r2_table_value']}</td>"
                f"<td>{row['test_R2']:.3f}</td>"
                f"<td>{row['delta_test_R2']:+.3f}</td>"
                f"<td>{row['test_MAE']:.3f}</td>"
                f"<td>{row['test_MAE_reduction_percent']:+.1f}%</td>"
                f"<td>{row['test_RMSE']:.3f}</td>"
                f"<td>{row['test_RMSE_reduction_percent']:+.1f}%</td>"
                f"<td>{row['cv_R2_mean']:.3f} ± {row['cv_R2_std']:.3f}</td>"
                '</tr>'
            )
        sections.append(
            f"<h2>{html.escape(str(group.iloc[0]['target_label']))}</h2>"
            '<table><thead><tr>'
            '<th>Behavioral categories</th><th>Requested R² display</th>'
            '<th>Test R²</th><th>Δ Test R²</th><th>Test MAE</th>'
            '<th>MAE improvement</th><th>Test RMSE</th><th>RMSE improvement</th>'
            '<th>CV R² mean ± SD</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
        )

    document = '''<!doctype html><html><head><meta charset="utf-8">
<title>Cross-category metric improvements</title>
<style>
body{font-family:Arial,sans-serif;margin:30px;line-height:1.35;color:#222}
table{border-collapse:collapse;width:100%;margin-bottom:32px;font-size:13px}
th,td{border:1px solid #bbb;padding:7px;text-align:right}
th:first-child,td:first-child{text-align:left}th{background:#eee}
h1,h2{margin-top:28px}.note{background:#f4f4f4;padding:12px;border-left:4px solid #777}
</style></head><body>
<h1>Cross-category prediction improvements</h1>
<div class="note">The target category alone is the baseline. For added-category rows, the requested R² display shows the change from that baseline. Positive MAE/RMSE improvement percentages mean lower error.</div>
''' + ''.join(sections) + '</body></html>'
    return document


def describe_change(row, baseline):
    added = str(row['combination_label'])
    delta_r2 = row['delta_test_R2']
    mae_pct = row['test_MAE_reduction_percent']
    rmse_pct = row['test_RMSE_reduction_percent']

    if delta_r2 > 0:
        direction = f'increased test R² by {delta_r2:.3f}'
    elif delta_r2 < 0:
        direction = f'decreased test R² by {abs(delta_r2):.3f}'
    else:
        direction = 'did not change test R²'

    if baseline['test_R2'] > 0:
        relative = safe_relative_change(row['test_R2'], baseline['test_R2'])
        relative_text = f' ({relative:+.1f}% relative R² change)'
    else:
        relative_text = ''

    return (
        f'- {added}: {direction}{relative_text}; '
        f'MAE changed by {mae_pct:+.1f}% and RMSE changed by {rmse_pct:+.1f}% '
        f'(positive percentages mean improved/lower error).'
    )


def narrative(metrics):
    lines = [
        'CROSS-CATEGORY PREDICTION SUMMARY',
        '=================================',
        '',
        'Each target starts with questionnaires from its own symptom category. '
        'Other categories are then added while the model, participant cohort, and split remain fixed.',
        '',
    ]

    for target, group in metrics.groupby('target_key', sort=False):
        baseline_rows = group[group['added_categories'].astype(str).isin(['[]', '', 'nan'])]
        baseline = baseline_rows.iloc[0] if not baseline_rows.empty else group.loc[group['delta_test_R2'].abs().idxmin()]
        lines.extend([
            str(group.iloc[0]['target_label']).upper(),
            '-' * len(str(group.iloc[0]['target_label'])),
            f"Baseline ({baseline['combination_label']}): test R²={baseline['test_R2']:.3f}, "
            f"MAE={baseline['test_MAE']:.3f}, RMSE={baseline['test_RMSE']:.3f}.",
            '',
        ])

        # Explain each single outside category separately.
        single_additions = group[group['added_categories'].astype(str).str.count(',') == 0]
        single_additions = single_additions[single_additions['delta_test_R2'].abs() > 1e-12]
        single_additions = single_additions.sort_values('delta_test_R2', ascending=False)
        lines.append('Individual category additions:')
        for _, row in single_additions.iterrows():
            lines.append(describe_change(row, baseline))

        best = group.sort_values(['test_R2', 'test_MAE'], ascending=[False, True]).iloc[0]
        lines.extend([
            '',
            f"Best combination: {best['combination_label']} with test R²={best['test_R2']:.3f}. "
            f"Compared with the baseline, R² changed by {best['delta_test_R2']:+.3f}, "
            f"MAE improved by {best['test_MAE_reduction_percent']:+.1f}%, and "
            f"RMSE improved by {best['test_RMSE_reduction_percent']:+.1f}%.",
            '',
        ])

    lines.extend([
        'INTERPRETATION NOTE',
        '-------------------',
        'A positive change on one held-out test split is preliminary. Give the most weight to additions '
        'that improve both held-out error and repeated cross-validation performance. Negative or highly '
        'variable cross-validation R² indicates that the apparent improvement may not generalize.',
    ])
    return '\n'.join(lines)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = train_models()
    raw.to_csv(RESULTS_DIR / 'cross_category_raw_metrics.csv', index=False)

    metrics = add_improvement_columns(raw)
    metrics.to_csv(RESULTS_DIR / 'cross_category_metrics.csv', index=False)

    html_output = html_table(metrics)
    (RESULTS_DIR / 'cross_category_metric_improvements.html').write_text(html_output, encoding='utf-8')

    text_output = narrative(metrics)
    (RESULTS_DIR / 'cross_category_summary.txt').write_text(text_output, encoding='utf-8')

    print('\nSaved:')
    print(' -', RESULTS_DIR / 'cross_category_metrics.csv')
    print(' -', RESULTS_DIR / 'cross_category_metric_improvements.html')
    print(' -', RESULTS_DIR / 'cross_category_summary.txt')


if __name__ == '__main__':
    main()
