"""Explore sampling stability of assembly continuity thresholds.

Run from the repository root with PYTHONPATH=. and provide the downloaded
qualibact-v1.0 PASS, WARN, and FAIL tables for each requested species.
The analysis measures stability, not biological accuracy or clinical validity.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from qualibact.species_util import DEF_METRIC_LIST, apply_outlier_filter


def fit_bounds(frame):
    assigned = apply_outlier_filter(frame)
    retained = assigned.loc[assigned['anomaly'].eq(1)]
    return {
        'n50_lower': float(retained['N50'].quantile(0.005)),
        'contigs_upper': float(retained['number'].quantile(0.995)),
        'retained_fraction': len(retained) / len(frame),
    }


def excluded(frame, bounds):
    return frame['N50'].lt(bounds['n50_lower']) | frame['number'].gt(bounds['contigs_upper'])


def analyze(inputs, output, species, replicates, seed):
    output.mkdir(parents=True, exist_ok=True)
    bootstrap_rows, fold_rows, prediction_rows, summary_rows = [], [], [], []
    for species_index, name in enumerate(species):
        frame = pd.concat(
            [pd.read_csv(inputs / f'{name}_atb_{tier}.csv.gz') for tier in ('pass', 'warn', 'fail')],
            ignore_index=True,
        ).sort_values('sample').reset_index(drop=True)
        assert frame['sample'].is_unique
        assert np.isfinite(frame[DEF_METRIC_LIST].to_numpy(dtype=float)).all()
        full = fit_bounds(frame)
        rng = np.random.default_rng(seed + species_index)
        for replicate in range(replicates):
            indices = rng.integers(0, len(frame), size=len(frame))
            bounds = fit_bounds(frame.iloc[indices].reset_index(drop=True))
            bootstrap_rows.append({'species': name, 'replicate': replicate, **bounds})
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(frame)):
            bounds = fit_bounds(frame.iloc[train].reset_index(drop=True))
            held_out = frame.iloc[test]
            flags = excluded(held_out, bounds)
            fold_rows.append({
                'species': name, 'fold': fold, 'train_count': len(train),
                'test_count': len(test), 'excluded_count': int(flags.sum()),
                'excluded_fraction': float(flags.mean()), **bounds,
            })
            for sample, flag in zip(held_out['sample'], flags):
                prediction_rows.append({
                    'species': name, 'sample': sample, 'fold': fold,
                    'outside_continuity_bounds': bool(flag),
                })
        sample_bounds = pd.DataFrame(bootstrap_rows).query('species == @name')
        summary = {'species': name, 'assemblies': len(frame), **full}
        for metric in ('n50_lower', 'contigs_upper', 'retained_fraction'):
            summary[metric + '_bootstrap_p025'] = float(sample_bounds[metric].quantile(0.025))
            summary[metric + '_bootstrap_p975'] = float(sample_bounds[metric].quantile(0.975))
        summary_rows.append(summary)
        print(json.dumps(summary), flush=True)
    for filename, rows in [
        ('bootstrap_bounds.csv', bootstrap_rows), ('held_out_folds.csv', fold_rows),
        ('held_out_predictions.csv', prediction_rows), ('summary.csv', summary_rows),
    ]:
        pd.DataFrame(rows).to_csv(output / filename, index=False)
    (output / 'settings.json').write_text(json.dumps({
        'species': species, 'replicates': replicates, 'seed': seed,
        'isolation_forest_random_state': 42,
        'isolation_forest_features': DEF_METRIC_LIST,
        'quantiles': {'N50_lower': 0.005, 'contig_count_upper': 0.995},
        'folds': 5,
        'interpretation': 'Descriptive assembly resampling ranges, not lineage-aware confidence intervals.',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--species', nargs='+', required=True)
    parser.add_argument('--replicates', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261030)
    args = parser.parse_args()
    analyze(args.inputs, args.output, args.species, args.replicates, args.seed)
