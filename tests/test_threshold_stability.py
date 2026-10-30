import numpy as np
import pandas as pd
import pytest

from analysis.threshold_stability import excluded, fit_bounds


def test_continuity_bounds_are_one_sided_and_include_endpoints():
    frame = pd.DataFrame({
        'N50': [50000, 49999, 50001, 1000000, 50000],
        'number': [420, 419, 421, 1, 421],
    })
    bounds = {'n50_lower': 50000, 'contigs_upper': 420}
    assert excluded(frame, bounds).tolist() == [False, True, True, False, True]


def test_change_of_n50_units_preserves_selection():
    rng = np.random.default_rng(12)
    frame = pd.DataFrame({
        'total_length': rng.uniform(4000000, 5000000, 60),
        'GC_Content': rng.uniform(35, 40, 60),
        'N50': rng.uniform(50000, 150000, 60),
        'number': rng.integers(50, 200, 60),
        'longest': rng.uniform(200000, 800000, 60),
        'Completeness_Specific': rng.uniform(95, 100, 60),
        'Contamination': rng.uniform(0, 2, 60),
    })
    original = fit_bounds(frame)
    changed = fit_bounds(frame.assign(N50=frame['N50'] * 2))
    assert changed['n50_lower'] == pytest.approx(original['n50_lower'] * 2)
    assert changed['contigs_upper'] == original['contigs_upper']
    assert changed['retained_fraction'] == original['retained_fraction']
