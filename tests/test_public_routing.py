import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='module')
def functions():
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks/WattMosaic.ipynb').read_text(encoding='utf-8'))
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'public_notebook', 'exec'), namespace)
    return namespace


def synthetic_frame(masks):
    return pd.DataFrame({
        sensor: [np.nan if mask & (1 << bit) else float(bit + 1) for mask in masks]
        for bit, sensor in enumerate(['temperature', 'humidity', 'occupancy', 'previous_usage'])
    }).assign(sample_id=np.arange(len(masks)), index_marker=np.arange(len(masks)) * 13)


def test_all_16_masks_and_repeated_patterns(functions):
    masks = [15, 0, 7, 1, 8, 4, 2, 3, 5, 6, 9, 10, 11, 12, 13, 14, 1]
    frame = synthetic_frame(masks)
    np.testing.assert_array_equal(functions['missingness_mask'](frame), masks)


def test_callbacks_follow_pattern_and_preserve_input_order(functions):
    masks = [15, 0, 7, 1, 8, 4, 2, 3, 5, 6, 9, 10, 11, 12, 13, 14, 1]
    frame = synthetic_frame(masks)
    frame.index = frame['index_marker']
    calls = {'core': [], 'specialist': []}

    def core(subset, mask):
        calls['core'].append((mask, subset['sample_id'].tolist()))
        return subset['sample_id'].to_numpy() * 2 + 10

    def specialist(subset, mask):
        calls['specialist'].append((mask, subset['sample_id'].tolist()))
        return np.full(len(subset), mask)

    output = functions['route_prediction'](core, specialist, frame)
    np.testing.assert_array_equal(output, np.arange(len(masks)) * 2 + 10 + np.array(masks))
    assert [mask for mask, _ in calls['core']] == list(range(16))
    assert [mask for mask, _ in calls['specialist']] == list(range(1, 16))
    assert dict(calls['specialist'])[1] == [3, 16]


def test_negative_predictions_are_clamped(functions):
    frame = synthetic_frame(list(range(16)))
    output = functions['route_prediction'](lambda subset, mask: np.full(len(subset), -5),
        lambda subset, mask: np.zeros(len(subset)), frame)
    np.testing.assert_array_equal(output, np.zeros(16))


def test_empty_input_does_not_call_models(functions):
    def forbidden(*args):
        raise AssertionError('Empty input must not invoke a model')
    output = functions['route_prediction'](forbidden, forbidden, synthetic_frame([]))
    assert output.size == 0
