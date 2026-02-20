import numpy as np
import pytest
from picasso.unmixing import select_representative_pixels

def test_saturation_exclusion():
    # 2 channels, 10x10.
    # All pixels = 100 except one saturated (255) in ch0.
    image = np.full((2, 10, 10), 100, dtype=np.uint8)
    image[0, 5, 5] = 255 # Saturated

    # Quantile=0.0 (all valid)
    # min_samples=0 to avoid forcing extra samples if logic tried to
    selected = select_representative_pixels(image, quantile=0.0, min_samples=0)

    # Should have 99 pixels.
    assert selected.shape[1] == 99

def test_quantile_selection():
    # 1 channel, 100 pixels. Values 0..99
    image = np.arange(100, dtype=np.uint8).reshape(1, 10, 10)

    # Quantile 0.90 -> top 10%. Values 90..99.
    # np.percentile(0..99, 90) = 89.1.
    # Values >= 89.1 are 90..99 (10 values).

    selected = select_representative_pixels(image, quantile=0.90, min_samples=0)
    # Should contain 10 values.
    assert selected.shape[1] == 10
    # Values are scaled 0-1. 90 -> 90/255
    assert np.min(selected) >= 89.0/255.0

def test_max_samples_int():
    # 1000 pixels. Top 100% selected.
    image = np.ones((1, 10, 100), dtype=np.uint8) * 100

    # Request 50 samples.
    selected = select_representative_pixels(image, quantile=0.0, max_samples=50, min_samples=0)
    assert selected.shape[1] == 50

def test_max_samples_float():
    # 100 pixels.
    image = np.ones((1, 10, 10), dtype=np.uint8) * 100

    # Request 0.5 ratio (50 pixels).
    # Note: 0.5 * 100 = 50.
    selected = select_representative_pixels(image, quantile=0.0, max_samples=0.5, min_samples=0)
    assert selected.shape[1] == 50

def test_min_samples_priority():
    # 1000 pixels. All same value.
    image = np.ones((1, 10, 100), dtype=np.uint8) * 100

    # max_samples=10, min_samples=50.
    # Logic: target = max(10, 50) = 50.
    selected = select_representative_pixels(image, quantile=0.0, max_samples=10, min_samples=50)
    assert selected.shape[1] == 50

def test_union_of_channels():
    # Ch0 high in first half. Ch1 high in second half.
    image = np.zeros((2, 10), dtype=np.uint8)
    image[0, :5] = 200
    image[1, 5:] = 200

    # Quantile 0.9.
    # Ch0: [200, 200, 200, 200, 200, 0, 0, 0, 0, 0]. Percentile 90 is 200.
    # So indices 0-4 are selected for Ch0.
    # Ch1: [0, 0, 0, 0, 0, 200, 200, 200, 200, 200]. Percentile 90 is 200.
    # So indices 5-9 are selected for Ch1.
    # Union should be all 10 pixels.

    selected = select_representative_pixels(image.reshape(2, 1, 10), quantile=0.9, min_samples=0)
    assert selected.shape[1] == 10
