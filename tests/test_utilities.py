from __future__ import annotations

import logging

import numpy as np
import pytest

import utilities


def test_reduce_extra_image_dimensions_rejects_invalid_method():
    image = np.ones((2, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="method must be one of"):
        utilities.reduce_extra_image_dimensions(image, method=max)


def test_label_main_feature_raises_when_no_beam_found():
    logger = logging.getLogger("test-utilities")
    image = np.zeros((10, 10), dtype=float)

    with pytest.warns(UserWarning, match="Input image is entirely zero"):
        with pytest.raises(ValueError, match="No beam found"):
            utilities.label_main_feature(image, logger)
