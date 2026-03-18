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


def test_remove_small_objects_compat_falls_back_to_old_keyword(monkeypatch):
    calls: list[tuple[str, int]] = []

    def fake_remove_small_objects(mask, **kwargs):
        if "max_size" in kwargs:
            raise TypeError("remove_small_objects() got an unexpected keyword argument 'max_size'")
        calls.append(("min_size", kwargs["min_size"]))
        return mask

    monkeypatch.setattr(utilities.morphology, "remove_small_objects", fake_remove_small_objects)

    mask = np.ones((3, 3), dtype=bool)
    result = utilities.remove_small_objects_compat(mask, cleanup_size=20)

    assert calls == [("min_size", 20)]
    assert result is mask


def test_remove_small_holes_compat_falls_back_to_old_keyword(monkeypatch):
    calls: list[tuple[str, int]] = []

    def fake_remove_small_holes(mask, **kwargs):
        if "max_size" in kwargs:
            raise TypeError("remove_small_holes() got an unexpected keyword argument 'max_size'")
        calls.append(("area_threshold", kwargs["area_threshold"]))
        return mask

    monkeypatch.setattr(utilities.morphology, "remove_small_holes", fake_remove_small_holes)

    mask = np.ones((3, 3), dtype=bool)
    result = utilities.remove_small_holes_compat(mask, cleanup_size=20)

    assert calls == [("area_threshold", 20)]
    assert result is mask


def test_get_weighted_centroid_compat_supports_both_attribute_names():
    class NewRegion:
        centroid_weighted = (1.0, 2.0)

    class OldRegion:
        weighted_centroid = (3.0, 4.0)

    assert utilities.get_weighted_centroid_compat(NewRegion()) == (1.0, 2.0)
    assert utilities.get_weighted_centroid_compat(OldRegion()) == (3.0, 4.0)
