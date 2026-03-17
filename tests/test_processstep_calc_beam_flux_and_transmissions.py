from __future__ import annotations

import numpy as np
import pytest

from processstep_calc_beam_flux_and_transmissions import dynamic_beam_analysis


def test_dynamic_beam_analysis_raises_for_mismatched_mask_shape():
    image = np.ones((5, 5), dtype=np.float32)
    beam_coverage_mask = np.ones((4, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape as imageData"):
        dynamic_beam_analysis(image, beam_coverage_mask=beam_coverage_mask)
