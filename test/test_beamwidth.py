import numpy as np
import numpy.typing as npt

from pylobe.utils import SphericalAxis
from pylobe.beamwidth import compute as compute_beamwidth


class TestBeamwidth:
    def test_compute_beamwidth(
        self,
        slice_xz: npt.NDArray[np.floating],
        slice_yz: npt.NDArray[np.floating],
    ):
        beamwidth_xz = compute_beamwidth(
            slice_xz.squeeze(),
            SphericalAxis.Theta,
            threshold=1.0,
        )
        beamwidth_yz = compute_beamwidth(
            slice_yz.squeeze(),
            SphericalAxis.Theta,
            threshold=1.0,
        )
        assert beamwidth_xz == beamwidth_yz
