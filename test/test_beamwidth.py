import numpy as np

from pylobe.utils import build_equiangular_grid, Axis, SphericalAxis
from pylobe.transform import linear_to_db, rotate_pattern_bilinear, extract_phi_plane
from pylobe.pattern import linear_array_factor
from pylobe.beamwidth import compute as compute_beamwidth


class TestBeamwidth:
    def test_compute_beamwidth(self):
        pattern = np.square(
            np.abs(
                linear_array_factor(
                    build_equiangular_grid()[1],
                    kd=np.complex128(np.pi),
                    beta=np.zeros(1),
                    num_elements=5,
                )
            )
        )
        pattern = linear_to_db(pattern)
        slice_xz = extract_phi_plane(pattern, 0, endpoint=False).squeeze()
        slice_yz = extract_phi_plane(pattern, 0, endpoint=False).squeeze()

        beamwidth_xz = compute_beamwidth(slice_xz, SphericalAxis.Theta, threshold=1.0)
        beamwidth_yz = compute_beamwidth(slice_yz, SphericalAxis.Theta, threshold=1.0)

        assert beamwidth_xz == beamwidth_yz
