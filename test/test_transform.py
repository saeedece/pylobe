import numpy as np
import numpy.typing as npt

from pylobe.utils import Axis
from pylobe.transform import (
    extract_phi_plane,
    extract_theta_cone,
    rotate_pattern_bilinear,
    pad_spherical,
)


class TestTransform:
    def test_extract_phi(self, simple_grid: npt.NDArray[np.floating]):
        result = extract_phi_plane(simple_grid, 0, endpoint=False)
        expected = np.array(
            [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                95,
                85,
                75,
                65,
                55,
                45,
                35,
                25,
                15,
            ]
        ).astype(dtype=np.float32)
        assert np.allclose(result.squeeze(), expected)

    def test_extract_theta(self, simple_grid: npt.NDArray[np.floating]):
        result = extract_theta_cone(simple_grid, np.pi / 2, endpoint=False)
        expected = np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59]).astype(
            dtype=np.float32
        )
        assert np.allclose(result.squeeze(), expected)

    def test_pad_spherical(self, simple_grid: npt.NDArray[np.floating]):
        result = pad_spherical(simple_grid, pad_widths=[(2, 3), (3, 0)])
        expected = np.array(
            [
                [22, 23, 24, 25, 26, 27, 28, 29, 20, 21, 22, 23, 24],
                [12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14],
                [7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [17, 18, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [27, 28, 29, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                [37, 38, 39, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                [47, 48, 49, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                [57, 58, 59, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                [67, 68, 69, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                [77, 78, 79, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                [87, 88, 89, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                [97, 98, 99, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
                [107, 108, 109, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                [92, 93, 94, 95, 96, 97, 98, 99, 90, 91, 92, 93, 94],
                [82, 83, 84, 85, 86, 87, 88, 89, 80, 81, 82, 83, 84],
                [72, 73, 74, 75, 76, 77, 78, 79, 70, 71, 72, 73, 74],
            ]
        )
        assert np.allclose(result, expected)

    def test_rotate_bilinear(self, intensity: npt.NDArray[np.floating]):
        rotated_pattern = rotate_pattern_bilinear(
            intensity,
            angles=[np.pi / 2],
            axes=[Axis.Y],
        )
        rerotated_pattern = rotate_pattern_bilinear(
            rotated_pattern,
            angles=[-np.pi / 2],
            axes=[Axis.Y],
        )
        assert np.allclose(rerotated_pattern, intensity, rtol=1e-2)
