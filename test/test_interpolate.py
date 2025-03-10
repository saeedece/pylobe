import numpy as np
from pylobe.pattern import linear_array_factor
from pylobe.utils import (
    FullPattern,
    HorizontalSlice,
    VerticalSlice,
    build_equiangular_grid,
)
from pylobe.interpolate import (
    summing,
    bilinear,
    weighted_bilinear,
    horizontal_projection,
    exponential,
)
from pylobe.transform import (
    extract_phi_plane,
    extract_theta_cone,
    linear_to_db,
)


PATTERN = linear_to_db(
    np.square(
        np.abs(
            linear_array_factor(
                build_equiangular_grid()[1],
                kd=np.complex64(1.0),
                beta=np.zeros(1),
                num_elements=5,
            )
        )
    ),
    clip_value=-100,
)
VERTICAL_SLICE = extract_phi_plane(PATTERN, 0, endpoint=False).squeeze()
HORIZONTAL_SLICE = extract_theta_cone(PATTERN, 0, endpoint=False).squeeze()


def naive_summing(vertical_slice: VerticalSlice, horizontal_slice: HorizontalSlice):
    theta_resolution = (vertical_slice.size // 2) + 1
    phi_resolution = horizontal_slice.size
    pattern = np.empty((theta_resolution, phi_resolution))

    for i in range(theta_resolution):
        for j in range(phi_resolution):
            pattern[i, j] = 0.5 * (vertical_slice[i] + horizontal_slice[j])

    return pattern


def naive_bilinear(
    vertical_slice: VerticalSlice,
    horizontal_slice: HorizontalSlice,
) -> FullPattern:
    theta_resolution = (vertical_slice.size // 2) + 1
    phi_resolution = horizontal_slice.size
    G = np.empty((theta_resolution, phi_resolution))

    for i in range(theta_resolution):
        for j in range(phi_resolution):
            if i == 0:
                G[i, j] = vertical_slice[0]
                continue
            elif i == theta_resolution - 1:
                G[i, j] = vertical_slice[theta_resolution - 1]
                continue

            theta = (np.pi * i) / (theta_resolution - 1)
            phi = (2 * np.pi * j) / phi_resolution

            phi1 = phi if phi <= np.pi else 2 * np.pi - phi
            phi2 = np.pi - phi if phi <= np.pi else phi - np.pi

            Gphi1 = vertical_slice[i]
            Gphi2 = vertical_slice[2 * (theta_resolution - 1) - i]

            theta1 = theta if theta <= np.pi / 2 else np.pi - theta
            theta2 = np.pi / 2 - theta if theta <= np.pi / 2 else theta - np.pi / 2

            Gtheta1 = (
                vertical_slice[0]
                if theta <= np.pi / 2
                else vertical_slice[theta_resolution - 1]
            )
            Gtheta2 = horizontal_slice[j]

            num1 = phi1 * Gphi2 + phi2 * Gphi1
            num2 = theta1 * Gtheta2 + theta2 * Gtheta1
            den1 = phi1 + phi2
            den2 = theta1 + theta2

            G[i, j] = (num1 + num2) / (den1 + den2)

    return G


def naive_weighted_bilinear(
    vertical_slice: VerticalSlice,
    horizontal_slice: HorizontalSlice,
) -> FullPattern:
    theta_resolution = (vertical_slice.size // 2) + 1
    phi_resolution = horizontal_slice.size
    G = np.empty((theta_resolution, phi_resolution))

    for i in range(theta_resolution):
        for j in range(phi_resolution):
            if i == 0:
                G[i, j] = vertical_slice[0]
                continue
            elif i == theta_resolution - 1:
                G[i, j] = vertical_slice[theta_resolution - 1]
                continue
            elif i == (theta_resolution - 1) // 2:
                if j == 0:
                    G[i, j] = horizontal_slice[0]
                    continue
                elif j == phi_resolution // 2:
                    G[i, j] = horizontal_slice[phi_resolution // 2]
                    continue

            theta = (np.pi * i) / (theta_resolution - 1)
            phi = (2 * np.pi * j) / phi_resolution

            phi1 = phi if phi <= np.pi else 2 * np.pi - phi
            phi2 = np.pi - phi if phi <= np.pi else phi - np.pi
            assert phi1 + phi2 == np.pi

            Gphi1 = vertical_slice[i]
            Gphi2 = vertical_slice[2 * (theta_resolution - 1) - i]

            theta1 = theta if theta <= np.pi / 2 else np.pi - theta
            theta2 = np.pi / 2 - theta if theta <= np.pi / 2 else theta - np.pi / 2

            Gtheta1 = (
                vertical_slice[0]
                if theta <= np.pi / 2
                else vertical_slice[theta_resolution - 1]
            )
            Gtheta2 = horizontal_slice[j]

            Wphi = (phi1 * phi2) / ((phi1 + phi2) ** 2)
            Wthe = (theta1 * theta2) / ((theta1 + theta2) ** 2)
            num1 = (phi1 * Gphi2 + phi2 * Gphi1) * Wthe
            num2 = (theta1 * Gtheta2 + theta2 * Gtheta1) * Wphi
            den1 = (phi1 + phi2) * Wthe
            den2 = (theta1 + theta2) * Wphi
            G[i, j] = (num1 + num2) / (den1 + den2)

    return G


def naive_horizontal_projection(
    vertical_slice: VerticalSlice,
    horizontal_slice: HorizontalSlice,
) -> FullPattern:
    theta_resolution = (vertical_slice.size // 2) + 1
    phi_resolution = horizontal_slice.size
    G = np.empty((theta_resolution, phi_resolution))

    for i in range(theta_resolution):
        for j in range(phi_resolution):
            phi = (2 * np.pi * j) / phi_resolution

            phi1 = phi if phi <= np.pi else 2 * np.pi - phi
            phi2 = np.pi - phi if phi <= np.pi else phi - np.pi
            assert phi1 + phi2 == np.pi

            if i == 0:
                cor1 = (phi2 / np.pi) * (horizontal_slice[0] - vertical_slice[0])
                cor2 = (phi1 / np.pi) * (
                    horizontal_slice[phi_resolution // 2] - vertical_slice[0]
                )
            elif i == theta_resolution - 1:
                cor1 = (phi2 / np.pi) * (
                    horizontal_slice[0] - vertical_slice[theta_resolution - 1]
                )
                cor2 = (phi1 / np.pi) * (
                    horizontal_slice[phi_resolution // 2]
                    - vertical_slice[theta_resolution - 1]
                )
            else:
                cor1 = (phi2 / np.pi) * (horizontal_slice[0] - vertical_slice[i])
                cor2 = (phi1 / np.pi) * (
                    horizontal_slice[phi_resolution // 2]
                    - vertical_slice[2 * (theta_resolution - 1) - i]
                )

            G[i, j] = horizontal_slice[j] - (cor1 + cor2)

    return G


class TestInterpolator:
    def test_summing(self):
        vectorized_pattern = summing(VERTICAL_SLICE, HORIZONTAL_SLICE)
        naive_pattern = naive_summing(VERTICAL_SLICE, HORIZONTAL_SLICE)
        assert np.allclose(vectorized_pattern, naive_pattern)
        assert np.allclose(vectorized_pattern, PATTERN, rtol=1e0)

    def test_bilinear(self):
        vectorized_pattern = bilinear(VERTICAL_SLICE, HORIZONTAL_SLICE)
        naive_pattern = naive_bilinear(VERTICAL_SLICE, HORIZONTAL_SLICE)
        assert np.allclose(vectorized_pattern, naive_pattern)
        assert np.allclose(vectorized_pattern, PATTERN, rtol=1e0)

    def test_weighted_bilinear(self):
        vectorized_pattern = weighted_bilinear(VERTICAL_SLICE, HORIZONTAL_SLICE)
        naive_pattern = naive_weighted_bilinear(VERTICAL_SLICE, HORIZONTAL_SLICE)
        assert np.allclose(vectorized_pattern, naive_pattern)
        assert np.allclose(vectorized_pattern, PATTERN, rtol=1e0)

    def test_horizontal_projection(self):
        vectorized_pattern = horizontal_projection(VERTICAL_SLICE, HORIZONTAL_SLICE)
        naive_pattern = naive_horizontal_projection(VERTICAL_SLICE, HORIZONTAL_SLICE)
        assert np.allclose(vectorized_pattern, naive_pattern)
        assert np.allclose(vectorized_pattern, PATTERN)

    def test_exponential(self):
        vectorized_pattern = exponential(VERTICAL_SLICE, HORIZONTAL_SLICE)
        assert np.allclose(vectorized_pattern, PATTERN - np.max(PATTERN), rtol=1e-6)
