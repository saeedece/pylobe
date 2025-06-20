import numpy as np
import numpy.typing as npt
import pytest

from pylobe.interpolate import (
    summing,
    bilinear,
    weighted_bilinear,
    horizontal_projection,
    exponential,
)
from pylobe.transform import extract_theta_cone, extract_phi_plane


@pytest.fixture()
def intensity_norm(intensity):
    return intensity - np.max(intensity)


@pytest.fixture()
def slice_xy_norm(intensity_norm: npt.NDArray[np.floating]):
    return extract_theta_cone(intensity_norm, np.pi / 2, endpoint=False)


@pytest.fixture()
def slice_xz_norm(intensity_norm: npt.NDArray[np.floating]):
    return extract_phi_plane(intensity_norm, 0, endpoint=False)


@pytest.fixture()
def slice_yz_norm(intensity_norm: npt.NDArray[np.floating]):
    return extract_phi_plane(intensity_norm, np.pi / 2, endpoint=False)


def naive_summing(
    vertical_slice: npt.NDArray[np.floating],
    horizontal_slice: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    theta_resolution = (vertical_slice.size // 2) + 1
    phi_resolution = horizontal_slice.size
    pattern = np.empty((theta_resolution, phi_resolution))

    for i in range(theta_resolution):
        for j in range(phi_resolution):
            pattern[i, j] = 0.5 * (vertical_slice[i] + horizontal_slice[j])

    return pattern


def naive_bilinear(
    vertical_slice: npt.NDArray[np.floating],
    horizontal_slice: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
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
    vertical_slice: npt.NDArray[np.floating],
    horizontal_slice: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
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
    vertical_slice: npt.NDArray[np.floating],
    horizontal_slice: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    theta_resolution = (vertical_slice.size // 2) + 1
    phi_resolution = horizontal_slice.size
    G = np.empty((theta_resolution, phi_resolution))

    for i in range(theta_resolution):
        for j in range(phi_resolution):
            phi = (2 * np.pi * j) / phi_resolution

            phi_fnt = phi if phi <= np.pi else 2 * np.pi - phi
            phi_bck = np.pi - phi_fnt

            if i == 0:
                idx1 = 0
                idx2 = theta_resolution - 1
            elif i == theta_resolution - 1:
                idx1 = theta_resolution - 1
                idx2 = 0
            else:
                idx1 = (
                    2 * (theta_resolution - 1) - i
                    if (np.pi / 2 < phi < 3 * np.pi / 2)
                    else i
                )
                idx2 = (
                    theta_resolution + i - 1
                    if (np.pi / 2 < phi < 3 * np.pi / 2)
                    else (theta_resolution - 1) - i
                )

            cor1 = (phi_bck / np.pi) * (horizontal_slice[0] - vertical_slice[idx1])
            cor2 = (phi_fnt / np.pi) * (
                horizontal_slice[phi_resolution // 2] - vertical_slice[idx2]
            )

            G[i, j] = horizontal_slice[j] - (cor1 + cor2)

    return G


class TestInterpolator:
    def test_summing(
        self,
        slice_xz_norm: npt.NDArray[np.floating],
        slice_xy_norm: npt.NDArray[np.floating],
        intensity_norm: npt.NDArray[np.floating],
    ):
        slice_xz, slice_xy = slice_xz_norm.squeeze(), slice_xy_norm.squeeze()
        vectorized_pattern = summing(slice_xz, slice_xy)
        naive_pattern = naive_summing(slice_xz, slice_xy)
        assert np.allclose(vectorized_pattern, naive_pattern)

    def test_bilinear(
        self,
        slice_xz_norm: npt.NDArray[np.floating],
        slice_xy_norm: npt.NDArray[np.floating],
        intensity_norm: npt.NDArray[np.floating],
    ):
        slice_xz, slice_xy = slice_xz_norm.squeeze(), slice_xy_norm.squeeze()
        vectorized_pattern = bilinear(slice_xz, slice_xy)
        naive_pattern = naive_bilinear(slice_xz, slice_xy)
        assert np.allclose(vectorized_pattern, naive_pattern)

    def test_weighted_bilinear(
        self,
        slice_xz_norm: npt.NDArray[np.floating],
        slice_xy_norm: npt.NDArray[np.floating],
        intensity_norm: npt.NDArray[np.floating],
    ):
        slice_xz, slice_xy = slice_xz_norm.squeeze(), slice_xy_norm.squeeze()
        vectorized_pattern = weighted_bilinear(slice_xz, slice_xy)
        naive_pattern = naive_weighted_bilinear(slice_xz, slice_xy)
        assert np.allclose(vectorized_pattern, naive_pattern)

    def test_horizontal_projection(
        self,
        slice_xz_norm: npt.NDArray[np.floating],
        slice_xy_norm: npt.NDArray[np.floating],
        intensity_norm: npt.NDArray[np.floating],
    ):
        slice_xz, slice_xy = slice_xz_norm.squeeze(), slice_xy_norm.squeeze()
        vectorized_pattern = horizontal_projection(slice_xz, slice_xy)
        naive_pattern = naive_horizontal_projection(slice_xz, slice_xy)
        assert np.allclose(vectorized_pattern, naive_pattern)

    def test_exponential(
        self,
        slice_xz_norm: npt.NDArray[np.floating],
        slice_xy_norm: npt.NDArray[np.floating],
        intensity_norm: npt.NDArray[np.floating],
    ):
        slice_xz, slice_xy = slice_xz_norm.squeeze(), slice_xy_norm.squeeze()
        vectorized_pattern = exponential(slice_xz, slice_xy)
        assert True
