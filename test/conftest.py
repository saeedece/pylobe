import numpy as np
import numpy.typing as npt
import pytest

from pylobe.pattern import planar_array_factor
from pylobe.transform import (
    extract_phi_plane,
    extract_theta_cone,
    linear_to_db,
)
from pylobe.utils import build_equiangular_grid


@pytest.fixture()
def simple_grid():
    return np.arange(110).reshape(11, 10).astype(dtype=np.float32)


@pytest.fixture()
def factor():
    phi_grid, theta_grid = build_equiangular_grid()
    return planar_array_factor(
        theta_grid,
        phi_grid,
        m=5,
        kdx=np.complex64(5 * np.pi / 7),
        betax=np.zeros(shape=(1,), dtype=np.complex64),
        im1=np.ones(shape=(1,), dtype=np.complex64),
        n=3,
        kdy=np.complex64(np.pi / 2),
        betay=np.zeros(shape=(1,), dtype=np.complex64),
        i1n=np.ones(shape=(1,), dtype=np.complex64),
    )


@pytest.fixture()
def intensity_linear_scale(factor: npt.NDArray[np.complexfloating]):
    return np.square(np.abs(factor))


@pytest.fixture()
def intensity(intensity_linear_scale: npt.NDArray[np.floating]):
    return linear_to_db(intensity_linear_scale, clip_value=-60)


@pytest.fixture()
def slice_xy(intensity: npt.NDArray[np.floating]):
    return extract_theta_cone(intensity, np.pi / 2, endpoint=False)


@pytest.fixture()
def slice_xz(intensity: npt.NDArray[np.floating]):
    return extract_phi_plane(intensity, 0, endpoint=False)


@pytest.fixture()
def slice_yz(intensity: npt.NDArray[np.floating]):
    return extract_phi_plane(intensity, np.pi / 2, endpoint=False)
