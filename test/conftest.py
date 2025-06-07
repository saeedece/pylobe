import numpy as np
import numpy.typing as npt
import pytest

from pylobe.pattern import linear_array_factor
from pylobe.transform import (
    extract_phi_plane,
    extract_theta_cone,
    linear_to_db,
    unit_normalize_db_pattern,
)
from pylobe.utils import build_equiangular_grid


@pytest.fixture()
def simple_grid():
    return np.arange(110).reshape(11, 10).astype(dtype=np.float32)


@pytest.fixture()
def factor():
    return linear_array_factor(
        theta_grid=build_equiangular_grid()[1],
        kd=np.complex64(1.0),
        beta=np.zeros(shape=(1,), dtype=np.complex64),
        num_elements=5,
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


@pytest.fixture()
def intensity_norm(intensity):
    return intensity - np.max(intensity)
    # return unit_normalize_db_pattern(intensity)


@pytest.fixture()
def slice_xy_norm(intensity_norm: npt.NDArray[np.floating]):
    return extract_theta_cone(intensity_norm, np.pi / 2, endpoint=False)


@pytest.fixture()
def slice_xz_norm(intensity_norm: npt.NDArray[np.floating]):
    return extract_phi_plane(intensity_norm, 0, endpoint=False)


@pytest.fixture()
def slice_yz_norm(intensity_norm: npt.NDArray[np.floating]):
    return extract_phi_plane(intensity_norm, np.pi / 2, endpoint=False)
