from enum import Enum
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

AnyArray: TypeAlias = npt.NDArray[Any]  # pyright: ignore[reportExplicitAny]
FloatArray: TypeAlias = npt.NDArray[np.float16 | np.float32 | np.float64]

RotationMatrix: TypeAlias = FloatArray
r"""RotationMatrix is a type alias for a NumPy array of floats with shape (3, 3). This is not reflected in the type annotation."""

CoordinateGrid: TypeAlias = FloatArray
r"""CoordinateGrid is a type alias for a NumPy array of floats with arbitrary shape."""

FullPattern: TypeAlias = FloatArray
r"""FullPattern is a type alias for a NumPy array of floats with arbitrary shape. However, we typically expect the shape of instances of `FullPattern` to be of the form (..., H, W), where H is odd and W is even. FullPattern expects the final two-dimensions to be (theta, phi), with colatitude-longitude arrangement."""

VerticalSlice: TypeAlias = FloatArray
r"""VerticalSlice is a type alias for a NumPy array of floats with arbitrary shape. However, we typically expect the shape of instances of `VerticalSlice` to be of the form (..., 2 * H - 2, 1), where H is odd."""

HorizontalSlice: TypeAlias = FloatArray
r"""VerticalSlice is a type alias for a NumPy array of floats with arbitrary shape. However, we typically expect the shape of instances of `HorizontalSlice` to be of the form (..., 1, W), where W is even."""


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2


class SphericalAxis(Enum):
    Theta = 0
    Phi = 1


class Plane(Enum):
    XZ = 0
    YZ = 1
    XY = 2


def build_equiangular_grid(
    phi_resolution: int = 360,
    theta_resolution: int = 181,
) -> tuple[CoordinateGrid, ...]:
    r"""Builds equiangular grid over the unit sphere with provided angular resolutions.

    Parameters
    ----------
    phi_resolution : int
        Angular resolution of phi values on the coordinate grid.
    theta_resolution : int
        Angular resolution of theta values on the coordinate grid.

    Returns
    -------
    tuple[CoordinateGrid, CoordinateGrid]
        2-tuple of phi and theta coordinate grids.

    Raises
    ------
    ValueError
        Constrains phi_resolution to be even so that :math:`\theta = \frac{\pi}{2}` is always included.

    """
    if phi_resolution % 2 != 0:
        raise ValueError("Angular resolution for phi coordinate values must be even.")

    longitude = np.linspace(0, 2 * np.pi, phi_resolution, endpoint=False)
    colatitude = np.linspace(0, np.pi, theta_resolution)
    return np.meshgrid(longitude, colatitude)


def spherical_to_cartesian(
    phi_grid: CoordinateGrid, theta_grid: CoordinateGrid
) -> tuple[CoordinateGrid, CoordinateGrid, CoordinateGrid]:
    r"""Converts spherical coordinate grids to cartesian coordinate grids.

    Parameters
    ----------
    phi_grid : CoordinateGrid
        Coordinate grid of phi values.
    theta_grid : CoordinateGrid
        Coordinate grid of theta values.

    Returns
    -------
    tuple[CoordinateGrid, CoordinateGrid, CoordinateGrid]
        3-tuple of x, y, z coordinate grids.
    """
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    return x, y, z


def cartesian_to_spherical(
    x: CoordinateGrid, y: CoordinateGrid, z: CoordinateGrid
) -> tuple[CoordinateGrid, CoordinateGrid]:
    r"""Converts cartesian coordinate grids to spherical coordinate grids.

    Parameters
    ----------
    x : CoordinateGrid
    y : CoordinateGrid
    z : CoordinateGrid

    Returns
    -------
    tuple[CoordinateGrid, CoordinateGrid]
        2-tuple of phi, theta coordinate grids.
    """
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    phi = np.where(np.isclose(phi, 2 * np.pi), 0, phi)
    theta = np.arccos(z)
    return phi, theta


def build_rotation_matrix(angle: float, axis: Axis) -> RotationMatrix:
    r"""Produces a rotation matrix from an angle values and a rotation axis.

    Parameters
    ----------
    angle: float
    axis: Axis

    Returns
    -------
    RotationMatrix
    """
    match axis:
        case Axis.X:
            return np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)],
                ]
            )

        case Axis.Y:
            return np.array(
                [
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ]
            )

        case Axis.Z:
            return np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )


def rotate_cartesian(
    x: CoordinateGrid,
    y: CoordinateGrid,
    z: CoordinateGrid,
    angles: list[float],
    axes: list[Axis],
) -> tuple[CoordinateGrid, CoordinateGrid, CoordinateGrid]:
    r"""Produces rotated Cartesian coordinate grids given a list of angles and a list of rotation axes.

    Parameters
    ----------
    x : CoordinateGrid
    y : CoordinateGrid
    z : CoordinateGrid
    angles: list[float]
    axes: list[Axis]

    Returns
    -------
    tuple[CoordinateGrid, CoordinateGrid, CoordinateGrid]
        3-tuple of rotated Cartesian coordinate grids.
    """
    xyz = np.stack([x, y, z]).reshape(3, -1)

    r = np.eye(3, 3, dtype=x.dtype)
    for angle, axis in zip(angles, axes):
        r @= build_rotation_matrix(angle, axis)

    xyz = np.dot(r, xyz).reshape(3, *x.shape)  # pyright: ignore[reportAny]
    return tuple(xyz)  # pyright: ignore[reportAny]


def unpad(arr: AnyArray, pad_widths: list[tuple[int, int]]) -> AnyArray:
    r"""Unpads a NumPy array given a list of padding widths, which are assumed to be assymetrical. Partially lifted from https://stackoverflow.com/a/57956349

    Parameters
    ----------
    arr : AnyArray
    pad_widths : list[tuple[int, int]]
        We require two widths to be set for each dimension. In the case that `len(pad_widths) < arr.ndim`, then un-padding is applied on the right-most dimensions.

    Returns
    -------
    AnyArray

    Raises
    ------
    ValueError
        `len(pad_widths)` must be less than or equal to `arr.ndim`.
    """
    num_pads = len(pad_widths)
    rank, shape = arr.ndim, arr.shape
    if num_pads > rank:
        raise ValueError(
            "The number of pad widths cannot exceed the rank of the array."
        )

    total_pad_widths = [(0, 0) for _ in range(rank)]
    for i in range(rank - num_pads, rank):
        total_pad_widths[i] = pad_widths[i - rank + num_pads]

    rpad = [
        slice(start, n - stop) for ((start, stop), n) in zip(total_pad_widths, shape)
    ]
    return arr[tuple(rpad)]
