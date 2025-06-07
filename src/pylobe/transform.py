import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator  # pyright: ignore[reportMissingTypeStubs]

from .utils import (
    Axis,
    cartesian_to_spherical,
    Plane,
    rotate_cartesian,
    spherical_to_cartesian,
    unpad,
)


def rotate_pattern_bilinear(
    pattern: npt.NDArray[np.floating],
    angles: list[float],
    axes: list[Axis],
) -> npt.NDArray[np.floating]:
    r"""Rotates a NumPy array representing a 3-D gain (npt.NDArray[np.floating]) pattern given a list of angles and axes using a regular grid interpolator.

    Parameters
    ----------
    pattern : npt.NDArray[np.floating]
    angles : list[float]
    axes : list[Axis]

    Returns
    -------
    npt.NDArray[np.floating]
    """
    # pad pattern to include phi = 2 * pi just for interpolation
    # this is because rotated patterns can have phi in [2pi - 2pi/phi_resolution, 2pi)
    pattern = pad_spherical(pattern, pad_widths=[(0, 1), (0, 0)])
    theta_resolution, phi_resolution = pattern.shape[-2:]
    longitude = np.linspace(0, 2 * np.pi, phi_resolution)
    colatitude = np.linspace(0, np.pi, theta_resolution)
    phi_grid, theta_grid = np.meshgrid(longitude, colatitude)

    x, y, z = spherical_to_cartesian(phi_grid, theta_grid)
    x_rotated, y_rotated, z_rotated = rotate_cartesian(x, y, z, angles, axes)
    phi_grid_rotated, theta_grid_rotated = cartesian_to_spherical(
        x_rotated,
        y_rotated,
        z_rotated,
    )
    phi_grid_rotated.resize((phi_resolution * theta_resolution, 1))
    theta_grid_rotated.resize((phi_resolution * theta_resolution, 1))
    coords_rotated = np.hstack((theta_grid_rotated, phi_grid_rotated))

    interpolator = RegularGridInterpolator(
        (colatitude, longitude),
        pattern,
        bounds_error=True,
    )

    pattern_rotated = interpolator(xi=coords_rotated).astype(pattern.dtype)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    _ = pattern_rotated.resize(pattern.shape)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    return unpad(pattern_rotated, pad_widths=[(0, 1), (0, 0)])  # pyright: ignore[reportUnknownArgumentType]


def linear_to_db(
    pattern: npt.NDArray[np.floating], clip_value: float = -60
) -> npt.NDArray[np.floating]:
    r"""Converts a NumPy array representing a linear-scale far-field pattern to a logarithmic-scale far-field pattern.

    We enforce a clip value to be set to deal with zeros in the linear pattern.

    Parameters
    ----------
    pattern : npt.NDArray[np.floating]
    clip_values: float

    Returns
    -------
    npt.NDArray[np.floating]
    """
    out_pattern = np.ones_like(pattern) * (clip_value / 10.0)
    _ = np.log10(pattern, out_pattern, where=(pattern != 0), dtype=pattern.dtype)  # pyright: ignore[reportAny]
    out_pattern *= 10.0
    out_pattern = np.clip(out_pattern, clip_value, None)
    return out_pattern


def db_to_linear(pattern: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""Converts a NumPy array representing a logarithmic-scale far-field pattern to a linear-scale far-field pattern.

    Parameters
    ----------
    pattern : npt.NDArray[np.floating]

    Returns
    -------
    npt.NDArray[np.floating]
    """
    return np.power(10, (1 / 10) * pattern)


def unit_normalize_db_pattern(
    pattern: npt.NDArray[np.floating],
    clip_value: float = -60,
) -> npt.NDArray[np.floating]:
    r"""Normalize a NumPy array representing a logarithmic-scale far-field pattern to a range of [0, 1].

    A clipping value is used to set the minimum value of the logarithmic-scaled pattern, which is necessary for minmax normalization.

    Parameters
    ----------
    pattern : npt.NDArray[np.floating]
    clip_value : float

    Returns
    -------
    npt.NDArray[np.floating]
    """
    # normalize dB pattern to (-inf, 0]
    out_pattern = pattern - np.max(pattern)
    out_pattern = np.clip(out_pattern, clip_value, None)
    out_pattern = (out_pattern - clip_value) * (1 / -clip_value)
    return out_pattern


def unit_denormalize_db_pattern(
    pattern: npt.NDArray[np.floating],
    clip_value: float = -60,
) -> npt.NDArray[np.floating]:
    r"""Reverse the normalization of a NumPy array representing a logarithmic-scale far-field pattern to a range of [0, 1].

    Parameters
    ----------
    pattern : npt.NDArray[np.floating]
    clip_value: float

    Returns
    -------
    npt.NDArray[np.floating]
    """
    return clip_value - clip_value * pattern


def extract_phi_plane(
    pattern: npt.NDArray[np.floating],
    phi: float,
    endpoint: bool = False,
) -> npt.NDArray[np.floating]:
    r"""Extracts a slice lying on a single phi-plane from a 3-D pattern.

    Parameters
    ----------
    pattern : npt.NDArray[np.floating]
    phi : float
    endpoint : bool
        Whether to include the point at theta = 2pi.

    Returns
    -------
    npt.NDArray[np.floating]

    Raises
    ------
    ValueError
        Phi must lie in [0, np.pi].
    """
    if not (0 <= phi <= np.pi):
        raise ValueError("Phi must lie in [0, np.pi / 2].")

    phi_resolution = pattern.shape[-1]
    lft_index = int(np.round((phi * phi_resolution) / (2 * np.pi)))  # pyright: ignore[reportAny]
    rgt_index = lft_index + phi_resolution // 2

    lft_half = pattern[..., :, lft_index : lft_index + 1]
    theta_bound = 0 if endpoint else 1
    rgt_half = np.flip(pattern[..., theta_bound:-1, rgt_index : rgt_index + 1], axis=-2)
    phi_plane = np.concatenate((lft_half, rgt_half), axis=-2)
    return phi_plane


def extract_theta_cone(
    pattern: npt.NDArray[np.floating],
    theta: float,
    endpoint: bool = False,
) -> npt.NDArray[np.floating]:
    r"""Extracts a slice lying on a single theta cone from a 3-D pattern.

    Parameters
    ----------
    pattern : npt.NDArray[np.floating]
    theta : float
    endpoint : bool
        Whether to include the point at phi = 2pi.

    Returns
    -------
    npt.NDArray[np.floating]

    Raises
    ------
    ValueError
        Theta must lie in [0, np.pi].
    """
    theta_resolution = pattern.shape[-2]
    idx = int(np.round(theta * (theta_resolution - 1)) / np.pi)  # pyright: ignore[reportAny]
    phi_bound = 1 if endpoint else 0
    theta_cone = np.concatenate(
        (pattern[..., idx : idx + 1, :], pattern[..., idx : idx + 1, 0:phi_bound]),
        axis=-1,
    )
    return theta_cone


def pad_spherical(
    pattern: npt.NDArray[np.floating],
    pad_widths: list[tuple[int, int]],
) -> npt.NDArray[np.floating]:
    r"""Pads a 3-D pattern while preserving spherical properties.

    Parameters
    ----------
    pattern : npt.NDArray[np.floating]
    pad_widths : list[tuple[int, int]]
        Two widths must be defined for each dimension.

    Returns
    -------
    npt.NDArray[np.floating]
    """
    phi_resolution = pattern.shape[-1]
    top_lft = pattern[..., 1 : pad_widths[0][0] + 1, : phi_resolution // 2]
    top_rgt = pattern[..., 1 : pad_widths[0][0] + 1, phi_resolution // 2 :]
    top_blk = np.concatenate((top_rgt, top_lft), axis=-1)
    top_blk = np.flip(top_blk, axis=-2)

    bot_lft = pattern[..., -1 - pad_widths[0][1] : -1, : phi_resolution // 2]
    bot_rgt = pattern[..., -1 - pad_widths[0][1] : -1, phi_resolution // 2 :]
    bot_blk = np.concatenate((bot_rgt, bot_lft), axis=-1)
    bot_blk = np.flip(bot_blk, axis=-2)

    out_pattern = np.concatenate((top_blk, pattern, bot_blk), axis=-2)

    ndim_pad_widths = [(0, 0) for _ in range(pattern.ndim)]
    ndim_pad_widths[-1] = pad_widths[-1]
    import numpy.typing as npt

    out_pattern = np.pad(out_pattern, pad_width=ndim_pad_widths, mode="wrap")
    return out_pattern


def mask(
    pattern: npt.NDArray[np.floating], planes: set[Plane]
) -> npt.NDArray[np.floating]:
    r"""Zero-masks a pattern such that only points on planes are shown.

    Parameters
    ----------
    import numpy.typing as npt
    pattern : npt.NDArray[np.floating]
    planes : set[Plane]

    Returns
    -------
    npt.NDArray[np.floating]
    """
    theta_resolution, phi_resolution = pattern.shape[-2:]

    rows_kept: list[int] = []
    cols_kept: list[int] = []

    for plane in planes:
        match plane:
            case Plane.XZ:
                cols_kept.append(0)
                cols_kept.append(phi_resolution // 2)

            case Plane.YZ:
                cols_kept.append(phi_resolution // 4)
                cols_kept.append(3 * phi_resolution // 4)

            case Plane.XY:
                rows_kept.append(theta_resolution // 2)

    out_pattern = np.zeros_like(pattern)
    out_pattern[..., cols_kept] = pattern[..., cols_kept]
    out_pattern[..., rows_kept, :] = pattern[..., rows_kept, :]
    return out_pattern
