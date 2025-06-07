import numpy as np
import numpy.typing as npt

from .utils import SphericalAxis


def _longest_subsequence(arr: npt.NDArray[np.bool]) -> int:
    if arr.ndim != 1:
        raise ValueError("Input array must have singular dimension.")

    if arr.size == 0:
        return 0

    padded = np.r_[False, arr, False].astype(int)
    diff = np.diff(padded)

    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    if starts.size == 0:
        return 0

    return int(np.max(ends - starts))


def compute(
    pattern_slice: npt.NDArray[np.floating],
    axis: SphericalAxis,
    threshold: float = 3,
) -> float:
    r"""Computes the beamwidth of a pattern given a vertical/horizontal slice.

    This function first wraps the pattern slice in the case that the primarly lobe lies across the boundaries of the slice.

    Parameters
    ----------
    pattern_slice : npt.NDArray[np.floating]
    axis : SphericalAxis
        Needed for proper wrapping of slices.
    threshold : float

    Returns
    -------
    float
    """
    gain_max = np.max(pattern_slice)
    match axis:
        case SphericalAxis.Phi:
            phi_resolution = pattern_slice.size
            pattern_slice_padded = np.pad(
                pattern_slice,
                (phi_resolution // 2, phi_resolution // 2),
                mode="wrap",
            )
            lobe_mask = pattern_slice_padded >= gain_max - threshold
            pixelated_beamwidth = min(_longest_subsequence(lobe_mask), 360)
            return pixelated_beamwidth * (2 * np.pi) / (phi_resolution + 1)

        case SphericalAxis.Theta:
            theta_resolution = pattern_slice.size // 2 + 1
            pattern_slice_padded = np.pad(
                pattern_slice,
                (theta_resolution - 1, theta_resolution - 1),
                mode="wrap",
            )
            lobe_mask = pattern_slice_padded >= gain_max - threshold
            pixelated_beamwidth = min(_longest_subsequence(lobe_mask), 360)
            return pixelated_beamwidth * (np.pi / theta_resolution)
