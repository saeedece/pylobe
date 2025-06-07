import numpy as np
import numpy.typing as npt


def linear_array_factor(
    theta_grid: npt.NDArray[np.floating],
    kd: np.complexfloating,
    beta: npt.NDArray[np.complexfloating],
    num_elements: int,
) -> npt.NDArray[np.complexfloating]:
    r"""Computes a linear array factor.

    The linear array is assumed to be oriented along the z-axis. This results in the array factor having no phi-dependence.

    Parameters
    ----------
    theta_grid : npt.NDArray[np.floating]
    kd : np.complex64 | np.complex128 | np.complex256
        Constant displacement between each array element scaled by the wave number.
    beta : npt.NDArray[np.complexfloating]
        Phase between each array element.
    num_elements:
        Number of elements forming the linear array.

    Returns
    -------
    npt.NDArray[np.complexfloating]
        The array is complex-valued, but we annotate it as npt.NDArray[np.complexfloating] because I am lazy.
    """
    psi = kd * np.cos(theta_grid) + beta.reshape(
        beta.size,
        *(1 for _ in range(theta_grid.ndim)),
    )
    ms = np.arange(num_elements).reshape(
        num_elements, *(1 for _ in range(psi.ndim - 1))
    )
    return np.sum(np.exp(1j * ms * psi), axis=0)  # pyright: ignore[reportAny]


def planar_array_factor(
    theta_grid: npt.NDArray[np.floating],
    phi_grid: npt.NDArray[np.floating],
    m: int,
    kdx: np.complex64 | np.complex128 | np.complex256,
    betax: npt.NDArray[np.complexfloating],
    im1: npt.NDArray[np.complexfloating],
    n: int,
    kdy: np.complex64 | np.complex128 | np.complex256,
    betay: npt.NDArray[np.complexfloating],
    i1n: npt.NDArray[np.complexfloating],
) -> npt.NDArray[np.complexfloating]:
    r"""Computes a planar array factor.

    The planar array is assumed to be oriented along the xy-plane.

    Parameters
    ----------
    theta_grid : npt.NDArray[np.floating]
    phi_grid : npt.NDArray[np.floating]
    kdx : np.complex64 | np.complex128 | np.complex256
        Constant displacement between each array element scaled by the wave number along the x-axis.
    betax : npt.NDArray[np.complexfloating]
        Phase between each array element along the x-axis.
    im1 : npt.NDArray[np.complexfloating]
        Excitation coefficient of each array element along the x-axis.
    m : int
        Number of array elements along the x-axis.
    kdy : np.complex64 | np.complex128 | np.complex256
        Constant displacement between each array element scaled by the wave number along the y-axis.
    betay : npt.NDArray[np.complexfloating]
        Phase between each array element along the y-axis.
    i1n : npt.NDArray[np.complexfloating]
        Excitation coefficient of each array element along the y-axis.
    n : int
        Number of array elements along the y-axis.

    Returns
    -------
    npt.NDArray[np.complexfloating]
        The array is complex-valued, but we annotate it as npt.NDArray[np.complexfloating] because I am lazy.
    """
    added_dims = tuple(1 for _ in range(theta_grid.ndim))

    betax = betax.reshape(betax.shape, *added_dims)
    betay = betay.reshape(betay.shape, *added_dims)

    psix = kdx * np.sin(theta_grid) * np.cos(phi_grid) + betax
    psiy = kdy * np.sin(theta_grid) * np.sin(phi_grid) + betay

    ms = np.arange(m).reshape(m, *added_dims)
    ns = np.arange(n).reshape(n, *added_dims)
    im1 = im1.reshape(m, *added_dims)
    i1n = i1n.reshape(n, *added_dims)

    sxm = np.sum(im1 * np.exp(1j * ms * psix), axis=0)  # pyright: ignore[reportAny]
    syn = np.sum(i1n * np.exp(1j * ns * psiy), axis=0)  # pyright: ignore[reportAny]
    return sxm * syn  # pyright: ignore[reportAny]


def gain_model_directive(
    theta_grid: npt.NDArray[np.floating],
    alpha: float,
    beta: float,
) -> npt.NDArray[np.floating]:
    r"""Computes a simple directive gain model.

     Parameters
     ----------
     theta_grid : npt.NDArray[np.floating]
        Normalization factor.
     beta : float
         Constant gain value for southern hemisphere.

     Returns
    npt.NDArray[np.floating]
    """
    gain = beta * np.ones_like(theta_grid)
    gain[theta_grid <= np.pi / 2] = (
        2 * (alpha + 1) * np.power(np.cos(theta_grid), alpha)
    )
    return gain
