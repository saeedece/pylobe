# pyright: basic

from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.colors import Normalize
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .utils import (
    AnyArray,
    CoordinateGrid,
    FullPattern,
    HorizontalSlice,
    SphericalAxis,
    VerticalSlice,
)

plt.rcParams.update({"font.size": 12})


def plot_axis_polar(
    ax: PolarAxes,
    domain: tuple[SphericalAxis, AnyArray],
    ranges: dict[str, VerticalSlice | HorizontalSlice],
    vlimits: tuple[float, float],
    modifiers: dict[str, dict[Any, Any]] | None = None,
    plot_legend: bool = False,
    legend_angle: float = 67.5,
    logarithmic_scale: bool = True,
) -> None:
    r"""Plots a set of vertical/horizontal pattern slices to a single set of polar axis.

    Parameters
    ----------
    ax : PolarAxes
        The polar axes to be modified in-place.
    domain : tuple[SphericalAxis, FloatArray]
        The domain is defined by the values of the independent variable and whether the variable is phi or theta. This allows us to correctly orient the polar axis.
    ranges : dict[str, VerticalSlice | HorizontalSlice]
        The ranges are the slices to plot. Each slice is labelled in the case that a legend must be plotted.
    vlimits : tuple[float, float]
        The minimum and maxmimum values of the r axis. Ideally, this range should contain the range of all the slices in `ranges`. `0 <= vlimits[0] <= vlimits[1]`. It is the user's responsiblity to properly normalize prior to plotting.
    modifiers : dict[str, dict[Any, Any]] | None
        Modifiers used when plotting each slice
    plot_legend : bool
    legend_angle : float
    logarithmic_scale : bool
    """
    if logarithmic_scale:
        ax.set_rscale("log")

    if modifiers is None:
        modifiers = defaultdict(dict)

    ax.set_rlim(*vlimits)
    domain_type, coords = domain

    match domain_type:
        case SphericalAxis.Theta:
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
        case SphericalAxis.Phi:
            pass

    assert len(ranges) > 0

    for label, y in ranges.items():
        ax.plot(coords, y, label=label, **modifiers[label])

    if plot_legend:
        legend_angle = np.deg2rad(legend_angle)
        _ = ax.legend(
            ncol=1,
            loc="lower left",
            bbox_to_anchor=(
                0.6 + np.cos(legend_angle) / 2,
                0.5 + np.sin(legend_angle) / 2,
            ),
            fontsize="12",
        )

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="x", which="major", pad=10)
    return


def plot_axis_spherical(
    ax: Axes3D,
    phi_grid: CoordinateGrid,
    theta_grid: CoordinateGrid,
    pattern: FullPattern,
    vlimits: tuple[float, float],
    visible_axes: bool = False,
) -> Poly3DCollection:
    r"""Plots a single full pattern to a 3-D axis.

    Parameters
    ----------
    ax : Axes3D
        The 3-D axes to be modified in-place.
    phi_grid : CoordinateGrid
    theta_grid : CoordinateGrid
    pattern : FullPattern
    vlimits : tuple[float, float]
        The minimum and maxmimum values of each axis. `vlimits` should be set such that the entire pattern is visualized. `0 <= vlimits[0] <= vlimits[1]`. It is the user's responsiblity to properly normalize prior to plotting.
    visible_axes : bool
        Boolean flag to visualize axis grids.

    Returns
    -------
    Poly3DCollection
    """
    x = pattern * np.sin(theta_grid) * np.cos(phi_grid)
    y = pattern * np.sin(theta_grid) * np.sin(phi_grid)
    z = pattern * np.cos(theta_grid)
    colors = Normalize(*vlimits, clip=True)(pattern)
    d = vlimits[1] - vlimits[0]
    ax.set_xlim(-d, d)
    ax.set_ylim(-d, d)
    ax.set_zlim(-d, d)

    if not visible_axes:
        ax.set_axis_off()
    cax = ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=plt.get_cmap("jet")(colors),
        antialiased=True,
        alpha=1.0,
    )
    return cax


def plot_axis_planar(
    ax: Axes,
    pattern: FullPattern,
    vlimits: tuple[float, float],
    extent: tuple[float, float, float, float] | None = None,
) -> AxesImage:
    r"""Plots a single full pattern to a 2-D image.

    Parameters
    ----------
    ax : Axes
        The 2-D axes to be modified in-place.
    pattern : FullPattern
    vlimits : tuple[float, float]
        The minimum and maxmimum values of each axis. `vlimits` should be set such that the entire pattern is visualized.
    extent : tuple[float, float, float, float] | None
        The extent of the axes.

    Returns
    -------
    AxesImage
    """
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\theta$")
    cax = ax.imshow(
        pattern,
        cmap=plt.get_cmap("viridis"),
        extent=extent,
        vmin=vlimits[0],
        vmax=vlimits[1],
    )
    return cax
