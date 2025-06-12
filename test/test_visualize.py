from pathlib import Path

import pytest
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from pylobe.transform import (
    extract_phi_plane,
    extract_theta_cone,
    unit_normalize_db_pattern,
)
from pylobe.utils import build_equiangular_grid, SphericalAxis
from pylobe.visualize import plot_axis_polar, plot_axis_spherical, plot_axis_planar


@pytest.fixture()
def intensity_norm(intensity):
    return unit_normalize_db_pattern(intensity)


@pytest.fixture()
def slice_xy_norm(intensity_norm: npt.NDArray[np.floating]):
    return extract_theta_cone(intensity_norm, np.pi / 2, endpoint=False)


@pytest.fixture()
def slice_xz_norm(intensity_norm: npt.NDArray[np.floating]):
    return extract_phi_plane(intensity_norm, 0, endpoint=False)


@pytest.fixture()
def slice_yz_norm(intensity_norm: npt.NDArray[np.floating]):
    return extract_phi_plane(intensity_norm, np.pi / 2, endpoint=False)


class TestVisualize:
    def test_plot_polar_vertical(
        self,
        slice_xz_norm: npt.NDArray[np.floating],
        slice_yz_norm: npt.NDArray[np.floating],
        tmp_path: Path,
    ):
        coords = np.linspace(0, 2 * np.pi, 360)
        domain = (SphericalAxis.Theta, coords)
        ranges = {"XZ": slice_xz_norm.squeeze(), "YZ": slice_yz_norm.squeeze()}
        modifiers = {
            "XZ": {"color": "red", "linewidth": 2.0},
            "YZ": {"color": "blue", "linewidth": 2.5, "linestyle": ":"},
        }
        vlimits = (0, 1)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")
        plot_axis_polar(
            ax,
            domain,
            ranges,
            vlimits,
            modifiers=modifiers,
            plot_legend=True,
            legend_angle=67.5,
            logarithmic_scale=False,
        )
        fig.savefig(tmp_path / "polar_vertical.png")
        return

    def test_plot_polar_horizontal(
        self,
        slice_xy_norm: npt.NDArray[np.floating],
        tmp_path: Path,
    ):
        coords = np.linspace(0, 2 * np.pi, 360)
        domain = (SphericalAxis.Phi, coords)
        ranges = {"XY": slice_xy_norm.squeeze()}
        modifiers = {"XY": {"color": "red", "linewidth": 2.0}}
        vlimits = (0, 1)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")
        plot_axis_polar(
            ax,
            domain,
            ranges,
            vlimits,
            modifiers=modifiers,
            plot_legend=True,
            legend_angle=67.5,
            logarithmic_scale=False,
        )
        fig.savefig(tmp_path / "polar_horizontal.png")
        return

    def test_plot_planar(
        self,
        intensity_norm: npt.NDArray[np.floating],
        tmp_path: Path,
    ):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        _ = plot_axis_planar(ax, intensity_norm, vlimits=(0, 1), extent=None)
        fig.savefig(tmp_path / "planar.png")
        return

    def test_plot_spherical(
        self,
        intensity_norm: npt.NDArray[np.floating],
        tmp_path: Path,
    ):
        phi_grid, theta_grid = build_equiangular_grid()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        _ = plot_axis_spherical(
            ax,
            phi_grid,
            theta_grid,
            intensity_norm,
            vlimits=(0, 1),
            visible_axes=True,
        )
        fig.savefig(tmp_path / "spherical.png")
