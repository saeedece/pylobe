from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from pylobe.utils import (
    CoordinateGrid,
    FullPattern,
    build_equiangular_grid,
    SphericalAxis,
)
from pylobe.transform import (
    extract_phi_plane,
    extract_theta_cone,
    linear_to_db,
    unit_normalize_db_pattern,
)
from pylobe.pattern import linear_array_factor
from pylobe.visualize import plot_axis_polar, plot_axis_spherical, plot_axis_planar


class TestVisualize:
    output_dir = Path(".", "test", "artifacts")
    output_dir.mkdir(parents=False, exist_ok=True)
    theta_grid, phi_grid = build_equiangular_grid()
    pattern: FullPattern = unit_normalize_db_pattern(
        linear_to_db(
            np.square(
                np.abs(
                    linear_array_factor(
                        theta_grid,
                        np.complex64(1.0),
                        np.zeros(1),
                        num_elements=5,
                    )
                )
            )
        ),
        clip_value=-60,
    )

    xz = extract_phi_plane(pattern, 0, endpoint=False).squeeze()
    yz = extract_phi_plane(pattern, np.pi / 4, endpoint=False).squeeze()
    xy = extract_theta_cone(pattern, 0, endpoint=False).squeeze()

    def test_plot_polar_vertical(self):
        coords = np.linspace(0, 2 * np.pi, 360)
        domain = (SphericalAxis.Theta, coords)
        ranges = {"XZ": self.xz, "YZ": self.yz}
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
        fig.savefig(Path(self.output_dir, "polar_vertical.png"))
        return

    def test_plot_polar_horizontal(self):
        coords = np.linspace(0, 2 * np.pi, 360)
        domain = (SphericalAxis.Phi, coords)
        ranges = {"XY": self.xy}
        modifiers = {
            "XY": {"color": "red", "linewidth": 2.0},
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
        fig.savefig(Path(self.output_dir, "polar_horizontal.png"))
        return

    def test_plot_planar(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        _ = plot_axis_planar(ax, self.pattern, vlimits=(0, 1), extent=None)
        fig.savefig(Path(".", "test", "artifacts", "planar.png"))
        return

    def test_plot_spherical(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        _ = plot_axis_spherical(
            ax,
            self.phi_grid,
            self.theta_grid,
            self.pattern,
            vlimits=(0, 1),
            visible_axes=True,
        )
        fig.savefig(Path(self.output_dir, "spherical.png"))
