#!/usr/bin/env python3
"""Generate manifold reference STLs matching beak_liner_reference.scad.

The OpenSCAD source is canonical and should be edited for an actual beak fit.
This dependency-light exporter performs a 0.25 mm boolean voxel union, then
emits only exterior faces. That makes each included STL a single slicer-safe
manifold instead of a collection of intersecting primitive shells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


INNER_WIDTH_MM = 28.0
CAVITY_DEPTH_MM = 26.0
PAD_THICKNESS_MM = 1.5
WALL_THICKNESS_MM = 2.0
LIP_RADIUS_MM = 2.0
CHEEK_HEIGHT_MM = 5.0
REAR_STOP_HEIGHT_MM = 5.0
OUTER_WIDTH_MM = INNER_WIDTH_MM + 2.0 * WALL_THICKNESS_MM
PITCH_MM = 0.25


def occupancy(part: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a boolean solid grid and its minimum XYZ corner."""
    origin = np.array([0.0, -OUTER_WIDTH_MM / 2.0, 0.0])
    maximum = np.array(
        [
            CAVITY_DEPTH_MM + LIP_RADIUS_MM,
            OUTER_WIDTH_MM / 2.0,
            PAD_THICKNESS_MM
            + max(2.0 * LIP_RADIUS_MM, CHEEK_HEIGHT_MM, REAR_STOP_HEIGHT_MM),
        ]
    )
    shape = np.ceil((maximum - origin) / PITCH_MM).astype(int)
    axes = [
        origin[axis] + (np.arange(shape[axis]) + 0.5) * PITCH_MM
        for axis in range(3)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")

    base = (
        (x >= 0.0)
        & (x <= CAVITY_DEPTH_MM)
        & (np.abs(y) <= OUTER_WIDTH_MM / 2.0)
        & (z >= 0.0)
        & (z <= PAD_THICKNESS_MM)
    )
    lip = (
        (
            (x - CAVITY_DEPTH_MM) ** 2
            + (z - (PAD_THICKNESS_MM + LIP_RADIUS_MM)) ** 2
        )
        <= LIP_RADIUS_MM**2
    ) & (np.abs(y) <= OUTER_WIDTH_MM / 2.0)
    solid = base | lip

    if part == "upper":
        rear = (
            (x >= 0.0)
            & (x <= WALL_THICKNESS_MM)
            & (np.abs(y) <= OUTER_WIDTH_MM / 2.0)
            & (z >= 0.0)
            & (z <= PAD_THICKNESS_MM + REAR_STOP_HEIGHT_MM)
        )
        cheeks = (
            (x >= 0.0)
            & (x <= CAVITY_DEPTH_MM)
            & (np.abs(y) >= INNER_WIDTH_MM / 2.0)
            & (np.abs(y) <= OUTER_WIDTH_MM / 2.0)
            & (z >= 0.0)
            & (z <= PAD_THICKNESS_MM + CHEEK_HEIGHT_MM)
        )
        solid |= rear | cheeks
    return solid, origin


# Face corners in outward winding, indexed from an occupied voxel's lower XYZ
# grid corner. Each tuple is (neighbor offset, four grid-corner offsets).
FACE_DEFS = (
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
)


def exterior_mesh(solid: np.ndarray, origin: np.ndarray) -> trimesh.Trimesh:
    padded = np.pad(solid, 1, constant_values=False)
    size = solid.shape
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    vertex_ids: dict[tuple[int, int, int], int] = {}

    for (dx, dy, dz), corners in FACE_DEFS:
        neighbor = padded[
            1 + dx : 1 + dx + size[0],
            1 + dy : 1 + dy + size[1],
            1 + dz : 1 + dz + size[2],
        ]
        for cell in np.argwhere(solid & ~neighbor):
            face: list[int] = []
            for corner in corners:
                key = tuple(int(cell[axis] + corner[axis]) for axis in range(3))
                if key not in vertex_ids:
                    vertex_ids[key] = len(vertices)
                    vertices.append(origin + np.asarray(key) * PITCH_MM)
                face.append(vertex_ids[key])
            faces.append(face)

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices),
        faces=np.asarray(faces),
        process=True,
        validate=True,
    )
    if not mesh.is_watertight or len(mesh.split()) != 1:
        raise RuntimeError("generated liner is not one watertight component")
    if mesh.volume < 0.0:
        mesh.invert()
    mesh.metadata["units"] = "mm"
    return mesh


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    for part in ("lower", "upper"):
        solid, origin = occupancy(part)
        mesh = exterior_mesh(solid, origin)
        path = output_dir / f"beak_liner_{part}_reference.stl"
        mesh.export(path)
        bounds = np.round(mesh.bounds, 2).tolist()
        print(
            f"wrote {path.name}: {len(mesh.faces)} faces, "
            f"bounds_mm={bounds}, volume_mm3={mesh.volume:.1f}"
        )


if __name__ == "__main__":
    main()
