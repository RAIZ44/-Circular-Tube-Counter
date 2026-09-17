from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _dominant_row_angle(points: np.ndarray) -> float:
    if len(points) < 4:
        return 0.0
    deltas = points[None, :, :] - points[:, None, :]
    distances = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbor_count = min(6, len(points) - 1)
    neighbor_indices = np.argpartition(distances, neighbor_count - 1, axis=1)[
        :, :neighbor_count
    ]
    angles: list[float] = []
    for row, indices in enumerate(neighbor_indices):
        for index in indices:
            dx, dy = deltas[row, index]
            if dx < 0:
                dx, dy = -dx, -dy
            angle = float(np.degrees(np.arctan2(dy, dx)))
            if abs(angle) <= 35.0:
                angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def _rotate(points: np.ndarray, angle_degrees: float) -> np.ndarray:
    center = points.mean(axis=0)
    radians = np.radians(-angle_degrees)
    rotation = np.asarray(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]],
        dtype=np.float64,
    )
    return (points - center) @ rotation.T + center


def analyze_lattice(
    centers: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Cluster staggered rows and flag unusually large within-row gaps.

    Geometry warnings are intentionally separate from the detected count.
    """
    points = np.asarray(centers, dtype=np.float64)
    result: dict[str, Any] = {
        "detected_count": int(len(points)),
        "possible_gap_count": 0,
        "possible_gap_points": [],
        "rows": [],
        "row_angle_degrees": 0.0,
        "nearest_neighbor_spacing": None,
        "horizontal_spacing": None,
    }
    if len(points) < 4:
        return result

    deltas = points[None, :, :] - points[:, None, :]
    distances = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    spacing = float(np.median(nearest[np.isfinite(nearest)]))
    if not np.isfinite(spacing) or spacing <= 1.0:
        return result

    angle = _dominant_row_angle(points)
    leveled = _rotate(points, angle)
    row_tolerance = max(3.0, 0.48 * spacing)

    rows: list[list[int]] = []
    row_means: list[float] = []
    for index in np.argsort(leveled[:, 1]):
        y = float(leveled[index, 1])
        if not rows:
            rows.append([int(index)])
            row_means.append(y)
            continue
        nearest_row = int(np.argmin(np.abs(np.asarray(row_means) - y)))
        if abs(row_means[nearest_row] - y) <= row_tolerance:
            rows[nearest_row].append(int(index))
            row_means[nearest_row] = float(np.mean(leveled[rows[nearest_row], 1]))
        else:
            rows.append([int(index)])
            row_means.append(y)

    ordered_rows = [
        row
        for _, row in sorted(zip(row_means, rows, strict=True), key=lambda pair: pair[0])
    ]
    candidate_gaps: list[float] = []
    for row in ordered_rows:
        xs = np.sort(leveled[row, 0])
        if len(xs) > 1:
            candidate_gaps.extend(np.diff(xs).tolist())
    plausible = [gap for gap in candidate_gaps if 0.55 * spacing <= gap <= 1.55 * spacing]
    horizontal_spacing = float(np.median(plausible)) if plausible else spacing

    possible_points: list[list[float]] = []
    row_summaries: list[dict[str, Any]] = []
    for row_number, row in enumerate(ordered_rows, start=1):
        ordered = sorted(row, key=lambda index: leveled[index, 0])
        suspicious_in_row = 0
        for left_index, right_index in zip(ordered[:-1], ordered[1:], strict=True):
            gap = float(leveled[right_index, 0] - leveled[left_index, 0])
            if 1.65 * horizontal_spacing < gap < 4.6 * horizontal_spacing:
                missing = max(1, int(round(gap / horizontal_spacing)) - 1)
                for step in range(1, missing + 1):
                    fraction = step / (missing + 1)
                    point = points[left_index] * (1.0 - fraction) + points[right_index] * fraction
                    possible_points.append([float(point[0]), float(point[1])])
                    suspicious_in_row += 1
        row_summaries.append(
            {
                "row": row_number,
                "detected": len(row),
                "possible_gaps": suspicious_in_row,
            }
        )

    result.update(
        {
            "possible_gap_count": len(possible_points),
            "possible_gap_points": possible_points,
            "rows": row_summaries,
            "row_angle_degrees": angle,
            "nearest_neighbor_spacing": spacing,
            "horizontal_spacing": horizontal_spacing,
        }
    )
    return result

