"""Deterministic synthetic geography for Phase 6 algorithm tests only."""
from __future__ import annotations

from dataclasses import dataclass, replace

from backend.app.gis.surface import SurfaceClassification, UNKNOWN_SURFACE


@dataclass(frozen=True, order=True)
class GridPoint:
    x: int
    y: int


@dataclass(frozen=True)
class GridCell:
    point: GridPoint
    elevation_m: float = 0.0
    terrain: str = "flat"
    terrain_multiplier: float = 1.0
    blocked: bool = False
    cell_size_m: float | None = None
    surface: SurfaceClassification = UNKNOWN_SURFACE

    def __post_init__(self) -> None:
        if self.terrain_multiplier < 1.0:
            raise ValueError("terrain_multiplier must be at least 1")
        if self.cell_size_m is not None and self.cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive when supplied")


class SyntheticGrid:
    """Small in-memory grid; grid units are interpreted as kilometres by the v1 engine."""

    def __init__(self, width: int, height: int, *, default_elevation_m: float = 0.0):
        if width <= 0 or height <= 0:
            raise ValueError("grid dimensions must be positive")
        self.width, self.height = width, height
        self._cells = {
            GridPoint(x, y): GridCell(GridPoint(x, y), elevation_m=default_elevation_m)
            for y in range(height)
            for x in range(width)
        }

    @classmethod
    def flat(cls, width: int, height: int) -> "SyntheticGrid":
        return cls(width, height)

    def contains(self, point: GridPoint) -> bool:
        return point in self._cells

    def cell(self, point: GridPoint) -> GridCell:
        try:
            return self._cells[point]
        except KeyError as exc:
            raise ValueError(f"point outside synthetic grid: {point}") from exc

    def set_cell(
        self,
        point: GridPoint,
        *,
        elevation_m: float | None = None,
        terrain: str | None = None,
        terrain_multiplier: float | None = None,
        blocked: bool | None = None,
        cell_size_m: float | None = None,
        surface: SurfaceClassification | None = None,
    ) -> None:
        current = self.cell(point)
        self._cells[point] = replace(
            current,
            elevation_m=current.elevation_m if elevation_m is None else elevation_m,
            terrain=current.terrain if terrain is None else terrain,
            terrain_multiplier=current.terrain_multiplier if terrain_multiplier is None else terrain_multiplier,
            blocked=current.blocked if blocked is None else blocked,
            cell_size_m=current.cell_size_m if cell_size_m is None else cell_size_m,
            surface=current.surface if surface is None else surface,
        )

    def neighbors4(self, point: GridPoint) -> tuple[GridPoint, ...]:
        candidates = (GridPoint(point.x, point.y - 1), GridPoint(point.x - 1, point.y), GridPoint(point.x + 1, point.y), GridPoint(point.x, point.y + 1))
        return tuple(candidate for candidate in candidates if self.contains(candidate))

    def neighbors8(self, point: GridPoint) -> tuple[GridPoint, ...]:
        candidates = (
            GridPoint(point.x - 1, point.y - 1), GridPoint(point.x, point.y - 1), GridPoint(point.x + 1, point.y - 1),
            GridPoint(point.x - 1, point.y), GridPoint(point.x + 1, point.y),
            GridPoint(point.x - 1, point.y + 1), GridPoint(point.x, point.y + 1), GridPoint(point.x + 1, point.y + 1),
        )
        return tuple(candidate for candidate in candidates if self.contains(candidate))
