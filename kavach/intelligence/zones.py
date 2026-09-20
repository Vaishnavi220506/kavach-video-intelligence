"""Configurable named zones built from KAVACH geometry primitives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .geometry import (
    Polygon,
    distance_to_region,
    point_inside_polygon,
    validate_polygon,
)

STAGING_ZONE = "STAGING_ZONE"
LOADING_ZONE = "LOADING_ZONE"
RESTRICTED_ZONE = "RESTRICTED_ZONE"
PALLET_ZONE = "PALLET_ZONE"
ZONE_NAMES = (STAGING_ZONE, LOADING_ZONE, RESTRICTED_ZONE, PALLET_ZONE)


@dataclass(frozen=True)
class Zone:
    """One named polygon and its BGR visualization color."""

    name: str
    polygon: Polygon
    color: tuple[int, int, int] = (0, 255, 0)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("zone name cannot be empty")
        normalized = validate_polygon(self.polygon)
        if len(self.color) != 3 or not all(0 <= int(channel) <= 255 for channel in self.color):
            raise ValueError("zone color must contain three channels from 0 to 255")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "polygon", normalized)
        object.__setattr__(self, "color", tuple(int(channel) for channel in self.color))

    def contains(self, point: Sequence[float]) -> bool:
        """Return whether a point lies inside or on the zone boundary."""

        return point_inside_polygon(point, self.polygon)

    def distance_to_boundary(self, point: Sequence[float]) -> float:
        """Return zero inside, otherwise pixel distance to the zone boundary."""

        return distance_to_region(point, self.polygon)


class ZoneManager:
    """Store and query configurable named polygon zones."""

    def __init__(
        self,
        zones: Mapping[str, Sequence[Sequence[float]]] | Iterable[Zone] | None = None,
    ) -> None:
        self._zones: dict[str, Zone] = {}
        if zones is None:
            return
        if isinstance(zones, Mapping):
            for name, polygon in zones.items():
                self.set_zone(name, polygon)
        else:
            for zone in zones:
                self.add_zone(zone)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Mapping[str, object] | Sequence[Sequence[float]]],
    ) -> ZoneManager:
        """Build zones from JSON-like polygon or polygon/color configuration."""

        manager = cls()
        for name, value in config.items():
            if isinstance(value, Mapping):
                polygon = value.get("polygon")
                if polygon is None:
                    raise ValueError(f"zone '{name}' is missing polygon")
                color = value.get("color", (0, 255, 0))
                manager.set_zone(name, polygon, color=color)  # type: ignore[arg-type]
            else:
                manager.set_zone(name, value)
        return manager

    def add_zone(self, zone: Zone) -> None:
        """Add or replace a complete Zone object."""

        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone instance")
        self._zones[zone.name] = zone

    def set_zone(
        self,
        name: str,
        polygon: Sequence[Sequence[float]],
        *,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> Zone:
        """Configure or replace a named polygon zone."""

        zone = Zone(name=str(name), polygon=validate_polygon(polygon), color=color)
        self.add_zone(zone)
        return zone

    def get_zone(self, name: str) -> Zone | None:
        """Return a configured zone, or None when absent."""

        return self._zones.get(str(name))

    @property
    def zones(self) -> tuple[Zone, ...]:
        """Return configured zones in insertion order."""

        return tuple(self._zones.values())

    def contains(self, zone_name: str, point: Sequence[float]) -> bool:
        """Return whether a point belongs to a named zone."""

        zone = self.get_zone(zone_name)
        if zone is None:
            raise KeyError(f"unknown zone: {zone_name}")
        return zone.contains(point)

    def distance_to_zone(self, zone_name: str, point: Sequence[float]) -> float:
        """Return distance in pixels from a point to a named zone region."""

        zone = self.get_zone(zone_name)
        if zone is None:
            raise KeyError(f"unknown zone: {zone_name}")
        return zone.distance_to_boundary(point)

    def zone_for_point(self, point: Sequence[float]) -> str | None:
        """Return the first matching zone, or None if no polygon contains it."""

        for zone in self._zones.values():
            if zone.contains(point):
                return zone.name
        return None

