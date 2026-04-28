"""Méthodes d'estimation de la carte probabiliste."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from aphid_spatial.simulation.field import Field
from aphid_spatial.simulation.sensors import SensorReadings


@runtime_checkable
class SpatialMethod(Protocol):
    """Interface uniforme pour toutes les méthodes spatiales.

    Une méthode est ajustée à partir des observations des capteurs
    (:meth:`fit`) puis interrogée sur des coordonnées arbitraires
    (:meth:`predict_proba`, :meth:`predict_uncertainty`).
    """

    name: str

    def fit(self, readings: SensorReadings, field_meta: Field) -> None: ...
    def predict_proba(self, query_coords: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def predict_uncertainty(
        self, query_coords: NDArray[np.floating]
    ) -> NDArray[np.float64] | None: ...


__all__ = ["SpatialMethod"]
