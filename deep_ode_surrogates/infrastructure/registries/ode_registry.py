# infrastructure/registries/ode_registry.py

from typing import Type
from pydantic import BaseModel

from deep_ode_surrogates.domain.odes.base import BaseODE


class ODERegistry:
    def __init__(self):
        self._odes: dict[str, type[BaseODE]] = {}

    def register(self, name: str, ode_cls: type[BaseODE]) -> None:
        if name in self._odes:
            raise ValueError(f"ODE '{name}' is already registered.")

        self._odes[name] = ode_cls

    def get(self, name: str) -> type[BaseODE]:
        try:
            return self._odes[name]
        except KeyError:
            available = ", ".join(self._odes.keys())
            raise ValueError(
                f"Unknown ODE '{name}'. Available ODEs: {available}"
            )

    def create(self, name: str, params: BaseModel) -> BaseODE:
        ode_cls = self.get(name)
        return ode_cls(params=params)

    def available(self) -> list[str]:
        return sorted(self._odes.keys())


ode_registry = ODERegistry()


def register_ode(name: str):
    def decorator(ode_cls: type[BaseODE]):
        ode_registry.register(name, ode_cls)
        return ode_cls

    return decorator