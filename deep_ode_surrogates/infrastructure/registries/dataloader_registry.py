# infrastructure/registries/dataloader_registry.py

class DataloaderRegistry:
    def __init__(self):
        self._dataloaders = {}

    def register(self, name: str, dataloader_cls):
        self._dataloaders[name] = dataloader_cls

    def create(self, name: str, data_config):
        if name not in self._dataloaders:
            raise ValueError(
                f"Unknown dataloader '{name}'. Available: {self.available()}"
            )
        return self._dataloaders[name](data_config=data_config)

    def available(self):
        return sorted(self._dataloaders.keys())


dataloader_registry = DataloaderRegistry()

def register_dataloader(name: str):
    def decorator(cls):
        dataloader_registry.register(name, cls)
        return cls
    return decorator