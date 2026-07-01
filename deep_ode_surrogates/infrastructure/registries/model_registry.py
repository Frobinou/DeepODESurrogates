class ModelRegistry:
    def __init__(self):
        self._models = {}

    def register(self, name: str, model_cls):
        self._models[name] = model_cls

    def create(self, name: str, **kwargs):
        return self._models[name](**kwargs)

    def available(self):
        return sorted(self._models.keys())


model_registry = ModelRegistry()


def register_model(name: str):
    def decorator(model_cls):
        model_registry.register(name, model_cls)
        return model_cls

    return decorator
