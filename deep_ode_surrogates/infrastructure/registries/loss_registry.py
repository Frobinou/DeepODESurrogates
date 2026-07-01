class LossRegistry:
    def __init__(self):
        self._losses = {}

    def register(self, name: str, loss_cls):
        self._losses[name] = loss_cls

    def create(self, name: str, **kwargs):
        return self._losses[name](**kwargs)

    def available(self):
        return sorted(self._losses.keys())


loss_registry = LossRegistry()


def register_loss(name: str):
    def decorator(loss_cls):
        loss_registry.register(name, loss_cls)
        return loss_cls

    return decorator
