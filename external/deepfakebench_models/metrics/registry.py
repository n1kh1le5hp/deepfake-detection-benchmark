"""Minimal stub for DeepfakeBench's metrics.registry so the vendored
network files import standalone (we instantiate classes directly; the
registry decorator is a no-op here)."""


class _Registry:
    def register_module(self, *args, **kwargs):
        def decorator(cls):
            return cls
        return decorator


BACKBONE = _Registry()
