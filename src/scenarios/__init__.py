def make_env(*args, **kwargs):
    # Local import to avoid circular dependency when utilities under
    # src.scenarios are imported from core modules.
    from .manager import make_env as _make_env
    return _make_env(*args, **kwargs)


def load_config(*args, **kwargs):
    from .manager import load_config as _load_config
    return _load_config(*args, **kwargs)


__all__ = ['make_env', 'load_config']

