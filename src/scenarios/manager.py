import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import yaml

from ..core.environment_3d import MultiAgentEnv


def _to_namespace(data: Union[dict, SimpleNamespace]) -> SimpleNamespace:
    if isinstance(data, SimpleNamespace):
        return data
    return SimpleNamespace(**data)


def load_config(path: Union[str, Path]) -> SimpleNamespace:
    """Load a YAML scenario config into a SimpleNamespace."""
    cfg_path = Path(path)
    cfg = _to_namespace(yaml.safe_load(cfg_path.read_text()))
    setattr(cfg, "_cfg_dir", cfg_path.parent)
    return cfg


def make_env(config: Optional[Union[dict, SimpleNamespace]] = None,
             config_path: Optional[Union[str, Path]] = None):
    """
    Build a MultiAgentEnv for the requested scenario.

    Either provide ``config`` as a dict/namespace or a ``config_path`` to
    a YAML file. When neither is given, defaults to the bundled TAD config.
    """
    if config_path:
        cfg = load_config(config_path)
    elif config is not None:
        cfg = _to_namespace(config)
    else:
        default_cfg_path = Path(__file__).resolve().parents[2] / "config" / "tad.yaml"
        cfg = load_config(default_cfg_path)

    if not hasattr(cfg, "_cfg_dir"):
        cfg._cfg_dir = Path(config_path).parent if config_path else Path(__file__).resolve().parents[2]

    scenario_name = getattr(cfg, "scenario_name", "tad")
    scenario_mod = importlib.import_module(f"{__package__}.{scenario_name}")
    scenario = scenario_mod.Scenario()

    world = scenario.make_world(cfg)
    return MultiAgentEnv(world,
                         scenario.reset_world,
                         scenario.reward,
                         scenario.observation,
                         scenario.info,
                         scenario.done,
                         getattr(scenario, "update_belief", None))
