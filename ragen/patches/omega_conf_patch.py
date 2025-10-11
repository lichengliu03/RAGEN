"""Monkey patch `omega_conf_to_dataclass` to allow extra fields on VERL configs.

The upstream implementation instantiates dataclasses directly and therefore
rejects any unexpected keyword arguments. RAGEN users often want to attach
custom flags in YAML without editing VERL source code. This patch splits the
input config into two parts:

* fields belonging to the underlying dataclass → delegated to the original
  implementation, preserving all validation behaviour;
* any additional fields → attached to the resulting object via
  ``object.__setattr__`` so they remain accessible.

The patch is idempotent and can be safely applied multiple times.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, Tuple

from hydra.utils import get_class
from omegaconf import DictConfig, ListConfig, OmegaConf

from verl.utils import config as verl_config


def apply_omega_conf_patch() -> None:
    """Apply the RAGEN override of ``omega_conf_to_dataclass`` if not present."""

    if getattr(verl_config, "_ragen_omega_conf_patch", False):
        return

    original_fn = verl_config.omega_conf_to_dataclass

    def _split_known_and_extra(
        cfg: DictConfig | dict,
        valid_fields: set[str],
        include_target: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split config into dataclass fields and extra attributes."""

        base: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}

        if isinstance(cfg, DictConfig):
            iterator = cfg.items()
        elif isinstance(cfg, dict):
            iterator = cfg.items()
        else:
            return base, extras

        for key, value in iterator:
            if include_target and key == "_target_":
                base[key] = value
            elif key in valid_fields:
                base[key] = value
            else:
                extras[key] = value
        return base, extras

    def _attach_extras(obj: Any, extras: Dict[str, Any]) -> None:
        for key, value in extras.items():
            object.__setattr__(obj, key, value)

    def _resolve_dataclass(target: Any) -> Any:
        if not isinstance(target, str):
            return None
        try:
            cls = get_class(target)
        except Exception:
            return None
        return cls if is_dataclass(cls) else None

    def patched_omega_conf_to_dataclass(
        config: DictConfig | dict | None,
        dataclass_type: Any | None = None,
    ) -> Any:
        # Delegate directly if no dataclass is involved.
        if dataclass_type is not None:
            if not is_dataclass(dataclass_type):
                return original_fn(config, dataclass_type)

            if config is None:
                return original_fn(config, dataclass_type)

            base_dict, extras = _split_known_and_extra(config, {f.name for f in fields(dataclass_type)}, False)
            base_cfg = OmegaConf.create(base_dict)
            obj = original_fn(base_cfg, dataclass_type)
            _attach_extras(obj, extras)
            return obj

        # No explicit dataclass type provided; rely on _target_.
        if config is None:
            return original_fn(config, dataclass_type)

        target = config.get("_target_") if isinstance(config, (DictConfig, dict)) else None

        dataclass_cls = _resolve_dataclass(target)
        if dataclass_cls is None:
            return original_fn(config, dataclass_type)

        valid_fields = {f.name for f in fields(dataclass_cls)}
        base_dict, extras = _split_known_and_extra(config, valid_fields, include_target=True)
        base_cfg = OmegaConf.create(base_dict)
        obj = original_fn(base_cfg, dataclass_type)
        _attach_extras(obj, extras)
        return obj

    verl_config.omega_conf_to_dataclass = patched_omega_conf_to_dataclass
    verl_config._ragen_omega_conf_patch = True


__all__ = ["apply_omega_conf_patch"]
