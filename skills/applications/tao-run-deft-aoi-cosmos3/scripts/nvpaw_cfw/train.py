"""Entrypoint that registers NVPAW, then delegates to Framework training."""

from __future__ import annotations

import importlib
import runpy


def install_registration_hook() -> None:
    from cosmos_framework.configs.base.vlm import config as base_vlm_config

    original_make_config = base_vlm_config.make_config

    def make_config_then_register():
        config = original_make_config()
        from . import processor
        from . import experiment

        importlib.reload(processor)
        importlib.reload(experiment).register_experiment()
        return config

    base_vlm_config.make_config = make_config_then_register


def main() -> None:
    install_registration_hook()
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
