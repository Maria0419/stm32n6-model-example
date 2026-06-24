from pathlib import Path

import yaml


def _read_yaml(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    return data


def load_train_config(config):
    config_path = Path(config).expanduser().resolve()
    cfg = _read_yaml(config_path)
    return config_path, cfg


def load_export_config(config):
    config_path = Path(config).expanduser().resolve()
    export_cfg = _read_yaml(config_path)
    train_ref = export_cfg.get("train_config")

    train_path = resolve_path(config_path, train_ref)
    train_cfg = _read_yaml(train_path)
    section = export_cfg.get("export", {})

    return config_path, train_path, train_cfg, section


def load_pipeline_config(config, section_name):
    config_path = Path(config).expanduser().resolve()
    cfg = _read_yaml(config_path)
    train_ref = cfg.get("train_config")

    train_path = resolve_path(config_path, train_ref)
    train_cfg = _read_yaml(train_path)
    section = cfg.get(section_name, {})

    return config_path, train_path, train_cfg, section


def resolve_path(config_path, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config_path).parent / path
    return path.resolve()

