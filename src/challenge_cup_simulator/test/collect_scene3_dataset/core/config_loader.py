"""
core/config_loader.py — YAML config loading, merging, path resolution.

Shared between the main collector and the lower-tray pick scripts.
"""

import copy
import os

import yaml

# Repository root: 4 levels up from this file
# core/config_loader.py → core/ → collect_scene3_dataset/ → test/ → challenge_cup_simulator/ → src/ → repo
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CORE_DIR, "..", "..", "..", "..", ".."))
SCRIPT_DIR = os.path.dirname(_CORE_DIR)  # collect_scene3_dataset/


def deep_update(dst, src):
    """Recursively merge src dict into dst dict."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_yaml(path):
    """Load a YAML file, returning an empty dict on missing/corrupt files."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path, data):
    """Write dict as YAML, creating parent directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def compact_yaml(data):
    """Inline YAML for logging."""
    if not data:
        return ""
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=True, width=4096).strip()


def ensure_repo_relative(path):
    """Convert a relative path to absolute, rooted at the workspace repo root."""
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(_REPO_ROOT, path))


def resolve_pose_config(script_dir, cfg):
    """Resolve absolute path to the named-pose YAML config."""
    path = cfg.get("expert", {}).get("pose_config", "configs/scene3_named_poses.yaml")
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(script_dir, path))


# Default config paths (relative to SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "configs", "scene3_collect.yaml")
DEFAULT_ROI_CONFIG = os.path.join(SCRIPT_DIR, "configs", "scene3_roi.yaml")
