"""
Utility for loading and using YAML templates.
"""

from pathlib import Path
from typing import Dict, Any

import yaml

# Directory of the templates folder
TEMPLATES_DIR = Path(__file__).parent.parent / "butler" / "prompts"


def _resolve_template_path(filename: str) -> Path:
    candidate = Path(filename)
    return candidate if candidate.is_absolute() else TEMPLATES_DIR / filename


def load_yaml_template(filename: str) -> Dict[str, Any]:
    """
    Load a YAML template file.
    """
    template_path = _resolve_template_path(filename)
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Template YAML must be a mapping: {template_path}")
    return data


def get_template(template_name: str, filename: str = "prompts.yaml") -> str:
    """
    Get a specific template from a YAML file.
    """
    templates = load_yaml_template(filename)
    if template_name not in templates:
        raise KeyError(f"Template '{template_name}' not found in {filename}")
    template = templates[template_name]
    if not isinstance(template, str):
        raise ValueError(f"Template '{template_name}' must be a string in {filename}")
    return template


def format_template(template_name: str, filename: str = None, **kwargs: Any) -> str:
    """
    Load and format a template with the given parameters.
    """
    if filename:
        template = get_template(template_name, filename=filename)
    else:
        template = get_template(template_name)
    return template.format(**kwargs)
