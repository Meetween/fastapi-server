"""
Utility for loading and using YAML templates.
"""

from pathlib import Path
from typing import Any

import yaml


class PromptLoader:
    """Single source prompt loader backed by one consolidated YAML file."""

    PROMPTS_DIR = Path(__file__).parent.parent / "butler" / "prompts"
    DEFAULT_FILENAME = "prompts.yml"
    DEFAULT_TASK_PROMPT = "Process the following:"
    _cache: dict[str, dict[str, Any]] = {}

    LANGUAGE_NAMES = {
        "en": "English",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "es": "Spanish",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "ar": "Arabic",
        "hi": "Hindi",
    }

    @classmethod
    def _resolve_path(cls, filename: str | None = None) -> Path:
        selected = filename or cls.DEFAULT_FILENAME
        candidate = Path(selected)
        return candidate if candidate.is_absolute() else cls.PROMPTS_DIR / selected

    @classmethod
    def load(cls, filename: str | None = None) -> dict[str, Any]:
        path = cls._resolve_path(filename)
        cache_key = str(path)
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Template YAML must be a mapping: {path}")
        cls._cache[cache_key] = data
        return data

    @classmethod
    def get_template(cls, template_name: str, filename: str | None = None) -> str:
        data = cls.load(filename)
        if template_name in data:
            template = data[template_name]
        else:
            templates = data.get("templates", {})
            if template_name not in templates:
                raise KeyError(f"Template '{template_name}' not found")
            template = templates[template_name]
        if not isinstance(template, str):
            raise ValueError(f"Template '{template_name}' must be a string")
        return template

    @classmethod
    def format_template(
        cls,
        template_name: str,
        filename: str | None = None,
        **kwargs: Any,
    ) -> str:
        return cls.get_template(template_name, filename=filename).format(**kwargs)

    @classmethod
    def get_task_prompt(cls, task: str, language: str, target_language: str = None,  filename: str | None = None) -> str:
        data = cls.load(filename)
        prompts = data.get("task_prompts", {})
        if not isinstance(prompts, dict):
            raise ValueError("Prompt YAML 'task_prompts' must be a mapping")

        lang_code = cls._normalize_language_code(language)
        target_lang_code = cls._normalize_language_code(target_language) if target_language is not None else None
        task_prompts = prompts.get(task) or prompts.get("ASR") or {}
        if not isinstance(task_prompts, dict) or not task_prompts:
            return cls.DEFAULT_TASK_PROMPT

        lang_prompts = cls._select_language_prompts(task_prompts, lang_code)
        selected = cls._select_prompt_variant(lang_prompts)
        source_lang_name = cls._get_language_name(lang_code)
        target_lang_name = cls._get_language_name(target_lang_code) if target_lang_code is not None else cls._get_language_name(lang_code)
        return (
            selected
            .replace("<target-language>", target_lang_name)
            .replace("<source-language>", source_lang_name)
        )

    @classmethod
    def _normalize_language_code(cls, language: str) -> str:
        return language.split("-")[0].lower() if "-" in language else language.lower()

    @classmethod
    def _get_language_name(cls, lang_code: str) -> str:
        """Convert language code to readable name."""
        return cls.LANGUAGE_NAMES.get(lang_code.lower(), lang_code.capitalize())

    @classmethod
    def _select_language_prompts(cls, task_prompts: dict[str, Any], lang_code: str) -> Any:
        if lang_code in task_prompts:
            return task_prompts[lang_code]
        if "en" in task_prompts:
            return task_prompts["en"]
        return next(iter(task_prompts.values()))

    @classmethod
    def _select_prompt_variant(cls, lang_prompts: Any) -> str:
        if isinstance(lang_prompts, list):
            if not lang_prompts:
                return cls.DEFAULT_TASK_PROMPT
            with_source = [p for p in lang_prompts if "<source-language>" in p]
            return (with_source or lang_prompts)[0]
        return str(lang_prompts)


def format_template(template_name: str, filename: str | None = None, **kwargs: Any) -> str:
    return PromptLoader.format_template(template_name, filename=filename, **kwargs)


def get_task_prompt(task: str, language: str, target_language: str = None, filename: str | None = None) -> str:
    return PromptLoader.get_task_prompt(task, language, target_language=target_language, filename=filename)
