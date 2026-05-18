"""Unit tests for settings / model profile loading."""

from pathlib import Path

import yaml

from config import (
    Settings,
    _build_triton_settings,
    _load_triton_profile_overrides,
)


def test_load_triton_profile_overrides_from_yaml(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.yml"
    profiles_path.write_text(
        yaml.dump(
            {
                "profiles": {
                    "qwen25_omni": {
                        "triton": {
                            "url": "http://triton:9000",
                            "model_name": "qwen25_omni",
                            "prompt_protocol": "general_json",
                            "caption_chars": 42,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    profile = _load_triton_profile_overrides(
        model_profile="qwen25_omni",
        profiles_path=profiles_path,
    )

    assert profile["url"] == "http://triton:9000"
    assert profile["model_name"] == "qwen25_omni"
    assert profile["prompt_protocol"] == "general_json"
    assert profile["caption_chars"] == 42


def test_load_triton_profile_unknown_profile_uses_defaults(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.yml"
    profiles_path.write_text(
        yaml.dump({"profiles": {"speechlmm": {"triton": {"url": "http://x"}}}}),
        encoding="utf-8",
    )

    profile = _load_triton_profile_overrides(
        model_profile="missing",
        profiles_path=profiles_path,
    )

    assert profile["url"] == "http://localhost:9100"
    assert profile["model_name"] == "speechlmm"


def test_build_triton_settings_env_limits_override_caption_default() -> None:
    triton = _build_triton_settings(
        profile={"url": "http://a", "model_name": "m", "prompt_protocol": "p"},
        timeout=99,
        max_context_chars=1,
        max_history_turns=2,
        max_history_chars=3,
        caption_chars_default=60,
    )

    assert triton.timeout == 99
    assert triton.max_context_chars == 1
    assert triton.caption_chars == 60


def test_build_triton_settings_profile_caption_override() -> None:
    triton = _build_triton_settings(
        profile={
            "url": "http://a",
            "model_name": "m",
            "prompt_protocol": "p",
            "caption_chars": 120,
        },
        timeout=1800,
        max_context_chars=300_000,
        max_history_turns=400,
        max_history_chars=600_000,
        caption_chars_default=60,
    )

    assert triton.caption_chars == 120


def test_settings_triton_resolves_speechlmm_profile() -> None:
    s = Settings(model_profile="speechlmm")

    assert s.triton.url == "http://localhost:9100"
    assert s.triton.model_name == "speechlmm"
    assert s.triton.prompt_protocol == "speechlmm_plain"
