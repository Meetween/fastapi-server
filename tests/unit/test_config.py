"""Unit tests for settings / model profile loading."""

from pathlib import Path

import pytest
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


def test_settings_nested_views_are_cached() -> None:
    s = Settings(model_profile="speechlmm")

    assert s.app_jwt is s.app_jwt
    assert s.meeting_jwt is s.meeting_jwt
    assert s.mongo is s.mongo
    assert s.triton is s.triton


def test_closed_caption_unset_by_default(tmp_path: Path) -> None:
    s = Settings(model_profile="speechlmm", closed_caption_model_profile=None)

    assert s.closed_caption_triton is None


def test_closed_caption_profile_resolves_from_yaml(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.yml"
    profiles_path.write_text(
        yaml.dump(
            {
                "profiles": {
                    "speechlmm": {"triton": {"url": "http://primary:9100", "model_name": "p"}},
                    "voxtral_mini_3b_2507": {
                        "triton": {
                            "url": "http://voxtral:9000",
                            "model_name": "voxtral_mini_3b_2507",
                            "prompt_protocol": "general_json",
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    s = Settings(
        model_profile="speechlmm",
        model_profiles_path=profiles_path,
        closed_caption_model_profile="voxtral_mini_3b_2507",
    )

    assert s.triton.url == "http://primary:9100"
    assert s.closed_caption_triton is not None
    assert s.closed_caption_triton.url == "http://voxtral:9000"
    assert s.closed_caption_triton.model_name == "voxtral_mini_3b_2507"
    assert s.closed_caption_triton.caption_chars == 60


def test_closed_caption_unknown_profile_fails_loudly(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.yml"
    profiles_path.write_text(
        yaml.dump({"profiles": {"speechlmm": {"triton": {"url": "http://x"}}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CLOSED_CAPTION_MODEL_PROFILE='typo'"):
        Settings(
            model_profile="speechlmm",
            model_profiles_path=profiles_path,
            closed_caption_model_profile="typo",
        )
