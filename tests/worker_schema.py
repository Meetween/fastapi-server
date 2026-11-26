"""Check a body against the worker's CreateBot schema (`tests/unit/data/vp_worker_create_bot.yaml`).

A small validator for exactly the keywords that schema uses, so the suite needs no new dependency.
A keyword it does not know raises, so a schema that grows a rule cannot pass unchecked.
`\\p{L}` and `\\p{M}`, which Python's `re` lacks, are replaced by explicit ranges that cover the
scripts the tests use (Latin with accents, Greek, Cyrillic, kana, CJK, combining marks).
"""
import base64
import binascii
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

SCHEMA = yaml.safe_load((Path(__file__).parent / "unit" / "data" / "vp_worker_create_bot.yaml").read_text("utf-8"))

_LETTERS = "A-Za-zªµºÀ-ÖØ-öø-ɏͰ-ϿЀ-ԯ぀-ヿ一-鿿"
_MARKS = "̀-ͯ"
_IGNORED = {"default", "description", "title"}
_KNOWN = {
    "type", "required", "properties", "additionalProperties", "pattern", "minLength", "maxLength",
    "enum", "maxItems", "items", "format", "contentEncoding", "contentMediaType",
} | _IGNORED


def _pattern(raw: str) -> re.Pattern:
    return re.compile(raw.replace(r"\p{L}", _LETTERS).replace(r"\p{M}", _MARKS))


def _is_type(value: Any, name: str) -> bool:
    return {
        "string": isinstance(value, str),
        "null": value is None,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "integer": isinstance(value, int) and not isinstance(value, bool),
    }[name]


def violations(value: Any, schema: dict, path: str = "$") -> list[str]:
    unknown = set(schema) - _KNOWN
    if unknown:
        raise AssertionError(f"{path}: the checker does not know {sorted(unknown)}")
    found: list[str] = []
    types = schema.get("type")
    if types is not None:
        names = types if isinstance(types, list) else [types]
        if not any(_is_type(value, n) for n in names):
            return [f"{path}: expected {names}, got {type(value).__name__}"]
    if value is None:
        return found
    if "enum" in schema and value not in schema["enum"]:
        found.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            found.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            found.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not _pattern(schema["pattern"]).search(value):
            found.append(f"{path}: does not match {schema['pattern']}")
        if schema.get("format") == "uri":
            parts = urlsplit(value)
            if not parts.scheme or not parts.netloc:
                found.append(f"{path}: not an absolute uri")
        if schema.get("contentEncoding") == "base64":
            try:
                decoded = base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError):
                found.append(f"{path}: not standard base64")
            else:
                if schema.get("contentMediaType") == "image/jpeg" and not decoded.startswith(b"\xff\xd8\xff"):
                    found.append(f"{path}: not a JPEG")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            found.append(f"{path}: more than {schema['maxItems']} items")
        for i, item in enumerate(value):
            found += violations(item, schema.get("items", {}), f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                found.append(f"{path}: missing {name}")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in props:
                    found.append(f"{path}: unexpected field {name}")
        for name, item in value.items():
            if name in props:
                found += violations(item, props[name], f"{path}.{name}")
    return found


def check_create_bot(body: dict) -> list[str]:
    return violations(body, SCHEMA)
