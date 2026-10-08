"""Organisation-specific text rules, kept out of the code and out of the repository.

The rules that depend on whose documents are ingested live in a local YAML file (`config/domain.yaml`,
path `KB_DOMAIN_PATH`, git-ignored). The repository ships generic defaults and an example file
(`config/domain.example.yaml`). When the local file sets a key, it **replaces** the default for that key.

    boilerplate_patterns   regexes for legal / footer lines removed wherever they appear
    command_patterns       extra regexes for command-prompt lines merged into code blocks
                           (in addition to the generic SQL, shell, XML and web-server patterns)
    reference_patterns     regexes for knowledge-base article ids copied into answers
    reference_label        what the prompt calls those ids, e.g. "knowledge-base article numbers"
    reference_example      one example id shown to the LLM
    synonym_examples       examples of "different words for the same thing" in the prompt
"""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from kb.core.config import get_settings

DEFAULT_BOILERPLATE = (
    r"^©",
    r"^copyright\b",
    r"^confidential information\b",
    r"^all rights reserved\b",
    r"^this document is provided for information purpose",
    r"^without prior written authori[sz]ation",
)
DEFAULT_REFERENCE_PATTERNS = (r"\bKB\d{6,}\b",)
DEFAULT_REFERENCE_LABEL = "knowledge-base article numbers"
DEFAULT_REFERENCE_EXAMPLE = "KB0012345"
DEFAULT_SYNONYMS = '"SQL queries" and "SQL statements", "licence" and "license", "set up" and "configure"'


class DomainError(Exception):
    """domain.yaml is not valid (bad YAML, wrong types or a regex that does not compile)."""


@dataclass(frozen=True)
class Domain:
    """The active organisation-specific rules."""

    boilerplate_patterns: tuple[str, ...] = DEFAULT_BOILERPLATE
    command_patterns: tuple[str, ...] = ()
    reference_patterns: tuple[str, ...] = DEFAULT_REFERENCE_PATTERNS
    reference_label: str = DEFAULT_REFERENCE_LABEL
    reference_example: str = DEFAULT_REFERENCE_EXAMPLE
    synonym_examples: str = DEFAULT_SYNONYMS

    @property
    def boilerplate(self) -> tuple[re.Pattern, ...]:
        """Compiled boilerplate patterns (case-insensitive)."""
        return _compile(self.boilerplate_patterns, re.IGNORECASE)

    @property
    def references(self) -> tuple[re.Pattern, ...]:
        """Compiled article-id patterns."""
        return _compile(self.reference_patterns, 0)

    def find_references(self, text: str) -> list[str]:
        """Article ids in the text, in order of appearance, without duplicates."""
        found: list[str] = []
        for pattern in self.references:
            for ref in pattern.findall(text):
                if ref not in found:
                    found.append(ref)
        return found


@cache
def _compile(patterns: tuple[str, ...], flags: int) -> tuple[re.Pattern, ...]:
    """Compile a tuple of regexes once."""
    return tuple(re.compile(p, flags) for p in patterns)


_STR_LISTS = ("boilerplate_patterns", "command_patterns", "reference_patterns")
_STRINGS = ("reference_label", "reference_example", "synonym_examples")


def load_domain(path: Path) -> Domain:
    """Read a domain file; missing file → generic defaults. Raises DomainError on invalid content."""
    if not path.exists():
        return Domain()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise DomainError(f"{path.name} is not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise DomainError(f"{path.name} must be a mapping")
    values: dict = {}
    for key in _STR_LISTS:
        if key in data:
            items = data[key] or []
            if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
                raise DomainError(f"{path.name}: '{key}' must be a list of strings")
            for pattern in items:
                try:
                    re.compile(pattern)
                except re.error as e:
                    raise DomainError(f"{path.name}: '{key}' has an invalid regex {pattern!r}: {e}") from e
            values[key] = tuple(items)
    for key in _STRINGS:
        if key in data:
            if not isinstance(data[key], str):
                raise DomainError(f"{path.name}: '{key}' must be a string")
            values[key] = data[key]
    unknown = set(data) - set(_STR_LISTS) - set(_STRINGS)
    if unknown:
        raise DomainError(f"{path.name}: unknown key(s) {', '.join(sorted(unknown))}")
    return Domain(**values)


_override: Domain | None = None


def get_domain() -> Domain:
    """The active rules: an override set by use_domain(), else the file at KB_DOMAIN_PATH (cached)."""
    return _override if _override is not None else _load_cached(get_settings().domain_path)


@cache
def _load_cached(path: Path) -> Domain:
    """load_domain(), once per path."""
    return load_domain(path)


def use_domain(domain: Domain | None) -> None:
    """Make `domain` the active rules (None: back to the file). Used by tests."""
    global _override
    _override = domain
