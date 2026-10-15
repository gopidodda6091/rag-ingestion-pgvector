"""Regex + rule PII redaction with an audit trail.

The audit log records the PII type, the rule that fired, why, and an HMAC-SHA256 of the value
(keyed by REDACTION_HMAC_KEY), never the raw value. A keyed hash matters because SSNs have a
small keyspace: a plain SHA-256 of one could be brute-forced.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    name: str
    pii_type: str
    pattern: re.Pattern[str]
    reason: str
    group: int = 0  # which regex group holds the value to redact


RULES: tuple[Rule, ...] = (
    Rule("email", "EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "e-mail address"),
    Rule("ssn", "SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN-shaped number ddd-dd-dddd"),
    Rule(
        "phone",
        "PHONE",
        re.compile(r"(?:\(\d{3}\)\s?|\b\d{3}[-.\s])\d{3}[-.]\d{4}\b"),
        "US phone number",
    ),
    Rule("member_id", "MEMBER_ID", re.compile(r"\bPHP-\d{8}\b"), "plan member identifier"),
    Rule(
        "dob",
        "DOB",
        re.compile(r"\b(?:DOB|date of birth)[:\s]+(\d{1,2}/\d{1,2}/\d{4})", re.I),
        "date following a DOB cue",
        group=1,
    ),
    Rule(
        "titled_name",
        "PERSON",
        re.compile(r"\b(?:Mr|Ms|Mrs|Dr)\.\s+([A-Z][a-z]+\s+[A-Z][a-z]+)"),
        "capitalised name after an honorific",
        group=1,
    ),
    Rule(
        "member_name",
        "PERSON",
        re.compile(r"\bmember\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b"),
        "capitalised name after the word 'member'",
        group=1,
    ),
)


@dataclass(frozen=True)
class AuditEntry:
    doc_id: str
    location: str
    pii_type: str
    rule: str
    reason: str
    value_hmac: str
    length: int


def value_digest(value: str, key: bytes) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def hmac_key() -> bytes:
    return os.getenv("REDACTION_HMAC_KEY", "local-dev-only-key").encode("utf-8")


def find_spans(text: str, rules: tuple[Rule, ...] = RULES) -> list[tuple[int, int, Rule]]:
    """Non-overlapping matches; earlier rules win ties, longer spans win overlaps."""
    spans: list[tuple[int, int, Rule]] = []
    for rule in rules:
        for match in rule.pattern.finditer(text):
            spans.append((match.start(rule.group), match.end(rule.group), rule))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    kept: list[tuple[int, int, Rule]] = []
    for span in spans:
        if not kept or span[0] >= kept[-1][1]:
            kept.append(span)
    return kept


def redact_text(
    text: str, doc_id: str, location: str, key: bytes | None = None
) -> tuple[str, list[AuditEntry]]:
    key = key or hmac_key()
    spans = find_spans(text)
    entries, out, cursor = [], [], 0
    for start, end, rule in spans:
        value = text[start:end]
        out += [text[cursor:start], f"[{rule.pii_type}]"]
        cursor = end
        entries.append(
            AuditEntry(
                doc_id,
                location,
                rule.pii_type,
                rule.name,
                rule.reason,
                value_digest(value, key),
                len(value),
            )
        )
    out.append(text[cursor:])
    return "".join(out), entries


class AuditLog:
    """Append-only JSONL audit log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entries: list[AuditEntry], run_id: str) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps({"run_id": run_id, **asdict(entry)}) + "\n")
