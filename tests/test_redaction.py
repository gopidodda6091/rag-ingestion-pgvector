import json
import re

from rag_ingest.ingest import ingest
from rag_ingest.redact import redact_text
from rag_ingest.store import InMemoryStore

RAW_PII = [
    "Jordan Sampleton",
    "900-12-3456",
    "(555) 010-4477",
    "jordan.sampleton@example.com",
    "PHP-00012345",
    "Casey Placeholder",
    "02/14/1985",
    "912-34-5678",
    "555-010-9921",
    "Riley Fictional",
]


def test_each_pii_type_is_redacted() -> None:
    text = (
        "Example: member Jordan Sampleton (SSN 900-12-3456, phone (555) 010-4477, "
        "jordan.sampleton@example.com, member ID PHP-00012345). Ms. Casey Placeholder, "
        "DOB 02/14/1985, called 555-010-9921."
    )
    redacted, audit = redact_text(text, "doc", "doc::s", key=b"k")
    for value in RAW_PII[:8]:
        assert value not in redacted
    assert {e.pii_type for e in audit} == {"PERSON", "SSN", "PHONE", "EMAIL", "MEMBER_ID", "DOB"}
    assert "member [PERSON]" in redacted and "DOB [DOB]" in redacted


def test_non_pii_is_left_alone() -> None:
    text = "Tier 2 costs $35. Claims are due within 180 days. Member Services opens at 8 a.m."
    redacted, audit = redact_text(text, "doc", "doc::s", key=b"k")
    assert redacted == text and audit == []


def test_audit_hash_is_keyed_and_stable() -> None:
    _, a = redact_text("SSN 900-12-3456", "d", "l", key=b"k1")
    _, b = redact_text("SSN 900-12-3456", "d", "l", key=b"k1")
    _, c = redact_text("SSN 900-12-3456", "d", "l", key=b"k2")
    assert a[0].value_hmac == b[0].value_hmac != c[0].value_hmac


def test_audit_log_and_store_contain_no_raw_pii(corpus, embedder, tmp_path) -> None:
    store, audit_path = InMemoryStore(), tmp_path / "audit.jsonl"
    report = ingest(corpus, store, embedder, audit_path)
    assert report.redactions >= 10
    log = audit_path.read_text()
    entries = [json.loads(line) for line in log.splitlines()]
    assert {"pii_type", "rule", "reason", "value_hmac"} <= set(entries[0])
    stored_text = "\n".join(c.text for c in store.chunks())
    for value in RAW_PII:
        assert value not in log, f"raw PII in audit log: {value}"
        assert value not in stored_text, f"raw PII in stored chunks: {value}"
    assert not re.search(r"\b9\d\d-\d\d-\d{4}\b", log + stored_text)
