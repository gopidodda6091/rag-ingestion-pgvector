"""Generate the synthetic Pecan Health Plans corpus and the retrieval eval set.

Everything here is fictional. Planted PII is deliberately fake and recognisable:
SSN-like numbers use the never-issued 900-999 area range, phone numbers use the
555-01xx fiction range, e-mails use example.com and names are invented.

Outputs: docs/*.md, docs/*.html, docs/*.pdf (needs reportlab; skipped otherwise)
and eval/questions.jsonl with gold chunk ids.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

Section = tuple[str, str]
CASE_A = (
    "Example case: member Jordan Sampleton (SSN 900-12-3456, phone (555) 010-4477, "
    "jordan.sampleton@example.com, member ID PHP-00012345) asked for a review after a denial."
)
CASE_B = (
    "Example case: Ms. Casey Placeholder, DOB 02/14/1985, SSN 912-34-5678, called 555-010-9921 "
    "to report a billing error on her account."
)

DOCS: dict[str, tuple[str, str, list[Section]]] = {
    "member-handbook": (
        "md",
        "Pecan Health Plans Member Handbook",
        [
            (
                "Welcome",
                "This handbook explains how your Pecan Health Plans coverage works. Keep "
                "your member ID card with you and show it at every visit.",
            ),
            (
                "Eligibility",
                "Employees who work at least 30 hours per week are eligible. Dependents "
                "are covered until age 26. Coverage begins on the first day of the month "
                "after hire.",
            ),
            (
                "Premiums and Billing",
                "Premiums are due on the first day of each month. A grace "
                "period of 31 days applies before coverage is suspended for non-payment.",
            ),
            (
                "Claims",
                "In-network providers file claims for you. For out-of-network care, submit "
                "the claim form within 180 days of the date of service.",
            ),
            (
                "Appeals",
                "You may appeal a denied claim within 60 days of the denial letter. A "
                "decision is mailed within 30 days. " + CASE_A,
            ),
            ("Contact Us", "Member Services is open Monday to Friday, 8 a.m. to 6 p.m. Central."),
        ],
    ),
    "pharmacy-benefits": (
        "md",
        "Pharmacy Benefits",
        [
            (
                "Formulary Tiers",
                "Tier 1 generics cost $10, tier 2 preferred brands cost $35 and "
                "tier 3 non-preferred brands cost $70 for a 30-day supply.",
            ),
            (
                "Mail Order",
                "A 90-day supply of maintenance medication through mail order costs two "
                "copays instead of three.",
            ),
            (
                "Specialty Drugs",
                "Specialty drugs must be filled at the Pecan Specialty Pharmacy and "
                "usually require prior authorization.",
            ),
        ],
    ),
    "telehealth-policy": (
        "md",
        "Telehealth Policy",
        [
            (
                "Covered Services",
                "Virtual visits for urgent care, behavioral health and dermatology "
                "photo review are covered.",
            ),
            ("Cost Sharing", "Telehealth urgent care visits have a $0 copay on all plans."),
            (
                "Limitations",
                "Telehealth cannot be used for controlled substance prescriptions or "
                "for emergencies. Call 911 for emergencies.",
            ),
        ],
    ),
    "faq-billing": (
        "md",
        "Billing FAQ",
        [
            (
                "Why did my premium change?",
                "Premiums change at renewal each January or when you " "add or remove a dependent.",
            ),
            (
                "How do I set up autopay?",
                "Sign in to the member portal, open Billing and choose "
                "Autopay. Bank drafts run on the first business day of the month.",
            ),
            (
                "Reporting a billing error",
                "Call Member Services or send a secure message. " + CASE_B,
            ),
        ],
    ),
    "privacy-notice": (
        "md",
        "Notice of Privacy Practices",
        [
            (
                "How We Use Information",
                "We use health information for treatment, payment and "
                "health care operations, and never sell member data.",
            ),
            (
                "Your Rights",
                "You may request a copy of your records, ask for corrections and "
                "request an accounting of disclosures once every 12 months at no charge.",
            ),
            (
                "Complaints",
                "Privacy complaints can be sent to the Privacy Officer at "
                "privacy.office@example.com or by phone at (555) 010-2200.",
            ),
        ],
    ),
    "prior-authorization-policy": (
        "html",
        "Prior Authorization Policy",
        [
            (
                "Services Requiring Prior Authorization",
                "MRI and CT scans, inpatient admissions, "
                "home health care and specialty drugs require prior authorization.",
            ),
            (
                "Turnaround Times",
                "Standard requests are decided within 15 calendar days. Urgent "
                "requests are decided within 72 hours.",
            ),
            (
                "Retroactive Requests",
                "Retroactive authorization is only considered for emergency "
                "admissions reported within 2 business days.",
            ),
        ],
    ),
    "out-of-network-policy": (
        "html",
        "Out-of-Network Coverage",
        [
            (
                "Coverage Levels",
                "Out-of-network services are paid at 60 percent of the allowed "
                "amount after the out-of-network deductible.",
            ),
            (
                "Balance Billing",
                "Out-of-network providers may bill you for the difference between "
                "their charge and the allowed amount, except for emergency care.",
            ),
            (
                "Emergency Care",
                "Emergency room care is always covered at the in-network level, "
                "even at an out-of-network hospital.",
            ),
        ],
    ),
    "faq-enrollment": (
        "html",
        "Enrollment FAQ",
        [
            (
                "When is open enrollment?",
                "Open enrollment runs from November 1 to November 30 "
                "each year for coverage starting January 1.",
            ),
            (
                "Qualifying life events",
                "Marriage, birth, adoption or loss of other coverage let "
                "you change plans within 30 days of the event.",
            ),
            (
                "Adding a newborn",
                "Add a newborn within 30 days of birth; coverage is retroactive "
                "to the date of birth.",
            ),
        ],
    ),
    "grievance-process": (
        "html",
        "Grievance Process",
        [
            (
                "Filing a Grievance",
                "A grievance is a complaint about service quality. File it by "
                "phone, mail or the member portal within 180 days.",
            ),
            (
                "Resolution",
                "Grievances are resolved within 30 days and you receive a written "
                "response. Dr. Riley Fictional, Medical Director, reviews clinical grievances.",
            ),
            (
                "External Review",
                "If you disagree with a final appeal decision you can request an "
                "independent external review within 4 months.",
            ),
        ],
    ),
    "dental-rider": (
        "pdf",
        "Dental Rider",
        [
            (
                "Preventive Care",
                "Two cleanings and two exams per calendar year are covered at "
                "100 percent with no waiting period.",
            ),
            (
                "Major Services",
                "Crowns and bridges are covered at 50 percent after a 12-month "
                "waiting period. The annual maximum is $1500.",
            ),
        ],
    ),
    "vision-rider": (
        "pdf",
        "Vision Rider",
        [
            ("Eye Exams", "One routine eye exam every 12 months with a $15 copay."),
            (
                "Eyewear Allowance",
                "Frames are covered up to $150 every 24 months. Contact lenses "
                "may be chosen instead of glasses.",
            ),
        ],
    ),
}

QUESTIONS: list[tuple[str, str, str]] = [
    ("How long do I have to appeal a denied claim?", "member-handbook", "Appeals"),
    (
        "What is the grace period for late premium payments?",
        "member-handbook",
        "Premiums and Billing",
    ),
    ("Until what age are dependents covered?", "member-handbook", "Eligibility"),
    ("Deadline to submit an out-of-network claim form", "member-handbook", "Claims"),
    ("How much does a tier 2 preferred brand drug cost?", "pharmacy-benefits", "Formulary Tiers"),
    ("Is a 90-day mail order supply cheaper?", "pharmacy-benefits", "Mail Order"),
    ("Where do I fill specialty drugs?", "pharmacy-benefits", "Specialty Drugs"),
    ("What is the copay for a telehealth urgent care visit?", "telehealth-policy", "Cost Sharing"),
    ("Can I get controlled substances through telehealth?", "telehealth-policy", "Limitations"),
    ("Why did my monthly premium go up?", "faq-billing", "Why did my premium change?"),
    ("How do I turn on autopay bank drafts?", "faq-billing", "How do I set up autopay?"),
    ("Do you sell member data?", "privacy-notice", "How We Use Information"),
    (
        "Which services need prior authorization?",
        "prior-authorization-policy",
        "Services Requiring Prior Authorization",
    ),
    (
        "How fast are urgent prior authorization requests decided?",
        "prior-authorization-policy",
        "Turnaround Times",
    ),
    (
        "What percent is paid for out-of-network services?",
        "out-of-network-policy",
        "Coverage Levels",
    ),
    (
        "Is the emergency room covered at an out-of-network hospital?",
        "out-of-network-policy",
        "Emergency Care",
    ),
    ("When does open enrollment run?", "faq-enrollment", "When is open enrollment?"),
    ("How soon must I add a newborn baby?", "faq-enrollment", "Adding a newborn"),
    ("How do I request an independent external review?", "grievance-process", "External Review"),
    ("What is the dental annual maximum for crowns?", "dental-rider", "Major Services"),
    ("How often is the eyewear frame allowance available?", "vision-rider", "Eyewear Allowance"),
]


def slugify(text: str) -> str:
    """Must match rag_ingest.chunk.slugify (duplicated so the script has no package import)."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render_md(title: str, sections: list[Section]) -> str:
    body = "\n\n".join(f"## {h}\n\n{t}" for h, t in sections)
    return f"# {title}\n\n{body}\n"


def render_html(title: str, sections: list[Section]) -> str:
    body = "\n".join(f"<h2>{html.escape(h)}</h2>\n<p>{html.escape(t)}</p>" for h, t in sections)
    return (
        f"<!doctype html>\n<html><head><title>{html.escape(title)}</title></head>\n"
        f"<body>\n<h1>{html.escape(title)}</h1>\n{body}\n</body></html>\n"
    )


def render_pdf(path: Path, title: str, sections: list[Section]) -> bool:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate
    except ImportError:
        print(f"reportlab not installed; skipping {path.name}")
        return False
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"])]
    for heading, text in sections:
        story += [Paragraph(heading, styles["Heading2"]), Paragraph(text, styles["BodyText"])]
    SimpleDocTemplate(str(path), pagesize=letter, invariant=1).build(story)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--eval", type=Path, default=Path("eval/questions.jsonl"))
    args = parser.parse_args()
    args.docs.mkdir(parents=True, exist_ok=True)
    written = []
    for doc_id, (fmt, title, sections) in DOCS.items():
        path = args.docs / f"{doc_id}.{fmt}"
        if fmt == "md":
            path.write_text(render_md(title, sections), encoding="utf-8")
        elif fmt == "html":
            path.write_text(render_html(title, sections), encoding="utf-8")
        elif not render_pdf(path, title, sections):
            continue
        written.append(doc_id)
    args.eval.parent.mkdir(parents=True, exist_ok=True)
    with args.eval.open("w", encoding="utf-8") as fh:
        for question, doc_id, heading in QUESTIONS:
            if doc_id in written:
                gold = [f"{doc_id}::{slugify(heading)}::0"]
                fh.write(json.dumps({"question": question, "gold_chunk_ids": gold}) + "\n")
    print(f"wrote {len(written)} documents to {args.docs} and eval set to {args.eval}")


if __name__ == "__main__":
    main()
