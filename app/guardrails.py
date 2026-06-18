"""Input/output guardrails and data masking for the health chatbot."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

MAX_INPUT_CHARS = 4000

DISCLAIMER = (
    "This AI-generated information is for educational purposes only and does "
    "NOT replace professional medical advice. Please consult a qualified "
    "healthcare provider."
)

EMERGENCY_NOTICE = (
    "> **Urgent safety note:** If this may be an emergency, call emergency "
    "services now or go to the nearest emergency department."
)

PROMPT_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bignore (all )?(previous|prior|above) instructions\b",
        r"\breveal (the )?(system|developer) prompt\b",
        r"\bshow (me )?(your )?(hidden|internal) instructions\b",
        r"\bprint (the )?(system|developer) message\b",
        r"\bdisregard (the )?(rules|guardrails|instructions)\b",
        r"\bact as (an )?(unfiltered|unrestricted|developer mode)\b",
    ]
]

EMERGENCY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(chest pain|crushing chest|heart attack)\b",
        r"\b(shortness of breath|difficulty breathing|can't breathe|cannot breathe)\b",
        r"\b(stroke|face droop|facial droop|one-sided weakness|slurred speech)\b",
        r"\b(seizure|unconscious|passed out|fainted)\b",
        r"\b(overdose|poisoning)\b",
        r"\b(suicidal|kill myself|self[- ]harm)\b",
        r"\b(severe bleeding|bleeding heavily)\b",
    ]
]

SENSITIVE_PATTERNS = [
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[EMAIL]"),
    (
        re.compile(
            r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
        ),
        "[PHONE]",
    ),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (
        re.compile(
            r"\b(?:dob|date of birth|birth date)\s*[:\-]?\s*"
            r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Z][a-z]+ \d{1,2},? \d{4})\b",
            re.IGNORECASE,
        ),
        "DOB: [DATE]",
    ),
    (re.compile(r"\b(?:mrn|medical record number)\s*[:#-]?\s*[A-Z0-9-]{5,}\b", re.IGNORECASE), "MRN: [ID]"),
    (
        re.compile(
            r"\b(?:\d[ -]*?){13,16}\b"
        ),
        "[CARD_OR_LONG_ID]",
    ),
    (
        re.compile(
            r"\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,4}\s+"
            r"(?:street|st|road|rd|avenue|ave|drive|dr|lane|ln|blvd|boulevard)\b",
            re.IGNORECASE,
        ),
        "[ADDRESS]",
    ),
]

MASK_MARKERS = {
    "[EMAIL]": "email",
    "[PHONE]": "phone",
    "[SSN]": "ssn",
    "DOB: [DATE]": "dob",
    "MRN: [ID]": "mrn",
    "[CARD_OR_LONG_ID]": "card_or_long_id",
    "[ADDRESS]": "address",
}


@dataclass
class GuardrailResult:
    """Result of screening a user message before the workflow runs."""

    allowed: bool
    sanitized_text: str
    masked_text: str
    blocked_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    emergency_detected: bool = False


def mask_sensitive_data(text: str) -> str:
    """Mask common personally identifiable data from free text."""
    masked = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked


def _masked_categories(masked_text: str) -> list[str]:
    """Return the categories that were masked without exposing the raw values."""
    return [
        category
        for marker, category in MASK_MARKERS.items()
        if marker in masked_text
    ]


def screen_input(user_input: str) -> GuardrailResult:
    """Validate, normalize, and mask a user message."""
    raw_text = (user_input or "").strip()
    if not raw_text:
        logger.warning("[Guardrails][Input] blocked reason=empty_message")
        return GuardrailResult(
            allowed=False,
            sanitized_text="",
            masked_text="",
            blocked_reason="Please enter a message.",
        )

    if len(raw_text) > MAX_INPUT_CHARS:
        logger.warning(
            "[Guardrails][Input] blocked reason=too_long chars={} max_chars={}",
            len(raw_text),
            MAX_INPUT_CHARS,
        )
        return GuardrailResult(
            allowed=False,
            sanitized_text="",
            masked_text="",
            blocked_reason=(
                f"Please keep your message under {MAX_INPUT_CHARS} characters."
            ),
        )

    if any(pattern.search(raw_text) for pattern in PROMPT_INJECTION_PATTERNS):
        logger.warning(
            "[Guardrails][Input] blocked reason=prompt_injection chars={}",
            len(raw_text),
        )
        return GuardrailResult(
            allowed=False,
            sanitized_text="",
            masked_text="",
            blocked_reason=(
                "I can help with health questions, symptoms, history, and next "
                "steps, but I cannot reveal or change system instructions."
            ),
        )

    masked_text = mask_sensitive_data(raw_text)
    emergency_detected = any(pattern.search(raw_text) for pattern in EMERGENCY_PATTERNS)
    masked_categories = _masked_categories(masked_text)
    warnings: list[str] = []
    if masked_text != raw_text:
        warnings.append("Sensitive identifiers were masked before processing.")
    if emergency_detected:
        warnings.append("Emergency red-flag symptoms were detected.")

    logger.info(
        (
            "[Guardrails][Input] allowed chars={} masked={} "
            "masked_categories={} emergency_detected={} warnings={}"
        ),
        len(raw_text),
        masked_text != raw_text,
        masked_categories,
        emergency_detected,
        warnings,
    )

    return GuardrailResult(
        allowed=True,
        sanitized_text=masked_text,
        masked_text=masked_text,
        warnings=warnings,
        emergency_detected=emergency_detected,
    )


def apply_output_guardrails(response: str, emergency_detected: bool = False) -> str:
    """Mask sensitive data and enforce health-safety messaging in final output."""
    original = response or ""
    guarded = mask_sensitive_data(original)
    output_masked = guarded != original
    masked_categories = _masked_categories(guarded)

    added_emergency_notice = False
    if emergency_detected and EMERGENCY_NOTICE not in guarded:
        guarded = f"{EMERGENCY_NOTICE}\n\n{guarded}".strip()
        added_emergency_notice = True

    added_disclaimer = False
    if DISCLAIMER not in guarded:
        guarded = f"{guarded}\n\n### Disclaimer\n{DISCLAIMER}".strip()
        added_disclaimer = True

    before_softening = guarded
    guarded = re.sub(
        r"\byou definitely have\b",
        "you may have",
        guarded,
        flags=re.IGNORECASE,
    )
    guarded = re.sub(
        r"\bthis is definitely\b",
        "this may be",
        guarded,
        flags=re.IGNORECASE,
    )

    logger.info(
        (
            "[Guardrails][Output] applied chars_in={} chars_out={} "
            "masked={} masked_categories={} emergency_notice_added={} "
            "disclaimer_added={} definitive_language_softened={}"
        ),
        len(original),
        len(guarded),
        output_masked,
        masked_categories,
        added_emergency_notice,
        added_disclaimer,
        guarded != before_softening,
    )
    return guarded
