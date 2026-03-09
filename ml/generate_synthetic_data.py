"""
Synthetic PII/PHI Dataset Generator for Obscura NER Benchmarking

Generates labeled NER samples using Faker with BIO tagging across
enterprise (names, orgs, SSNs, phones, emails), clinical (MRN,
patient names, DOB), financial (credit cards), and identity/tech
(passport, IPv4) entity types.

Output format: JSONL compatible with HuggingFace datasets, where each
record contains `tokens` (word list) and `ner_tags` (BIO label list).

Usage:
    python generate_synthetic_data.py --num-samples 500 --output data/synthetic.jsonl

HIPAA/GDPR: All data is purely synthetic. No real PII/PHI is used.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from faker import Faker

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ml.nemotron_data import (
    CANONICAL_LABELS,
    LABEL_TO_ID,
    build_nemotron_jsonl,
    load_nemotron_samples,
    resolve_nemotron_snapshot,
)

fake = Faker()
Faker.seed(42)
random.seed(42)

# BIO label vocabulary — shared with the Nemotron adapter and model pipeline.
LABEL_LIST: list[str] = CANONICAL_LABELS

# Nationalities for MISC entity generation. These are the most reliably
# detected MISC entities from CoNLL-2003 training data.
NATIONALITIES: list[str] = [
    "American",
    "British",
    "Canadian",
    "French",
    "German",
    "Japanese",
    "Chinese",
    "Australian",
    "Indian",
    "Italian",
    "Mexican",
    "Brazilian",
    "Korean",
    "Spanish",
    "Russian",
]


def _ssn(dashless: bool = False) -> str:
    """Generate a realistic SSN with IRS-valid components.

    Per SSA/IRS rules (Publication 4557), invalid SSNs have:
    - Area (first 3 digits): 000, 666, or 900-999
    - Group (middle 2 digits): 00
    - Serial (last 4 digits): 0000

    Args:
        dashless: If True, return 9 digits without dashes (e.g. "123456789").
                  If False, return dashed format (e.g. "123-45-6789").
    """
    while True:
        area = random.randint(1, 899)
        if area == 666:
            continue
        break
    group = random.randint(1, 99)
    serial = random.randint(1, 9999)

    if dashless:
        return f"{area:03d}{group:02d}{serial:04d}"
    return f"{area:03d}-{group:02d}-{serial:04d}"


def _mrn() -> str:
    """Generate a medical record number (MRN-XXXXXXX)."""
    return f"MRN-{random.randint(1000000, 9999999)}"


def _phone() -> str:
    """Generate a US phone number in several common display formats."""
    a = random.randint(200, 999)
    b = random.randint(200, 999)
    c = random.randint(1000, 9999)
    formats = [
        f"({a}) {b}-{c}",
        f"{a}-{b}-{c}",
        f"{a} {b} {c}",
        f"1-{a}-{b}-{c}",
    ]
    return random.choice(formats)


def _local_grouped_phone() -> str:
    """Generate a grouped local phone number with a leading zero."""
    templates = [
        (4, 3, 3),
        (4, 3, 4),
        (3, 3, 3),
    ]
    sizes = random.choice(templates)
    first = "0" + "".join(str(random.randint(0, 9)) for _ in range(sizes[0] - 1))
    rest = [
        "".join(str(random.randint(0, 9)) for _ in range(size))
        for size in sizes[1:]
    ]
    separator = random.choice([" ", "-", "."])
    return separator.join([first, *rest])


def _international_phone() -> str:
    """Generate an international phone number with a country code prefix."""
    country_codes = ["+44", "+49", "+61", "+81", "+91", "+995", "00420"]
    group_templates = [
        (3, 3, 2, 2),
        (2, 4, 4),
        (3, 3, 4),
        (2, 3, 3, 3),
    ]
    prefix = random.choice(country_codes)
    groups = [
        str(random.randint(10 ** (size - 1), (10**size) - 1))
        for size in random.choice(group_templates)
    ]
    separator = random.choice([" ", "-", " "])
    return prefix + separator + separator.join(groups)


def _contact_phone() -> str:
    """Generate a phone number across US, international, and local styles."""
    return random.choice([_phone, _international_phone, _local_grouped_phone])()


def _credit_card(dashed: bool = True) -> str:
    """Generate a 16-digit credit card number."""
    groups = [random.randint(1000, 9999) for _ in range(4)]
    if dashed:
        return "-".join(str(g) for g in groups)
    return "".join(str(g) for g in groups)


def _ipv4() -> str:
    """Generate a random IPv4 address (non-reserved)."""
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _passport() -> str:
    """Generate a US passport number (1 uppercase letter + 8 digits)."""
    letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = random.randint(10000000, 99999999)
    return f"{letter}{digits}"


def _tokenize_and_label(
    text: str, entities: list[tuple[str, str]]
) -> tuple[list[str], list[str]]:
    """
    Given a text string and a list of (entity_text, entity_type) pairs,
    produce aligned token and BIO-tag lists.

    Uses whitespace tokenization for simplicity. Entity boundaries are
    detected by exact string match against the template-inserted values.

    Args:
        text: The full sentence with entities already substituted in.
        entities: List of (surface_form, BIO_prefix) tuples,
                  e.g. ("John Smith", "PER").

    Returns:
        (tokens, ner_tags) where both lists have the same length.
    """
    # Build a map of character spans to entity types.
    char_labels: list[str] = ["O"] * len(text)

    # Sort entities longest-first so longer matches take priority.
    # Prevents a short entity (e.g. "American") from consuming characters
    # that belong to a longer entity (e.g. "American Airlines").
    sorted_entities = sorted(entities, key=lambda e: len(e[0]), reverse=True)

    for entity_text, entity_type in sorted_entities:
        # Find all non-overlapping occurrences.
        start = 0
        while True:
            idx = text.find(entity_text, start)
            if idx == -1:
                break
            # Only label if not already labeled (avoid double-labeling).
            if all(c == "O" for c in char_labels[idx : idx + len(entity_text)]):
                for i in range(idx, idx + len(entity_text)):
                    char_labels[i] = entity_type
            start = idx + len(entity_text)

    # Whitespace-tokenize and derive BIO tags.
    tokens: list[str] = []
    ner_tags: list[str] = []

    for match in re.finditer(r"\S+", text):
        token = match.group()
        tok_start = match.start()
        tok_end = match.end()

        # Determine the majority label for this token's character span.
        span_labels = char_labels[tok_start:tok_end]
        non_o = [lbl for lbl in span_labels if lbl != "O"]

        if not non_o:
            tokens.append(token)
            ner_tags.append("O")
        else:
            entity_type = non_o[0]
            # Check if previous token was the same entity type (I- vs B-).
            if ner_tags and ner_tags[-1] in (
                f"B-{entity_type}",
                f"I-{entity_type}",
            ):
                tag = f"I-{entity_type}"
            else:
                tag = f"B-{entity_type}"
            tokens.append(token)
            ner_tags.append(tag)

    return tokens, ner_tags


# --- Template generators ---


def _enterprise_sample() -> tuple[str, list[tuple[str, str]]]:
    """Generate one enterprise-domain sample with PII entities.

    Includes both dashed and dashless SSN templates. Dashless templates
    use SSN-context trigger words ("social security number", "SSN",
    "tax ID") so the context-aware regex detector can identify them.
    Some templates include MISC entities (nationalities) for coverage.
    """
    person = fake.name()
    org = fake.company()
    phone = _contact_phone()
    email = fake.email()
    location = fake.city()
    misc = random.choice(NATIONALITIES)

    # Templates with dashed SSNs. `misc` is captured from enclosing scope.
    dashed_templates = [
        lambda p, o, s, ph, e, loc: (
            f"Please summarize the case for {p} (SSN: {s}), "
            f"who works at {o} and can be reached at {ph} or {e}.",
            [(p, "PER"), (o, "ORG"), (s, "SSN"), (ph, "PHONE"), (e, "EMAIL")],
        ),
        lambda p, o, s, ph, e, loc: (
            f"The meeting between {p} from {o} was held in {loc}. "
            f"Contact: {ph}, email {e}.",
            [(p, "PER"), (o, "ORG"), (ph, "PHONE"), (e, "EMAIL"), (loc, "LOC")],
        ),
        lambda p, o, s, ph, e, loc: (
            f"{p} submitted an application to {o}. " f"SSN on file: {s}. Phone: {ph}.",
            [(p, "PER"), (o, "ORG"), (s, "SSN"), (ph, "PHONE")],
        ),
        lambda p, o, s, ph, e, loc: (
            f"Employee {p} at {o} reported a security incident. "
            f"Verified identity via SSN {s} and callback to {ph}.",
            [(p, "PER"), (o, "ORG"), (s, "SSN"), (ph, "PHONE")],
        ),
        lambda p, o, s, ph, e, loc: (
            f"Send the contract to {p} at {e}. "
            f"Their office at {o} is located in {loc}.",
            [(p, "PER"), (e, "EMAIL"), (o, "ORG"), (loc, "LOC")],
        ),
        # MISC templates (nationality captured via closure)
        lambda p, o, s, ph, e, loc: (
            f"The {misc} employee {p} at {o} has SSN {s} on file. Phone: {ph}.",
            [(misc, "MISC"), (p, "PER"), (o, "ORG"), (s, "SSN"), (ph, "PHONE")],
        ),
        lambda p, o, s, ph, e, loc: (
            f"{p}, a {misc} national, is employed by {o} in {loc}. "
            f"Contact: {ph} or {e}.",
            [
                (p, "PER"),
                (misc, "MISC"),
                (o, "ORG"),
                (loc, "LOC"),
                (ph, "PHONE"),
                (e, "EMAIL"),
            ],
        ),
        lambda p, o, s, ph, e, loc: (
            f"Background check for {misc} citizen {p} at {o}. "
            f"SSN: {s}, callback {ph}.",
            [(misc, "MISC"), (p, "PER"), (o, "ORG"), (s, "SSN"), (ph, "PHONE")],
        ),
    ]

    # Templates with dashless SSNs — include trigger words for context.
    dashless_templates = [
        lambda p, o, sd, ph, e, loc: (
            f"Employee {p} at {o} has social security number {sd} on file.",
            [(p, "PER"), (o, "ORG"), (sd, "SSN")],
        ),
        lambda p, o, sd, ph, e, loc: (
            f"Update the SSN for {p} to {sd}. Current employer: {o}.",
            [(p, "PER"), (sd, "SSN"), (o, "ORG")],
        ),
        lambda p, o, sd, ph, e, loc: (
            f"Tax ID verification: {p}, taxpayer identification "
            f"number {sd}, employed at {o}.",
            [(p, "PER"), (sd, "SSN"), (o, "ORG")],
        ),
        # MISC dashless template
        lambda p, o, sd, ph, e, loc: (
            f"The {misc} taxpayer {p} at {o} has SSN {sd} per tax records.",
            [(misc, "MISC"), (p, "PER"), (o, "ORG"), (sd, "SSN")],
        ),
    ]

    # 40% chance of dashless SSN template.
    if random.random() < 0.4:
        ssn = _ssn(dashless=True)
        template_fn = random.choice(dashless_templates)
        text, entities = template_fn(person, org, ssn, phone, email, location)
    else:
        ssn = _ssn(dashless=False)
        template_fn = random.choice(dashed_templates)
        text, entities = template_fn(person, org, ssn, phone, email, location)
    return text, entities


def _clinical_sample() -> tuple[str, list[tuple[str, str]]]:
    """Generate one clinical-domain sample with PHI entities.

    Some templates include MISC entities (nationalities) for coverage.
    """
    conditions = [
        "Type 2 Diabetes",
        "Hypertension",
        "Acute Bronchitis",
        "Major Depressive Disorder",
        "Chronic Kidney Disease",
        "Atrial Fibrillation",
        "COPD",
        "Rheumatoid Arthritis",
    ]
    medications = [
        "Metformin 500mg",
        "Lisinopril 10mg",
        "Amoxicillin 250mg",
        "Sertraline 50mg",
        "Atorvastatin 20mg",
        "Warfarin 5mg",
        "Albuterol inhaler",
        "Prednisone 10mg",
    ]

    person = fake.name()
    mrn = _mrn()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%m/%d/%Y")
    condition = random.choice(conditions)
    medication = random.choice(medications)
    misc = random.choice(NATIONALITIES)

    templates = [
        lambda p, m, d, cond, med: (
            f"Patient {p}, MRN {m}, DOB {d}, presents with {cond}. "
            f"Current medication: {med}.",
            [(p, "PER"), (m, "MRN"), (d, "DOB")],
        ),
        lambda p, m, d, cond, med: (
            f"Dr. {p} reviewed MRN {m} and diagnosed {cond}. "
            f"Prescribed {med}. Follow-up in 2 weeks.",
            [(p, "PER"), (m, "MRN")],
        ),
        lambda p, m, d, cond, med: (
            f"Discharge summary for {p} (MRN: {m}, DOB: {d}). "
            f"Admitted for {cond}, treated with {med}, stable at discharge.",
            [(p, "PER"), (m, "MRN"), (d, "DOB")],
        ),
        lambda p, m, d, cond, med: (
            f"Lab results for patient {p}, MRN {m}. "
            f"Diagnosis: {cond}. Adjusting {med} dosage.",
            [(p, "PER"), (m, "MRN")],
        ),
        # MISC templates (nationality captured via closure)
        lambda p, m, d, cond, med: (
            f"The {misc} patient {p}, MRN {m}, DOB {d}, "
            f"presented with {cond}. Started {med}.",
            [(misc, "MISC"), (p, "PER"), (m, "MRN"), (d, "DOB")],
        ),
        lambda p, m, d, cond, med: (
            f"Patient {p} ({misc} national), MRN {m}. "
            f"Diagnosed with {cond}, prescribed {med}.",
            [(p, "PER"), (misc, "MISC"), (m, "MRN")],
        ),
    ]

    template_fn = random.choice(templates)
    text, entities = template_fn(person, mrn, dob, condition, medication)
    return text, entities


def _financial_sample() -> tuple[str, list[tuple[str, str]]]:
    """Generate financial-domain samples with credit card numbers.

    Dashed CC templates use the distinctive format (no trigger needed).
    Undashed CC templates include trigger words (visa, credit, card)
    so the context-aware regex detector can identify them.
    """
    person = fake.name()
    org = fake.company()
    phone = _contact_phone()

    dashed_templates = [
        lambda p, o, cc, ph: (
            f"Process refund for {p} at {o}. Card: {cc}. Contact: {ph}.",
            [(p, "PER"), (o, "ORG"), (cc, "CREDIT_CARD"), (ph, "PHONE")],
        ),
        lambda p, o, cc, ph: (
            f"Charge {cc} for the subscription renewal. Account holder: {p} at {o}.",
            [(cc, "CREDIT_CARD"), (p, "PER"), (o, "ORG")],
        ),
        lambda p, o, cc, ph: (
            f"{p} authorized payment via {cc} for services from {o}.",
            [(p, "PER"), (cc, "CREDIT_CARD"), (o, "ORG")],
        ),
    ]

    undashed_templates = [
        lambda p, o, cc, ph: (
            f"Visa card {cc} on file for {p} at {o}.",
            [(cc, "CREDIT_CARD"), (p, "PER"), (o, "ORG")],
        ),
        lambda p, o, cc, ph: (
            f"Credit card number {cc} belongs to {p}. Employer: {o}.",
            [(cc, "CREDIT_CARD"), (p, "PER"), (o, "ORG")],
        ),
        lambda p, o, cc, ph: (
            f"Debit card {cc} linked to {p} at {o}. Call {ph} to verify.",
            [(cc, "CREDIT_CARD"), (p, "PER"), (o, "ORG"), (ph, "PHONE")],
        ),
    ]

    if random.random() < 0.4:
        cc = _credit_card(dashed=False)
        template_fn = random.choice(undashed_templates)
    else:
        cc = _credit_card(dashed=True)
        template_fn = random.choice(dashed_templates)

    text, entities = template_fn(person, org, cc, phone)
    return text, entities


def _identity_tech_sample() -> tuple[str, list[tuple[str, str]]]:
    """Generate samples with US passport numbers and/or IPv4 addresses."""
    person = fake.name()
    org = fake.company()
    passport = _passport()
    ipv4 = _ipv4()
    email = fake.email()

    templates = [
        lambda p, o, pp, ip, e: (
            f"Passport {pp} verified for {p} traveling on behalf of {o}.",
            [(pp, "PASSPORT"), (p, "PER"), (o, "ORG")],
        ),
        lambda p, o, pp, ip, e: (
            f"{p} from {o} presented passport number {pp} at the border.",
            [(p, "PER"), (o, "ORG"), (pp, "PASSPORT")],
        ),
        lambda p, o, pp, ip, e: (
            f"Employee {p} at {o} connected from {ip}. Email: {e}.",
            [(p, "PER"), (o, "ORG"), (ip, "IPV4"), (e, "EMAIL")],
        ),
        lambda p, o, pp, ip, e: (
            f"Login from {ip} by {p} flagged. Notify {o} security at {e}.",
            [(ip, "IPV4"), (p, "PER"), (o, "ORG"), (e, "EMAIL")],
        ),
        lambda p, o, pp, ip, e: (
            f"{p} at {o}: passport {pp}, last login from {ip}.",
            [(p, "PER"), (o, "ORG"), (pp, "PASSPORT"), (ip, "IPV4")],
        ),
    ]

    template_fn = random.choice(templates)
    text, entities = template_fn(person, org, passport, ipv4, email)
    return text, entities


def _negative_sample() -> tuple[str, list[tuple[str, str]]]:
    """Generate a sample with 9-digit numbers in NON-SSN contexts.

    These are negative examples: 9-digit strings that appear near
    phone/order/serial/tracking context words and should NOT be
    classified as SSN. The numbers are IRS-structurally-valid so
    the only distinguishing signal is surrounding context.
    """
    person = fake.name()
    org = fake.company()
    nine_digits = _ssn(dashless=True)  # Valid structure, but NOT an SSN

    templates = [
        lambda p, o, nd: (
            f"Call {p} at phone number {nd} regarding the shipment from {o}.",
            [(p, "PER"), (o, "ORG")],
        ),
        lambda p, o, nd: (
            f"Tracking number {nd} for the order placed by {p} at {o}.",
            [(p, "PER"), (o, "ORG")],
        ),
        lambda p, o, nd: (
            f"Reference account number {nd} for {p} at {o} on the invoice.",
            [(p, "PER"), (o, "ORG")],
        ),
        lambda p, o, nd: (
            f"Serial number {nd} registered to {p} at {o}.",
            [(p, "PER"), (o, "ORG")],
        ),
    ]

    template_fn = random.choice(templates)
    text, entities = template_fn(person, org, nine_digits)
    return text, entities


def _international_contact_sample() -> tuple[str, list[tuple[str, str]]]:
    """Generate positive samples with international phone formats."""
    person = fake.name()
    org = fake.company()
    phone = random.choice([_international_phone, _local_grouped_phone, _phone])()
    email = fake.email()
    location = fake.city()

    templates = [
        lambda p, o, ph, e, loc: (
            f"Please feel free to contact {p} at {ph} or via email at {e}. {p} works with {o} in {loc}.",
            [(p, "PER"), (ph, "PHONE"), (e, "EMAIL"), (o, "ORG"), (loc, "LOC")],
        ),
        lambda p, o, ph, e, loc: (
            f"International support callback for {p}: {ph}. Escalation owner is {o}; confirmation sent to {e}.",
            [(p, "PER"), (ph, "PHONE"), (o, "ORG"), (e, "EMAIL")],
        ),
        lambda p, o, ph, e, loc: (
            f"Reach the regional office in {loc} on {ph}. Account manager {p} from {o} will respond at {e}.",
            [(loc, "LOC"), (ph, "PHONE"), (p, "PER"), (o, "ORG"), (e, "EMAIL")],
        ),
    ]

    template_fn = random.choice(templates)
    return template_fn(person, org, phone, email, location)


def _business_hard_negative_sample() -> tuple[str, list[tuple[str, str]]]:
    """Generate business-style text that should not be redacted as PII."""
    teams = [
        "Government Support Team",
        "Customer Success Team",
        "Regional Operations Team",
        "Platform Enablement Team",
        "Compliance Review Team",
        "Service Delivery Team",
    ]
    departments = [
        "Public Services",
        "Operations",
        "Engineering",
        "Customer Support",
        "Community Outreach",
    ]
    projects = [
        "Q3 readiness review",
        "service rollout plan",
        "regional support handbook",
        "customer care transition",
        "incident response drill",
    ]
    company = fake.company()
    city = fake.city()
    team = random.choice(teams)
    department = random.choice(departments)
    project = random.choice(projects)
    generic_date = fake.date_between(start_date="-30d", end_date="+30d").strftime("%m/%d/%Y")
    generic_time = fake.time(pattern="%H:%M")
    url = f"https://{fake.domain_name()}/updates/{random.randint(100, 999)}"

    templates = [
        lambda: (
            f"Best regards,\n\n{team}.\n{department} Division\n{company}",
            [],
        ),
        lambda: (
            f"The {team} will review the {project} on {generic_date} at {generic_time}. Notes will be published at {url}.",
            [],
        ),
        lambda: (
            f"Please route this request to the {department} desk in {city}. The {team} is handling the next update cycle.",
            [],
        ),
        lambda: (
            f"Meeting summary: {team} coordinated with {department} on the {project}. No customer data was included.",
            [],
        ),
    ]

    return random.choice(templates)()


def _transcript_multi_org_sample() -> tuple[str, list[tuple[str, str]]]:
    """Generate transcript-style samples with one company mentioned early and another later.

    This targets the failure mode where the model anchors on the first company
    and misses a second organization mentioned later in a conversational Q&A.
    """
    person = fake.name()
    host = fake.first_name()
    primary_org = fake.company()
    secondary_org = fake.company()
    while secondary_org == primary_org:
        secondary_org = fake.company()

    city = fake.city()
    country = fake.country()
    ssn = _ssn()
    ipv4 = _ipv4()
    phone = _contact_phone()
    url = f"https://{fake.domain_name()}/episodes/{random.randint(1000, 9999)}"
    role = random.choice(
        [
            "operations director",
            "platform lead",
            "content strategy manager",
            "engineering lead",
            "distribution manager",
        ]
    )
    podcast = random.choice(
        [
            "The Future of Media",
            "Signal & Stream",
            "Digital Frontlines",
            "The Platform Brief",
            "Through the Feed",
        ]
    )

    templates = [
        lambda: (
            f"**Episode Title:** {podcast}\n\n"
            f"**Guest:** {person}\n\n"
            f"**Company:** {primary_org}\n\n"
            f"**Transcript URL:** {url}\n\n"
            f"**Host:** Today we have {person}, who works at {primary_org}. Welcome!\n\n"
            f"**{person}:** Thanks for having me. My work at {primary_org} keeps me busy.\n\n"
            f"**Host:** You previously mentioned partnerships in {city}, {country}.\n\n"
            f"**{person}:** Yes, and we also review logs like {ipv4} and identifiers such as {ssn}.\n\n"
            f"**Host:** Separate question: you also consult with {secondary_org}, correct? What is it like working there?",
            [
                (person, "PER"),
                (primary_org, "ORG"),
                (url, "MISC"),
                (city, "LOC"),
                (country, "LOC"),
                (ipv4, "IPV4"),
                (ssn, "SSN"),
                (secondary_org, "ORG"),
            ],
        ),
        lambda: (
            f"Transcript excerpt:\n"
            f"Host {host}: We are joined by {person} from {primary_org}.\n"
            f"{person}: I lead {role} at {primary_org} and work with teams in {city}.\n"
            f"Host {host}: You referenced support calls to {phone} and incident traffic from {ipv4}.\n"
            f"{person}: Correct, and some audit records still contain tokens like {ssn}.\n"
            f"Host {host}: One more thing, you work at {secondary_org} too, right? What's the culture like there?",
            [
                (host, "PER"),
                (person, "PER"),
                (primary_org, "ORG"),
                (city, "LOC"),
                (phone, "PHONE"),
                (ipv4, "IPV4"),
                (ssn, "SSN"),
                (secondary_org, "ORG"),
            ],
        ),
        lambda: (
            f"Interview Notes\n\n"
            f"Guest: {person}\n"
            f"Primary employer: {primary_org}\n"
            f"Current city: {city}, {country}\n"
            f"Reference link: {url}\n\n"
            f"Q: At {primary_org}, what does your team handle?\n"
            f"A: As {role}, I help operate the platform and review events tagged with addresses like {ipv4}.\n\n"
            f"Q: We also heard you recently interviewed with {secondary_org}. Is that accurate, and what do you think of the company?\n"
            f"A: Yes, I spent time speaking with {secondary_org} after a callback at {phone}.",
            [
                (person, "PER"),
                (primary_org, "ORG"),
                (city, "LOC"),
                (country, "LOC"),
                (url, "MISC"),
                (ipv4, "IPV4"),
                (secondary_org, "ORG"),
                (phone, "PHONE"),
            ],
        ),
    ]

    return random.choice(templates)()


def generate_synthetic_records(num_samples: int) -> tuple[list[dict], dict[str, int]]:
    """Generate the existing Faker-based synthetic dataset in memory.

    Splits: 40% enterprise, 25% clinical, 15% financial (credit cards),
    10% identity/tech (passport + IPv4), 10% negative (disambiguation).
    """
    entity_counts: dict[str, int] = {}
    num_enterprise = int(num_samples * 0.40)
    num_clinical = int(num_samples * 0.25)
    num_financial = int(num_samples * 0.15)
    num_identity = int(num_samples * 0.10)
    num_negative = (
        num_samples - num_enterprise - num_clinical - num_financial - num_identity
    )

    samples: list[dict] = []

    generators = (
        [(_enterprise_sample, num_enterprise)]
        + [(_clinical_sample, num_clinical)]
        + [(_financial_sample, num_financial)]
        + [(_identity_tech_sample, num_identity)]
        + [(_negative_sample, num_negative)]
    )

    for gen_fn, count in generators:
        for _ in range(count):
            text, entities = gen_fn()
            tokens, ner_tags = _tokenize_and_label(text, entities)
            tag_ids = [LABEL_TO_ID.get(tag, 0) for tag in ner_tags]
            samples.append(
                {
                    "tokens": tokens,
                    "ner_tags": tag_ids,
                    "ner_tag_labels": ner_tags,
                }
            )
            for _, etype in entities:
                entity_counts[etype] = entity_counts.get(etype, 0) + 1

    random.shuffle(samples)

    return samples, entity_counts


def generate_hard_negative_benchmark_records(
    num_samples: int,
) -> tuple[list[dict], dict[str, int]]:
    """Generate a benchmark slice focused on phone recall and business false positives."""
    samples: list[dict] = []
    entity_counts: dict[str, int] = {}

    num_business_negative = int(num_samples * 0.50)
    num_international_contact = int(num_samples * 0.30)
    num_regular_positive = num_samples - num_business_negative - num_international_contact

    generators = (
        [(_business_hard_negative_sample, num_business_negative)]
        + [(_international_contact_sample, num_international_contact)]
        + [(_enterprise_sample, num_regular_positive)]
    )

    for gen_fn, count in generators:
        for _ in range(count):
            text, entities = gen_fn()
            tokens, ner_tags = _tokenize_and_label(text, entities)
            tag_ids = [LABEL_TO_ID.get(tag, 0) for tag in ner_tags]
            samples.append(
                {
                    "tokens": tokens,
                    "ner_tags": tag_ids,
                    "ner_tag_labels": ner_tags,
                }
            )
            for _, etype in entities:
                entity_counts[etype] = entity_counts.get(etype, 0) + 1

    random.shuffle(samples)
    return samples, entity_counts


def generate_transcript_org_records(
    num_samples: int,
) -> tuple[list[dict], dict[str, int]]:
    """Generate transcript-style multi-company training records."""
    samples: list[dict] = []
    entity_counts: dict[str, int] = {}

    for _ in range(num_samples):
        text, entities = _transcript_multi_org_sample()
        tokens, ner_tags = _tokenize_and_label(text, entities)
        tag_ids = [LABEL_TO_ID.get(tag, 0) for tag in ner_tags]
        samples.append(
            {
                "tokens": tokens,
                "ner_tags": tag_ids,
                "ner_tag_labels": ner_tags,
            }
        )
        for _, etype in entities:
            entity_counts[etype] = entity_counts.get(etype, 0) + 1

    random.shuffle(samples)
    return samples, entity_counts


def generate_transcript_org_dataset(
    num_samples: int,
    output_path: Path,
) -> dict[str, int]:
    """Generate transcript-style multi-company training data."""
    samples, entity_counts = generate_transcript_org_records(num_samples)
    _write_samples(output_path, samples)
    return entity_counts


def generate_hard_negative_benchmark(
    num_samples: int,
    output_path: Path,
) -> dict[str, int]:
    """Generate a targeted benchmark with hard-negative business text."""
    samples, entity_counts = generate_hard_negative_benchmark_records(num_samples)
    _write_samples(output_path, samples)
    return entity_counts


def _write_samples(output_path: Path, samples: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def generate_dataset(num_samples: int, output_path: Path) -> dict[str, int]:
    """Generate a mixed Faker-only synthetic dataset."""
    samples, entity_counts = generate_synthetic_records(num_samples)
    _write_samples(output_path, samples)

    return entity_counts


def generate_hybrid_dataset(
    num_samples: int,
    output_path: Path,
    *,
    nemotron_dir: str | Path | None = None,
    nemotron_fraction: float = 0.7,
    seed: int = 42,
) -> dict[str, int]:
    """Generate a training set mixed from Nemotron and local synthetic data."""
    nemotron_target = max(0, min(num_samples, int(num_samples * nemotron_fraction)))
    synthetic_target = num_samples - nemotron_target

    samples: list[dict] = []
    counts: Counter[str] = Counter()

    snapshot = resolve_nemotron_snapshot(nemotron_dir)
    if snapshot is not None and nemotron_target > 0:
        nemotron_samples = load_nemotron_samples(
            snapshot,
            split="train",
            limit=nemotron_target,
            seed=seed,
        )
        samples.extend(nemotron_samples)
        for sample in nemotron_samples:
            for tag in sample["ner_tag_labels"]:
                if tag != "O":
                    counts[tag.split("-", 1)[1]] += 1
        synthetic_target = num_samples - len(nemotron_samples)

    synthetic_samples, synthetic_counts = generate_synthetic_records(synthetic_target)
    samples.extend(synthetic_samples)
    counts.update(synthetic_counts)

    random.Random(seed).shuffle(samples)
    _write_samples(output_path, samples)

    return dict(counts)

    return entity_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic PII/PHI NER dataset for Obscura benchmarking"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Total number of samples to generate (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic.jsonl",
        help="Output JSONL file path (default: data/synthetic.jsonl)",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "nemotron", "hybrid", "hard-negative", "transcript-org"],
        default="hybrid",
        help="Dataset source to build (default: hybrid)",
    )
    parser.add_argument(
        "--nemotron-dir",
        type=str,
        default=None,
        help="Optional path to a downloaded Nemotron-PII snapshot",
    )
    parser.add_argument(
        "--nemotron-fraction",
        type=float,
        default=0.7,
        help="Fraction of rows to source from Nemotron in hybrid mode",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path

    print(f"Generating {args.num_samples} NER samples from {args.source} data...")
    if args.source == "synthetic":
        entity_counts = generate_dataset(args.num_samples, output_path)
    elif args.source == "nemotron":
        entity_counts = build_nemotron_jsonl(
            output_path,
            dataset_dir=args.nemotron_dir,
            split="train",
            limit=args.num_samples,
        )
    elif args.source == "hard-negative":
        entity_counts = generate_hard_negative_benchmark(args.num_samples, output_path)
    elif args.source == "transcript-org":
        entity_counts = generate_transcript_org_dataset(args.num_samples, output_path)
    else:
        entity_counts = generate_hybrid_dataset(
            args.num_samples,
            output_path,
            nemotron_dir=args.nemotron_dir,
            nemotron_fraction=args.nemotron_fraction,
        )

    print(f"\nWrote {args.num_samples} samples to {output_path}")
    print("\nEntity distribution:")
    for etype, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:>8}: {count}")
    print(f"  {'TOTAL':>8}: {sum(entity_counts.values())}")

    # Quick sanity check — read back first sample and verify alignment.
    with open(output_path) as f:
        first = json.loads(f.readline())
    assert len(first["tokens"]) == len(
        first["ner_tags"]
    ), "Token/tag length mismatch in first sample"
    print("\nSanity check passed: token/tag alignment OK.")


if __name__ == "__main__":
    main()
