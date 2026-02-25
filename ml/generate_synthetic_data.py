"""
Synthetic PII/PHI Dataset Generator for Obscura NER Benchmarking

Generates labeled NER samples using Faker with BIO tagging for both
enterprise (names, orgs, SSNs, phones, emails) and clinical (MRN,
patient names, DOB) entity types.

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
from pathlib import Path

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

# BIO label vocabulary — shared with evaluate.py and the model pipeline.
LABEL_LIST: list[str] = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-SSN",
    "I-SSN",
    "B-PHONE",
    "I-PHONE",
    "B-EMAIL",
    "I-EMAIL",
    "B-MRN",
    "I-MRN",
    "B-DOB",
    "I-DOB",
    "B-MISC",
    "I-MISC",
]

LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(LABEL_LIST)}


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
    """Generate a US phone number."""
    a = random.randint(200, 999)
    b = random.randint(200, 999)
    c = random.randint(1000, 9999)
    return f"({a}) {b}-{c}"


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

    for entity_text, entity_type in entities:
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
    """
    # Templates with dashed SSNs (original).
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
    ]

    person = fake.name()
    org = fake.company()
    phone = _phone()
    email = fake.email()
    location = fake.city()

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
    """Generate one clinical-domain sample with PHI entities."""
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
    ]

    person = fake.name()
    mrn = _mrn()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%m/%d/%Y")
    condition = random.choice(conditions)
    medication = random.choice(medications)

    template_fn = random.choice(templates)
    text, entities = template_fn(person, mrn, dob, condition, medication)
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


def generate_dataset(num_samples: int, output_path: Path) -> dict[str, int]:
    """Generate a mixed enterprise + clinical + negative synthetic dataset.

    Splits: 50% enterprise (mix of dashed and dashless SSNs), 35% clinical,
    15% negative (9-digit numbers in non-SSN context for disambiguation
    testing).

    Args:
        num_samples: Total number of samples to generate.
        output_path: Path to write the JSONL file.

    Returns:
        Dictionary of entity type counts for the summary.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entity_counts: dict[str, int] = {}
    num_enterprise = int(num_samples * 0.50)
    num_clinical = int(num_samples * 0.35)
    num_negative = num_samples - num_enterprise - num_clinical

    samples: list[dict] = []

    generators = (
        [(_enterprise_sample, num_enterprise)]
        + [(_clinical_sample, num_clinical)]
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

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

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
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path

    print(f"Generating {args.num_samples} synthetic NER samples...")
    entity_counts = generate_dataset(args.num_samples, output_path)

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
