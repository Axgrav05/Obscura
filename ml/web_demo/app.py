"""
Obscura Web Demo - Real-time PII Redaction Visualizer
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os
import random
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.nemotron_data import load_nemotron_demo_texts, resolve_nemotron_snapshot
from ml.regex_detector import RegexDetector

app = Flask(__name__)
CORS(app)

# Global model cache
MODEL_CACHE = {}
REGEX_DETECTOR = RegexDetector()
ENTITY_SCORE_THRESHOLDS = {
    "MISC": 0.97,
}
DEFAULT_ENTITY_THRESHOLD = 0.90
MODEL_MAX_LENGTH = 512
MODEL_STRIDE = 128
SEGMENT_MIN_CHARS = 48


def _preferred_onnx_providers() -> list[str]:
    """Select ONNX providers, with an optional explicit override."""
    override = os.getenv("OBSCURA_ONNX_PROVIDER", "auto").strip().lower()
    available = set(ort.get_available_providers())

    if override == "cpu":
        return ["CPUExecutionProvider"]

    if override == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def resolve_bundle_dir() -> Path:
    """Locate the newest packaged ONNX bundle.

    Allows explicit override via OBSCURA_ONNX_BUNDLE.
    """
    env_bundle = os.getenv("OBSCURA_ONNX_BUNDLE")
    if env_bundle:
        bundle_dir = Path(env_bundle).expanduser()
        if (bundle_dir / "model.onnx").exists():
            return bundle_dir

    bundles_root = Path(__file__).parent.parent / "models/onnx"
    candidates = sorted(
        (path for path in bundles_root.glob("*/") if (path / "model.onnx").exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No packaged ONNX bundle found under ml/models/onnx")
    return candidates[0]


def load_model():
    """Load the ONNX model and tokenizer (cached)."""
    if 'session' in MODEL_CACHE:
        return MODEL_CACHE['session'], MODEL_CACHE['tokenizer'], MODEL_CACHE['label_map'], MODEL_CACHE['bundle_dir']

    bundle_dir = resolve_bundle_dir()
    model_path = bundle_dir / "model.onnx"

    # Load ONNX Runtime session
    session = ort.InferenceSession(
        str(model_path),
        providers=_preferred_onnx_providers(),
    )

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir.parent))

    # Load label map
    with open(bundle_dir / "label_map.json") as f:
        label_map = json.load(f)

    MODEL_CACHE['session'] = session
    MODEL_CACHE['tokenizer'] = tokenizer
    MODEL_CACHE['label_map'] = label_map
    MODEL_CACHE['bundle_dir'] = bundle_dir
    MODEL_CACHE['providers'] = session.get_providers()

    return session, tokenizer, label_map, bundle_dir


def generate_demo_text() -> str:
    """Prefer Nemotron source text for the demo, with synthetic fallback."""
    snapshot = resolve_nemotron_snapshot()
    if snapshot is not None:
        try:
            if 'demo_texts' not in MODEL_CACHE:
                MODEL_CACHE['demo_texts'] = load_nemotron_demo_texts(snapshot, split="train")
            texts = MODEL_CACHE.get('demo_texts', [])
            if texts:
                return random.choice(texts)
        except RuntimeError:
            pass
    return generate_synthetic_data()


def generate_synthetic_data():
    """Generate random synthetic PII data with high variability."""

    # Randomized data pools - much larger variety
    names = [
        "John Smith", "Sarah Johnson", "Alice Williams", "Bob Chen", "Maria Garcia",
        "David Lee", "Jennifer Martinez", "Michael O'Brien", "Rachel Kim", "Thomas Anderson",
        "Emma Brown", "James Wilson", "Olivia Davis", "William Moore", "Sophia Taylor",
        "Lucas Anderson", "Ava Thomas", "Henry Jackson", "Isabella White", "Alexander Harris",
        "Mia Martin", "Benjamin Thompson", "Charlotte Garcia", "Ethan Martinez", "Amelia Robinson"
    ]

    doctors = [
        "Dr. Sarah Johnson", "Dr. Michael Brown", "Dr. Lisa Chen", "Dr. Robert Martinez",
        "Dr. Emily Thompson", "Dr. David Kim", "Dr. Amanda Patel", "Dr. Christopher Lee",
        "Dr. Jessica Wang", "Dr. Daniel Rodriguez", "Dr. Michelle Taylor", "Dr. Kevin Singh"
    ]

    patients = [
        "Jane Doe", "Robert Wilson", "Emily Davis", "Carlos Rodriguez", "Patricia Lee",
        "James Taylor", "Mary Johnson", "Richard Anderson", "Linda Thomas", "Mark Jackson",
        "Elizabeth White", "Joseph Harris", "Susan Martin", "Charles Thompson", "Karen Garcia"
    ]

    hospitals = [
        "City Hospital", "St. Mary's Medical Center", "General Hospital",
        "Metropolitan Medical Center", "University Hospital", "Regional Health Center",
        "Memorial Hospital", "Community Medical Center", "Central Hospital", "Valley Health"
    ]

    companies = [
        "Acme Corp", "Global Industries", "Tech Solutions Inc", "DataSystems LLC",
        "CloudFirst Technologies", "SecureNet Inc", "Innovate Systems", "Digital Dynamics",
        "Apex Technologies", "Quantum Solutions", "NexGen Corp", "Synergy Systems",
        "Vanguard Tech", "Precision Analytics", "CoreLogic Inc"
    ]

    cities = [
        "New York", "Chicago", "San Francisco", "Boston", "Seattle", "Austin", "Denver",
        "Los Angeles", "Portland", "Miami", "Houston", "Atlanta", "Philadelphia", "Phoenix",
        "San Diego", "Dallas", "Detroit", "Minneapolis", "Washington DC"
    ]

    streets = [
        "Main St", "Oak Ave", "Park Blvd", "Cedar Lane", "Elm Street", "Maple Drive",
        "Washington Ave", "Lincoln Road", "Madison Street", "Jefferson Way", "Adams Court",
        "Pine Avenue", "Birch Lane", "Willow Drive", "Cherry Street", "Sunset Boulevard"
    ]

    offices = [
        "Building A - Floor 12", "Tower B - Suite 304", "Campus West - Room 210",
        "HQ - 5th Floor", "North Wing - Office 415", "South Tower - Level 8",
        "East Building - Suite 201", "Executive Tower - Floor 15", "Innovation Hub - Room 101"
    ]

    departments = [
        "Engineering", "Data Science", "Security", "Infrastructure", "Platform Engineering",
        "DevOps", "Backend Systems", "Cloud Architecture", "Research & Development"
    ]

    positions = [
        "Senior Software Engineer", "Staff Engineer", "Principal Engineer", "Lead Developer",
        "Senior Data Scientist", "Systems Architect", "Security Engineer", "DevOps Engineer",
        "Full Stack Developer", "Backend Engineer", "Research Scientist"
    ]

    conditions = [
        "acute symptoms", "respiratory distress", "cardiac evaluation", "chronic pain management",
        "post-operative care", "emergency assessment", "routine examination", "diagnostic testing"
    ]

    # Dynamic templates with randomized content
    medical_template = """MEDICAL RECORD - CONFIDENTIAL
Patient Name: {name}
Date of Birth: {dob}
Social Security Number: {ssn}
Medical Record Number: {mrn}
Primary Contact: {phone}
Email Address: {email}

ADMISSION SUMMARY:
Patient {patient} was admitted to {hospital} on {admission_date} presenting with {condition}.
Attending physician {doctor} conducted initial assessment. Patient history indicates previous
treatment at {prev_hospital} for related conditions. {additional_note}

CONTACT INFORMATION:
Emergency Contact: {emergency_name} at {emergency_phone}
Primary Care Physician: {doctor2}
Insurance Provider: {company}
Policy Number: {policy_num}

TECHNICAL NOTES:
- Chart accessed from workstation IP: {ip}
- Authorization token: {passport}
- Last updated: {update_date} at {update_time}
- Billing account: {cc}
- Laboratory results pending from {lab_location}

Patient resides at {address} and works at {employer} as {position}. Follow-up appointment
scheduled for {followup_date} with {doctor3}. Prescription refills authorized via {pharmacy}."""

    employee_template = """EMPLOYEE PERSONNEL FILE
=========================================
Name: {name}
Employee ID: EMP-{emp_id}
SSN: {ssn}
Date of Birth: {dob_iso}
Hire Date: {hire_date}

CONTACT DETAILS:
Personal Email: {email}
Work Phone: {phone}
Mobile: {phone2}
Home Address: {address}

EMPLOYMENT INFORMATION:
Department: {department}
Position: {position}
Manager: {manager}
Office Location: {office}, {city}
Salary Band: Level {level}

PAYROLL & BENEFITS:
Direct Deposit Account: {cc}
Health Insurance ID: {mrn}
401k Contribution: {contribution}%
FSA Account: {fsa_id}

SYSTEM ACCESS:
- VPN IP Range: {ip}/24
- SSH Key Fingerprint: {passport}
- Last Login: {login_date} from {login_ip}
- Workstation: {workstation}
- GitHub: {github_user}

Emergency Contact: {emergency_name}, relationship: {relationship}, phone: {emergency_phone}

Performance review scheduled by {manager} for {review_date}. Current projects include {project1}
for {company} and {project2} integration with {company2}."""

    financial_template = """FINANCIAL TRANSACTION LOG
Transaction ID: TXN-{txn_id}
Timestamp: {timestamp}
Processing Center: {processing_center}

CUSTOMER INFORMATION:
Account Holder: {name}
Date of Birth: {dob}
SSN: {ssn}
Customer ID: CUST-{cust_id}
Account Number: {account_num}

PAYMENT DETAILS:
Credit Card: {cc}
Card Type: {card_type}
Billing Address: {address}
Phone Verification: {phone}
Email Receipt: {email}

TRANSACTION SUMMARY:
Merchant: {company}
Merchant ID: {merchant_id}
Location: {merchant_address}
Amount: ${amount}
Currency: USD
Category: {category}
Authorization Code: {auth_code}

FRAUD PREVENTION:
IP Address: {ip}
Device ID: {device_id}
Device Type: {device_type}
Geolocation: {city}, {state}
Risk Score: {risk_score}
Passport Verification: {passport}
3D Secure: {secure_status}

Additional charges processed for {service_provider} services totaling ${amount2}. Account managed
by {manager} at {employer}. Secondary contact {emergency_name} at {emergency_phone}. Transaction
approved by {approver} on {approval_date}. Refund policy expires {expiry_date}."""

    security_template = """SYSTEM ACCESS LOG - CLASSIFIED
=========================================
Log ID: SEC-{log_id}
Timestamp: {timestamp}
Event: {event_type}
Severity: {severity}

USER CREDENTIALS:
Username: {username}
Full Name: {name}
Employee SSN: {ssn}
Clearance Level: {clearance_level}
Clearance ID: {passport}
DOB: {dob_iso}
Badge Number: {badge_num}

CONNECTION DETAILS:
Source IP: {ip}
Source Port: {src_port}
Destination IP: {dest_ip}
Destination Port: 8443
Session ID: {session_id}
Protocol: HTTPS/TLS 1.3
User Agent: {user_agent}

AUTHENTICATION:
MFA Token: {mfa_token}
MFA Method: {mfa_method}
Phone Verified: {phone}
Email Verified: {email}
Biometric: {biometric_type}
Previous Login: {prev_login}

ACCESSED RESOURCES:
- Database: {company}_prod_{env}
- Server: {hostname}
- Application: {app_name}
- Credit Card Vault: {cc} (last 4 digits)
- Medical Records System: {mrn}
- File Path: /{filepath}

LOCATION DATA:
Physical Location: {address}
Office: {office}, {city}
Building Access: {access_time}
Manager Approval: {manager}
Approving Official: {approver}

Related security incident #{incident_id} at {prev_hospital} reviewed by {doctor}. Emergency
contact {emergency_name} notified at {emergency_phone}. Session terminated at {end_time}.
Audit trail logged to {audit_server}. Next review scheduled for {review_date}."""

    templates = [medical_template, employee_template, financial_template, security_template]

    # Helper functions for random data
    def rand_ssn():
        return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"

    def rand_phone():
        return f"({random.randint(200,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}"

    def rand_dob():
        return f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/{random.randint(1950, 2000)}"

    def rand_dob_iso():
        return f"{random.randint(1950, 2000)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"

    def rand_email(name):
        domains = ["gmail.com", "outlook.com", "yahoo.com", "email.com", "company.com", "corporate.net"]
        return f"{name.lower().replace(' ', '.')}{random.randint(1,999)}@{random.choice(domains)}"

    def rand_cc():
        return f"{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"

    def rand_mrn():
        return f"MRN-{random.randint(1000000,9999999)}"

    def rand_ip():
        return f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

    def rand_passport():
        return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10000000,99999999)}"

    def rand_address():
        return f"{random.randint(100, 9999)} {random.choice(streets)}, {random.choice(cities)}"

    def rand_date(year_start=2020, year_end=2024):
        return f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/{random.randint(year_start, year_end)}"

    def rand_date_iso(year_start=2020, year_end=2024):
        return f"{random.randint(year_start, year_end)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"

    def rand_time():
        return f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"

    # Select random template and populate
    template = random.choice(templates)
    name = random.choice(names)
    patient = random.choice(patients)
    emergency_name = random.choice([n for n in names if n != name])
    manager = random.choice([n for n in names if n not in [name, emergency_name]])

    import datetime
    now = datetime.datetime.now()

    # Comprehensive field mapping with high randomization
    data = template.format(
        # People
        name=name,
        patient=patient,
        emergency_name=emergency_name,
        manager=manager,
        doctor=random.choice(doctors),
        doctor2=random.choice(doctors),
        doctor3=random.choice(doctors),
        approver=random.choice(names),

        # Organizations
        hospital=random.choice(hospitals),
        prev_hospital=random.choice(hospitals),
        company=random.choice(companies),
        company2=random.choice(companies),
        employer=random.choice(companies),
        service_provider=random.choice(companies),

        # Locations
        city=random.choice(cities),
        state=random.choice(["NY", "CA", "TX", "IL", "WA", "CO", "FL", "MA"]),
        office=random.choice(offices),
        address=rand_address(),
        merchant_address=rand_address(),
        lab_location=random.choice(cities),
        processing_center=f"{random.choice(cities)} Processing Center",

        # Employment
        department=random.choice(departments),
        position=random.choice(positions),
        level=random.randint(3, 7),
        contribution=random.choice([6, 8, 10, 12, 15]),
        relationship=random.choice(["spouse", "parent", "sibling", "partner"]),

        # PII
        ssn=rand_ssn(),
        phone=rand_phone(),
        phone2=rand_phone(),
        emergency_phone=rand_phone(),
        dob=rand_dob(),
        dob_iso=rand_dob_iso(),
        email=rand_email(name),
        cc=rand_cc(),
        mrn=rand_mrn(),
        passport=rand_passport(),

        # Network/System
        ip=rand_ip(),
        login_ip=rand_ip(),
        dest_ip=rand_ip(),
        username=name.lower().replace(" ", "."),
        session_id=''.join(random.choices('0123456789abcdef', k=32)),
        device_id=''.join(random.choices('0123456789ABCDEF', k=16)),
        mfa_token=''.join(random.choices('0123456789', k=6)),
        hostname=f"srv-{random.choice(['prod', 'dev', 'staging'])}-{random.randint(1,99):02d}",
        workstation=f"WS-{random.randint(1000,9999)}",
        audit_server=f"audit-{random.randint(1,10):02d}.internal",
        github_user=name.lower().replace(" ", ""),

        # IDs
        emp_id=random.randint(100000, 999999),
        cust_id=random.randint(100000, 999999),
        txn_id=random.randint(1000000, 9999999),
        log_id=random.randint(100000, 999999),
        incident_id=random.randint(1000, 9999),
        policy_num=f"POL-{random.randint(1000000, 9999999)}",
        fsa_id=f"FSA-{random.randint(100000, 999999)}",
        account_num=f"{random.randint(100000000, 999999999)}",
        merchant_id=f"MER-{random.randint(100000, 999999)}",
        badge_num=f"B{random.randint(10000, 99999)}",
        auth_code=f"{''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=8))}",

        # Dates
        hire_date=rand_date_iso(2015, 2023),
        admission_date=rand_date(),
        update_date=rand_date(),
        followup_date=rand_date(2024, 2026),
        review_date=rand_date(2024, 2026),
        approval_date=rand_date(),
        expiry_date=rand_date(2025, 2026),
        login_date=rand_date_iso(),
        prev_login=rand_date_iso(),
        access_time=rand_time(),

        # Times
        update_time=rand_time(),
        end_time=rand_time(),
        timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),

        # Monetary
        amount=f"{random.randint(100, 9999)}.{random.randint(0,99):02d}",
        amount2=f"{random.randint(50, 999)}.{random.randint(0,99):02d}",

        # Medical
        condition=random.choice(conditions),
        pharmacy=random.choice(companies),
        additional_note=random.choice([
            "No known allergies reported.",
            "Patient requested pain management consultation.",
            "Awaiting specialist referral.",
            "Previous imaging studies reviewed."
        ]),

        # Technical/Security
        env=random.choice(["east", "west", "central"]),
        app_name=random.choice(["Portal", "Dashboard", "Analytics", "Admin Console"]),
        filepath=f"{random.choice(['data', 'logs', 'reports'])}/{random.choice(['2024', '2023'])}/file_{random.randint(1000,9999)}.dat",
        src_port=random.randint(40000, 60000),
        user_agent=random.choice(["Chrome/120.0", "Firefox/119.0", "Safari/17.0"]),
        mfa_method=random.choice(["SMS", "Authenticator App", "Hardware Token"]),
        biometric_type=random.choice(["Fingerprint Match", "Face Recognition", "Iris Scan"]),
        event_type=random.choice(["Authorized Access Attempt", "Login Success", "Resource Access"]),
        severity=random.choice(["INFO", "WARNING", "NOTICE"]),
        clearance_level=random.choice(["Level 3", "Level 4", "Level 5"]),
        card_type=random.choice(["Visa", "Mastercard", "Amex", "Discover"]),
        category=random.choice(["Professional Services", "Healthcare", "Technology", "Consulting"]),
        risk_score=random.choice(["Low", "Medium-Low", "Minimal"]),
        secure_status=random.choice(["Verified", "Passed", "Authenticated"]),
        device_type=random.choice(["Mobile", "Desktop", "Tablet"]),

        # Projects (for employee file)
        project1=random.choice(["cloud infrastructure migration", "security enhancement", "platform modernization"]),
        project2=random.choice(["API integration", "data pipeline", "monitoring system"])
    )

    return data


def _iter_segment_ranges(text: str) -> list[tuple[int, int]]:
    """Return document ranges that should receive independent NER passes."""
    ranges = [(0, len(text))]

    if "\n\n" not in text:
        return ranges

    cursor = 0
    for block in text.split("\n\n"):
        start = text.find(block, cursor)
        if start < 0:
            continue
        end = start + len(block)
        cursor = end
        if len(block.strip()) < SEGMENT_MIN_CHARS:
            continue
        if (start, end) not in ranges:
            ranges.append((start, end))

    return sorted(ranges)


def _should_run_focused_segment_passes(text: str, entities: list[dict]) -> bool:
    """Run extra segment passes when the document has distinct paragraph blocks."""
    return "\n\n" in text


def _run_bert_ner_segment(text: str, start_offset: int = 0):
    """Run BERT NER inference over a text segment and return document offsets."""
    session, tokenizer, label_map, _bundle_dir = load_model()

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=MODEL_MAX_LENGTH,
        stride=MODEL_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    # Prepare ONNX inputs
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    # Add token_type_ids if needed
    input_names = {inp.name for inp in session.get_inputs()}
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"], dtype=np.int64)

    # Run inference
    outputs = session.run(None, ort_inputs)
    logits = outputs[0]
    predictions = np.argmax(logits, axis=-1)
    stabilized_logits = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(stabilized_logits)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)

    # Get attention weights for visualization (from first layer)
    attention_weights = None
    if len(outputs) > 1:
        attention_weights = outputs[1]

    # Convert token predictions into word-level spans using tokenizer offsets.
    words = []
    for chunk_index in range(len(inputs["input_ids"])):
        word_ids = inputs.word_ids(chunk_index)
        offset_mapping = inputs["offset_mapping"][chunk_index]

        token_index = 0
        while token_index < len(word_ids):
            word_id = word_ids[token_index]
            if word_id is None:
                token_index += 1
                continue

            word_start_index = token_index
            while (
                token_index + 1 < len(word_ids)
                and word_ids[token_index + 1] == word_id
            ):
                token_index += 1
            word_end_index = token_index

            start = int(offset_mapping[word_start_index][0])
            end = int(offset_mapping[word_end_index][1])
            if start >= end:
                token_index += 1
                continue

            predicted_label = int(predictions[chunk_index][word_start_index])
            label = label_map.get(str(predicted_label), "O")
            score = float(probabilities[chunk_index][word_start_index][predicted_label])
            words.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "score": score,
                }
            )
            token_index += 1

    words.sort(key=lambda word: (word["start"], word["end"], -word["score"]))
    deduped_words = []
    seen_word_spans = set()
    for word in words:
        word_key = (word["start"], word["end"], word["label"])
        if word_key in seen_word_spans:
            continue
        seen_word_spans.add(word_key)
        deduped_words.append(word)
    words = deduped_words

    entities = []
    current_entity = None
    current_start = None
    current_end = None
    current_scores = []

    def flush_current_entity() -> None:
        nonlocal current_entity, current_start, current_end, current_scores
        if current_entity is None or current_start is None or current_end is None:
            return
        score = round(float(np.mean(current_scores)), 4)
        threshold = ENTITY_SCORE_THRESHOLDS.get(current_entity, DEFAULT_ENTITY_THRESHOLD)
        if score < threshold:
            current_entity = None
            current_start = None
            current_end = None
            current_scores = []
            return
        entities.append(
            {
                "type": current_entity,
                "text": text[current_start:current_end],
                "start": current_start + start_offset,
                "end": current_end + start_offset,
                "score": score,
                "source": "bert",
            }
        )
        current_entity = None
        current_start = None
        current_end = None
        current_scores = []

    for word in words:
        label = word["label"]

        if label == "O":
            flush_current_entity()
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or current_entity != entity_type:
            flush_current_entity()
            current_entity = entity_type
            current_start = word["start"]
            current_end = word["end"]
            current_scores = [word["score"]]
            continue

        current_end = word["end"]
        current_scores.append(word["score"])

    flush_current_entity()

    # Get logit distribution for visualization
    logit_stats = {
        "max": float(np.max(logits)),
        "min": float(np.min(logits)),
        "mean": float(np.mean(logits)),
        "std": float(np.std(logits))
    }

    return entities, logit_stats


def run_bert_ner(text: str):
    """Run one full-document pass, then selectively fall back to focused tail segments."""
    full_entities, logit_stats = _run_bert_ner_segment(text)
    all_entities = list(full_entities)

    if _should_run_focused_segment_passes(text, full_entities):
        for start, end in _iter_segment_ranges(text)[1:]:
            segment_entities, _segment_logit_stats = _run_bert_ner_segment(
                text[start:end],
                start_offset=start,
            )
            all_entities.extend(segment_entities)

    return merge_demo_entities(text, all_entities), logit_stats or {
        "max": 0.0,
        "min": 0.0,
        "mean": 0.0,
        "std": 0.0,
    }


def run_regex_detection(text: str):
    """Run regex-based detection."""
    entities = REGEX_DETECTOR.detect(text)

    return [
        {
            "type": e.entity_type,
            "text": e.text,
            "start": e.start,
            "end": e.end,
            "score": e.score,
            "source": "regex",
        }
        for e in entities
    ]


def merge_demo_entities(text: str, entities: list[dict]) -> list[dict]:
    """Collapse duplicate or overlapping entities for stable UI rendering."""
    merged = []
    for entity in sorted(
        entities,
        key=lambda item: (item["start"], -(item["end"] - item["start"])),
    ):
        if entity["end"] <= entity["start"]:
            continue

        if not merged:
            merged.append(entity)
            continue

        current = merged[-1]
        if entity["start"] >= current["end"]:
            gap = text[current["end"] : entity["start"]]
            if (
                entity["type"] == current["type"]
                and gap.isspace()
                and "\n" not in gap
                and current.get("source") == "bert"
                and entity.get("source") == "bert"
            ):
                merged[-1] = {
                    **current,
                    "text": text[current["start"] : entity["end"]],
                    "end": entity["end"],
                    "score": round(
                        max(current.get("score", 0.0), entity.get("score", 0.0)),
                        4,
                    ),
                    "source": "regex"
                    if "regex" in {current.get("source"), entity.get("source")}
                    else current.get("source", entity.get("source", "bert")),
                }
                continue

            merged.append(entity)
            continue

        current_length = current["end"] - current["start"]
        entity_length = entity["end"] - entity["start"]
        current_score = current.get("score", 0.0)
        entity_score = entity.get("score", 0.0)

        replace_current = False
        if entity["source"] == "regex" and current.get("source") != "regex":
            replace_current = True
        elif entity_length > current_length:
            replace_current = True
        elif entity_length == current_length and entity_score > current_score:
            replace_current = True

        if replace_current:
            merged[-1] = entity

    return merged


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/generate', methods=['GET'])
def generate():
    """Generate synthetic data."""
    return jsonify({
        "text": generate_demo_text()
    })


@app.route('/api/redact', methods=['POST'])
def redact():
    """Process text and return redacted entities."""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Run both detection systems
    bert_entities, logit_stats = run_bert_ner(text)
    regex_entities = run_regex_detection(text)

    # Combine and deduplicate entities
    all_entities = merge_demo_entities(text, bert_entities + regex_entities)

    return jsonify({
        "entities": all_entities,
        "bert_count": len(bert_entities),
        "regex_count": len(regex_entities),
        "total_count": len(all_entities),
        "logit_stats": logit_stats
    })


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 OBSCURA PII REDACTION DEMO")
    print("=" * 80)
    print("\n📦 Loading model...")
    _session, _tokenizer, _label_map, bundle_dir = load_model()  # Pre-load model
    print("✅ Model loaded successfully!\n")
    print(f"📁 Bundle: {bundle_dir}")
    print("🌐 Starting web server...")
    print("📍 URL: http://localhost:8080")
    print("\nPress Ctrl+C to stop the server.\n")
    print("=" * 80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=8080)
