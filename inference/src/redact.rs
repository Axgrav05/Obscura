use std::collections::HashMap;
use std::sync::OnceLock;

use crate::mapping::MappingDictionary;
use crate::ner::EntitySpan;

/// Entity types where regex is authoritative over BERT on exact-span overlaps.
/// Mirrors Python `REGEX_AUTHORITATIVE_TYPES` in `ml/pii_engine.py`.
const REGEX_AUTHORITATIVE_TYPES: &[&str] = &[
    "SSN", "PHONE", "EMAIL", "MRN", "DOB", "CREDIT_CARD", "IPV4", "IPV6", "PASSPORT",
];

/// Production-grade regex patterns mirroring `ml/regex_detector.py`.
///
/// Uses `fancy_regex` for lookbehind/lookahead support (`(?<!\w)`, `(?!\w)`,
/// `(?!000|666|...)`) that the `regex` crate does not support.
///
/// Patterns are compiled once at startup via `compiled_patterns()` and
/// stored in `COMPILED_PATTERNS` — never inside the hot request path.
static COMPILED_PATTERNS: OnceLock<Vec<(&'static str, fancy_regex::Regex)>> = OnceLock::new();

static RAW_PATTERNS: &[(&str, &str)] = &[
    // SSN: dashed format. Rejects IRS-invalid area codes (000, 666, 9xx),
    // invalid group (00), and invalid serial (0000).
    // Mirrors: (?<!\w)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\w)
    (
        "SSN",
        r"(?<!\w)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\w)",
    ),
    // PHONE: US (xxx) xxx-xxxx format and E.164 international (+country...digits).
    // Mirrors: (?<!\w)(?:\(\d{3}\)\s?\d{3}-\d{4}|\+\d{1,3}(?:[\s\-.]?\(?\d{1,4}\)?){2,6})(?!\w)
    (
        "PHONE",
        r"(?<!\w)(?:\(\d{3}\)\s?\d{3}-\d{4}|\+\d{1,3}(?:[\s\-.]?\(?\d{1,4}\)?){2,6})(?!\w)",
    ),
    // EMAIL: RFC-style local@domain.tld
    // Mirrors: (?<!\w)[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?!\w)
    (
        "EMAIL",
        r"(?<!\w)[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}(?!\w)",
    ),
    // MRN: Medical Record Number — MRN-XXXXXXX (7 digits, case-insensitive prefix).
    // Mirrors: (?<!\w)MRN-\d{7}(?!\w) with re.IGNORECASE
    (
        "MRN",
        r"(?i)(?<!\w)MRN-\d{7}(?!\w)",
    ),
    // DOB: Strict MM/DD/YYYY or YYYY-MM-DD with validated month/day/year ranges.
    // Month: 01-12, Day: 01-31, Year: 1900-2099.
    // Mirrors: (?<!\w)(?:MM/DD/YYYY|YYYY-MM-DD strict)(?!\w)
    (
        "DOB",
        r"(?<!\w)(?:(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}|\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))(?!\w)",
    ),
    // CREDIT_CARD: 16-digit dashed/spaced format (4-4-4-4) OR bare 16 digits.
    // Mirrors: (?<!\w)(?:\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}|\d{16})(?!\w)
    // Undashed \d{16} is low false-positive risk in enterprise PII contexts (CLAUDE.md).
    (
        "CREDIT_CARD",
        r"(?<!\w)(?:\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}|\d{16})(?!\w)",
    ),
    // IPV4: Dotted quad with 0-255 octet validation.
    // Mirrors: (?<!\w)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}...(?!\w)
    (
        "IPV4",
        r"(?<!\w)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\w)",
    ),
    // IPV6: Full/zero-compressed/IPv4-mapped forms. MAC rejected (MAC has no ::).
    // IPv4-mapped branch listed first to prevent generic :: from partial-consuming it.
    // Mirrors: ml/regex_detector.py _ipv6 (4-branch alternation)
    (
        "IPV6",
        r"(?<!\w)(?:::(?:ffff(?::0{1,4})?:)?(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}|(?:[0-9a-fA-F]{1,4}:){6}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}|(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,6}:(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})?|::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,6})?)(?!\w)",
    ),
    // PASSPORT: US format — exactly 1 uppercase letter + 8 digits.
    // Mirrors: (?<!\w)[A-Z]\d{8}(?!\w)
    (
        "PASSPORT",
        r"(?<!\w)[A-Z]\d{8}(?!\w)",
    ),
];

/// Returns regex patterns compiled once at startup via OnceLock.
/// Panics at process start if any static pattern fails to compile (a bug, not a runtime error).
fn compiled_patterns() -> &'static Vec<(&'static str, fancy_regex::Regex)> {
    COMPILED_PATTERNS.get_or_init(|| {
        RAW_PATTERNS
            .iter()
            .map(|&(label, pattern)| {
                let re = fancy_regex::Regex::new(pattern).unwrap_or_else(|e| {
                    panic!("Static regex failed to compile ({}): {}", label, e)
                });
                (label, re)
            })
            .collect()
    })
}

/// Internal span representation with source annotation for conflict resolution.
struct Span {
    start: usize,
    end: usize,
    label: String,
    /// "bert" or "regex" — used to apply REGEX_AUTHORITATIVE_TYPES priority.
    source: &'static str,
}

/// Perform redaction on a string.
///
/// Merges BERT and regex spans with conflict resolution that mirrors
/// Python's `merge_entities()`:
///  - On exact-span ties: regex wins for REGEX_AUTHORITATIVE_TYPES.
///  - On partial/nested overlap: longer span wins (via sort-then-greedy-cursor).
///
/// Returns the redacted string and a populated MappingDictionary.
pub fn redact(
    text: &str,
    bert_spans: Vec<EntitySpan>,
    skipped_entities: &[String],
) -> anyhow::Result<(String, MappingDictionary)> {
    let mut mapping = MappingDictionary::new();
    let mut counters: HashMap<String, usize> = HashMap::new();

    // Collect all spans: BERT + regex with source annotation.
    let mut all_spans: Vec<Span> = Vec::new();

    // BERT spans
    for span in bert_spans {
        if skipped_entities.iter().any(|s| s.eq_ignore_ascii_case(&span.label)) {
            continue;
        }
        all_spans.push(Span {
            start: span.start,
            end: span.end,
            label: span.label,
            source: "bert",
        });
    }

    // Regex spans — compiled_patterns() initializes once on first call.
    for (label, re) in compiled_patterns() {
        if skipped_entities.iter().any(|s| s.eq_ignore_ascii_case(label)) {
            continue;
        }
        for m in re.find_iter(text) {
            let m = m?;
            all_spans.push(Span {
                start: m.start(),
                end: m.end(),
                label: label.to_string(),
                source: "regex",
            });
        }
    }

    // Sort: start ascending, end descending (longer first on start tie),
    // then regex-authoritative types first on exact-span tie.
    // This ensures the greedy cursor accepts the correct winner for exact overlaps.
    all_spans.sort_by(|a, b| {
        a.start.cmp(&b.start).then(b.end.cmp(&a.end)).then_with(|| {
            let a_auth =
                a.source == "regex" && REGEX_AUTHORITATIVE_TYPES.contains(&&a.label[..]);
            let b_auth =
                b.source == "regex" && REGEX_AUTHORITATIVE_TYPES.contains(&&b.label[..]);
            // true > false — regex-authoritative comes first
            b_auth.cmp(&a_auth)
        })
    });

    // Greedy dedup: first accepted span per position wins.
    // Combined with the sort above: on exact-span ties, regex-authoritative
    // types are accepted and BERT duplicates at the same position are skipped.
    let mut deduped: Vec<Span> = Vec::new();
    let mut cursor = 0usize;
    for span in all_spans {
        if span.start < cursor {
            continue; // overlaps with an already-accepted span
        }
        cursor = span.end;
        deduped.push(span);
    }

    // Build redacted string (process spans in order).
    let mut output = String::with_capacity(text.len());
    let mut pos = 0usize;

    for span in deduped {
        // Append unchanged text before this span
        output.push_str(&text[pos..span.start]);

        // Generate token, e.g. [PERSON_1]
        let count = counters.entry(span.label.clone()).or_insert(0);
        *count += 1;
        let token = format!("[{}_{}]", span.label, count);

        let original = text[span.start..span.end].to_string();
        mapping.insert(token.clone(), original);
        output.push_str(&token);

        pos = span.end;
    }

    // Append any trailing text
    output.push_str(&text[pos..]);

    Ok((output, mapping))
}

/// Restore original PII values in a response body by replacing tokens.
/// Operates on raw bytes; assumes UTF-8 text (JSON response from LLM).
///
/// Tokens are sorted by length descending before replacement to prevent
/// [PER_1] from being partially matched inside [PER_10].
pub fn rehydrate(text: &str, mapping: &MappingDictionary) -> String {
    let mut output = text.to_string();
    let mut entries: Vec<(&String, &String)> = mapping.mappings.iter().collect();
    entries.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    for (token, original) in entries {
        output = output.replace(token.as_str(), original.as_str());
    }
    output
}
