use std::collections::HashMap;

use crate::mapping::MappingDictionary;
use crate::ner::EntitySpan;

/// Regex patterns for deterministic PII types (Presidio-style).
/// These run alongside BERT and cover structured patterns BERT misses.
static REGEX_PATTERNS: &[(&str, &str)] = &[
    ("SSN",         r"\b\d{3}-\d{2}-\d{4}\b"),
    ("PHONE",       r"\b(\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    ("EMAIL",       r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ("MRN",         r"\bMRN[\s:]*\d{4,10}\b"),
    ("DOB",         r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    ("CREDIT_CARD", r"\b(?:\d[ -]?){13,16}\b"),
    ("IPV4",        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("PASSPORT",    r"\b[A-Z]{1,2}\d{6,9}\b"),
];

/// Perform redaction on a string.
/// Returns the redacted string and a populated MappingDictionary.
pub fn redact(
    text: &str,
    bert_spans: Vec<EntitySpan>,
    skipped_entities: &[String],
) -> anyhow::Result<(String, MappingDictionary)> {
    let mut mapping = MappingDictionary::new();
    let mut counters: HashMap<String, usize> = HashMap::new();

    // Collect all spans: BERT + regex, then sort by start offset
    let mut all_spans: Vec<(usize, usize, String)> = Vec::new();

    // BERT spans
    for span in bert_spans {
        if skipped_entities.iter().any(|s| s.eq_ignore_ascii_case(&span.label)) {
            continue;
        }
        all_spans.push((span.start, span.end, span.label));
    }

    // Regex spans
    for &(label, pattern) in REGEX_PATTERNS {
        if skipped_entities.iter().any(|s| s.eq_ignore_ascii_case(label)) {
            continue;
        }
        let re = regex::Regex::new(pattern)?;
        for m in re.find_iter(text) {
            all_spans.push((m.start(), m.end(), label.to_string()));
        }
    }

    // Sort by start; on tie, prefer longer span (more specific match)
    all_spans.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

    // Remove overlapping spans (keep first / longest)
    let mut deduped: Vec<(usize, usize, String)> = Vec::new();
    let mut cursor = 0usize;
    for (start, end, label) in all_spans {
        if start < cursor {
            continue; // overlaps with a previously accepted span
        }
        deduped.push((start, end, label));
        cursor = end;
    }

    // Build redacted string
    let mut output = String::with_capacity(text.len());
    let mut pos = 0usize;

    for (start, end, label) in deduped {
        // Append unchanged text before this span
        output.push_str(&text[pos..start]);

        // Generate token, e.g. [PERSON_1]
        let count = counters.entry(label.clone()).or_insert(0);
        *count += 1;
        let token = format!("[{}_{}]", label, count);

        let original = text[start..end].to_string();
        mapping.insert(token.clone(), original);
        output.push_str(&token);

        pos = end;
    }

    // Append any trailing text
    output.push_str(&text[pos..]);

    Ok((output, mapping))
}

/// Restore original PII values in a response body by replacing tokens.
/// Operates on raw bytes; assumes UTF-8 text (JSON response from LLM).
pub fn rehydrate(text: &str, mapping: &MappingDictionary) -> String {
    let mut output = text.to_string();
    // Iterate in insertion order — HashMap doesn't guarantee order, so sort
    // by token length descending to avoid partial replacements
    // (e.g. [PER_1] being replaced inside [PER_10])
    let mut entries: Vec<(&String, &String)> = mapping.mappings.iter().collect();
    entries.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    for (token, original) in entries {
        output = output.replace(token.as_str(), original.as_str());
    }
    output
}
