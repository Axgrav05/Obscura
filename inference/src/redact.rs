use std::collections::HashMap;
use std::sync::OnceLock;


use crate::mapping::MappingDictionary;
use crate::ner::EntitySpan;

/// Types where regex is considered more reliable than BERT (e.g. structured IDs).
static REGEX_AUTHORITATIVE_TYPES: &[&str] = &[
    "SSN",
    "PHONE",
    "EMAIL",
    "MRN",
    "DOB",
    "CREDIT_CARD",
    "IPV4",
    "IPV6",
    "PASSPORT",
];

static COMPILED_PATTERNS: OnceLock<Vec<(&'static str, fancy_regex::Regex)>> = OnceLock::new();

static RAW_PATTERNS: &[(&str, &str)] = &[
    (
        "SSN",
        r"(?<!\w)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\w)",
    ),
    (
        "PHONE",
        r"(?<!\w)(?:\(\d{3}\)\s?\d{3}-\d{4}|\+\d{1,3}(?:[\s\-.]?\(?\d{1,4}\)?){2,6})(?!\w)",
    ),
    (
        "EMAIL",
        r"(?<!\w)[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}(?!\w)",
    ),
    (
        "MRN",
        r"(?i)(?<!\w)MRN-\d{7}(?!\w)",
    ),
    (
        "DOB",
        r"(?<!\w)(?:(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}|\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))(?!\w)",
    ),
    (
        "CREDIT_CARD",
        r"(?<!\w)(?:\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}|\d{16})(?!\w)",
    ),
    (
        "IPV4",
        r"(?<!\w)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\w)",
    ),
    (
        "IPV6",
        r"(?<!\w)(?:::(?:ffff(?::0{1,4})?:)?(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}|(?:[0-9a-fA-F]{1,4}:){6}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}|(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,6}:(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})?|::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,6})?)(?!\w)",
    ),
    (
        "PASSPORT",
        r"(?<!\w)[A-Z]\d{8}(?!\w)",
    ),
];

fn compiled_patterns() -> &'static Vec<(&'static str, fancy_regex::Regex)> {
    COMPILED_PATTERNS.get_or_init(|| {
        RAW_PATTERNS
            .iter()
            .map(|&(label, pattern)| {
                let re = fancy_regex::Regex::new(pattern).unwrap_or_else(|e| {
                    tracing::error!("Static regex failed to compile ({}): {}", label, e);
                    fancy_regex::Regex::new("").unwrap() // Should never happen with empty string
                });
                (label, re)
            })
            .collect()
    })
}

struct Span {
    start: usize,
    end: usize,
    label: String,
    source: &'static str,
}

pub fn redact(
    text: &str,
    bert_spans: Vec<EntitySpan>,
    skipped_entities: &[String],
) -> anyhow::Result<(String, MappingDictionary)> {
    let mut mapping = MappingDictionary::new();
    let mut counters: HashMap<String, usize> = HashMap::new();
    let mut all_spans: Vec<Span> = Vec::new();

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

    all_spans.sort_by(|a, b| {
        a.start.cmp(&b.start).then(b.end.cmp(&a.end)).then_with(|| {
            let a_auth = a.source == "regex" && REGEX_AUTHORITATIVE_TYPES.contains(&&a.label[..]);
            let b_auth = b.source == "regex" && REGEX_AUTHORITATIVE_TYPES.contains(&&b.label[..]);
            b_auth.cmp(&a_auth)
        })
    });

    let mut deduped: Vec<Span> = Vec::new();
    let mut cursor = 0usize;
    for span in all_spans {
        if span.start < cursor {
            continue;
        }
        cursor = span.end;
        deduped.push(span);
    }

    let mut output = String::with_capacity(text.len());
    let mut pos = 0usize;

    for span in deduped {
        output.push_str(&text[pos..span.start]);
        let count = counters.entry(span.label.clone()).or_insert(0);
        *count += 1;
        let token = format!("[{}_{}]", span.label, count);
        let original = text[span.start..span.end].to_string();
        mapping.insert(token.clone(), original);
        output.push_str(&token);
        pos = span.end;
    }
    output.push_str(&text[pos..]);

    Ok((output, mapping))
}

pub fn rehydrate(text: &str, mapping: &MappingDictionary) -> String {
    let mut output = text.to_string();
    let mut entries: Vec<(&String, &String)> = mapping.mappings.iter().collect();
    entries.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    for (token, original) in entries {
        output = output.replace(token.as_str(), original.as_str());
    }
    output
}
