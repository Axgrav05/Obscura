use anyhow::Context;
use ndarray::{Array2, Axis};
use ort::inputs;

use crate::NerModel;

const CONFIDENCE_THRESHOLD: f32 = 0.90;
const MAX_SEQ_LEN: usize = 512;

/// A detected entity span with its label and position in the original text.
#[derive(Debug, Clone)]
pub struct EntitySpan {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
}

/// Labels from dslim/bert-base-NER (id2label order).
/// Index must match the model's label_map.json exactly.
static LABEL_MAP: &[&str] = &[
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
];

impl NerModel {
    /// Run NER inference on a piece of text.
    /// Returns a list of entity spans sorted by start offset.
    pub fn predict(&self, text: &str) -> anyhow::Result<Vec<EntitySpan>> {
        // --- Tokenize ---
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let offsets = encoding.get_offsets(); // (char_start, char_end) per token

        // Truncate to MAX_SEQ_LEN
        let seq_len = ids.len().min(MAX_SEQ_LEN);

        // Build input tensors: shape [1, seq_len]
        let input_ids: Array2<i64> = Array2::from_shape_vec(
            (1, seq_len),
            ids[..seq_len].iter().map(|&x| x as i64).collect(),
        )
        .context("Failed to build input_ids tensor")?;

        let attention_mask: Array2<i64> = Array2::from_shape_vec(
            (1, seq_len),
            mask[..seq_len].iter().map(|&x| x as i64).collect(),
        )
        .context("Failed to build attention_mask tensor")?;

        // --- Run inference ---
        let outputs = self
            .session
            .run(inputs![
                "input_ids" => input_ids.view(),
                "attention_mask" => attention_mask.view()
            ]?)
            .context("ORT session run failed")?;

        // logits shape: [1, seq_len, num_labels]
        let logits = outputs["logits"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract logits tensor")?;

        // --- Decode: argmax + confidence filter + BIO assembly ---
        let mut spans: Vec<EntitySpan> = Vec::new();
        let mut current: Option<(String, usize, usize)> = None; // (label, start, end)

        let token_logits = logits.index_axis(Axis(0), 0); // shape [seq_len, num_labels]

        for (i, token_scores) in token_logits.outer_iter().enumerate() {
            // Softmax
            let max_val = token_scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = token_scores.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

            let (best_idx, &confidence) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            let label = LABEL_MAP.get(best_idx).copied().unwrap_or("O");

            let (char_start, char_end) = offsets.get(i).copied().unwrap_or((0, 0));

            if label == "O" || confidence < CONFIDENCE_THRESHOLD {
                // Flush any open span
                if let Some((lbl, s, e)) = current.take() {
                    spans.push(EntitySpan {
                        text: text[s..e].to_string(),
                        label: lbl,
                        start: s,
                        end: e,
                    });
                }
                continue;
            }

            if label.starts_with("B-") {
                // Flush previous span, start new one
                if let Some((lbl, s, e)) = current.take() {
                    spans.push(EntitySpan {
                        text: text[s..e].to_string(),
                        label: lbl,
                        start: s,
                        end: e,
                    });
                }
                let entity_type = label[2..].to_string();
                current = Some((entity_type, char_start, char_end));
            } else if label.starts_with("I-") {
                // Extend current span if labels match
                if let Some((ref lbl, s, _)) = current {
                    let entity_type = &label[2..];
                    if lbl == entity_type {
                        current = Some((lbl.clone(), s, char_end));
                    }
                }
            }
        }

        // Flush final span
        if let Some((lbl, s, e)) = current.take() {
            spans.push(EntitySpan {
                text: text[s..e].to_string(),
                label: lbl,
                start: s,
                end: e,
            });
        }

        spans.sort_by_key(|s| s.start);
        Ok(spans)
    }
}
