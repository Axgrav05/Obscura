use anyhow::Context;
use ndarray::Array2;
use ort::{inputs, value::TensorRef};

use crate::NerModel;

const CONFIDENCE_THRESHOLD: f32 = 0.90;

/// Tokens per inference chunk — matches BERT's 512-token window.
const CHUNK_SIZE: usize = 512;

/// Stride overlap between consecutive chunks (in tokens).
/// Ensures entities near chunk boundaries are captured by both chunks
/// so the deduplication step can discard the duplicate.
const STRIDE: usize = 50;

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
    /// Run NER on a single token-level chunk.
    ///
    /// `ids`, `mask`, and `offsets` are slices from the full encoding.
    /// `offsets` contains global character positions (from the full-text
    /// encoding), so no re-mapping is needed after chunking.
    fn run_chunk(
        &self,
        text: &str,
        ids: &[u32],
        mask: &[u32],
        offsets: &[(usize, usize)],
    ) -> anyhow::Result<Vec<EntitySpan>> {
        let seq_len = ids.len();

        // Build input tensors: shape [1, seq_len]
        // ort v2: TensorArrayData is implemented for &Array2<i64> (reference to owned),
        // not for Array2::view() (ViewRepr). Pass &array, not array.view().
        let input_ids: Array2<i64> = Array2::from_shape_vec(
            (1, seq_len),
            ids.iter().map(|&x| x as i64).collect(),
        )
        .context("Failed to build input_ids tensor")?;

        let attention_mask: Array2<i64> = Array2::from_shape_vec(
            (1, seq_len),
            mask.iter().map(|&x| x as i64).collect(),
        )
        .context("Failed to build attention_mask tensor")?;

        // --- Run inference ---
        // ort v2: Session::run() requires &mut self — lock the Mutex for this call.
        // The guard must be bound to a let so it outlives the outputs borrow.
        // inputs! macro returns Vec (not Result) — no ? on the macro itself.
        // TensorRef::from_array_view takes &Array2 (reference to owned array).
        let mut session_guard = self.session.lock().unwrap();
        let outputs = session_guard
            .run(inputs![
                "input_ids" => TensorRef::<i64>::from_array_view(&input_ids)
                    .context("Failed to create input_ids TensorRef")?,
                "attention_mask" => TensorRef::<i64>::from_array_view(&attention_mask)
                    .context("Failed to create attention_mask TensorRef")?
            ])
            .context("ORT session run failed")?;

        // ort v2.0.0-rc.12: try_extract_tensor returns (&Shape, &[T]) where
        // Shape = Vec<i64>. Logits layout: [1, seq_len, num_labels] (row-major).
        let (shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract logits tensor")?;

        let num_labels = shape[2] as usize;

        // --- Decode: argmax + confidence filter + BIO assembly ---
        let mut spans: Vec<EntitySpan> = Vec::new();
        let mut current: Option<(String, usize, usize)> = None; // (label, start, end)

        for i in 0..seq_len {
            // Slice this token's scores from the flat row-major layout.
            // data index: [0, i, :] = data[i * num_labels .. (i+1) * num_labels]
            let token_scores: &[f32] = &data[i * num_labels..(i + 1) * num_labels];

            // Softmax — use total_cmp to avoid panic on NaN logits (stable since Rust 1.62).
            let max_val: f32 = token_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = token_scores.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

            let (best_idx, &confidence) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a): &(_, &f32), (_, b): &(_, &f32)| a.total_cmp(b))
                .unwrap(); // safe: probs is non-empty and finite after softmax

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

        Ok(spans)
    }

    /// Run NER inference on a piece of text.
    ///
    /// For texts that tokenize to ≤ 512 tokens, a single inference pass is
    /// used. For longer texts, the token sequence is split into overlapping
    /// chunks of CHUNK_SIZE with a STRIDE-token overlap. Spans from the
    /// overlap zone are deduplicated by a greedy cursor (first accepted wins).
    ///
    /// Returns entity spans sorted by start offset.
    pub fn predict(&self, text: &str) -> anyhow::Result<Vec<EntitySpan>> {
        // --- Tokenize the full text once ---
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        // offsets contains (global_char_start, global_char_end) per token —
        // already relative to the full text, so no re-mapping after chunking.
        let offsets = encoding.get_offsets();

        // Fast path: fits in one chunk — no splitting overhead.
        if ids.len() <= CHUNK_SIZE {
            let mut spans = self.run_chunk(text, ids, mask, offsets)?;
            spans.sort_by_key(|s| s.start);
            return Ok(spans);
        }

        // Multi-chunk path: stride loop over the full token sequence.
        let mut all_spans: Vec<EntitySpan> = Vec::new();
        let mut chunk_start = 0;

        loop {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(ids.len());
            let chunk_spans = self.run_chunk(
                text,
                &ids[chunk_start..chunk_end],
                &mask[chunk_start..chunk_end],
                &offsets[chunk_start..chunk_end],
            )?;
            all_spans.extend(chunk_spans);

            if chunk_end == ids.len() {
                break;
            }
            chunk_start += CHUNK_SIZE - STRIDE;
        }

        // Deduplicate spans from stride overlap: sort by start ascending, end descending.
        // End-descending ensures that if a partial entity from chunk N and a full entity
        // from chunk N+1 share the same start offset, the longer (full) span sorts first
        // and the greedy cursor accepts it — preventing the partial span from leaking the
        // tail of the entity to the LLM.
        all_spans.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
        let mut deduped: Vec<EntitySpan> = Vec::new();
        let mut cursor = 0usize;
        for span in all_spans {
            if span.start >= cursor {
                cursor = span.end;
                deduped.push(span);
            }
        }

        Ok(deduped)
    }
}
