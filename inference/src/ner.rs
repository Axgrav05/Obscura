use anyhow::Context;
use ndarray::Array2;
use ort::value::TensorRef;
use serde::Serialize;

use crate::NerModel;

const CHUNK_SIZE: usize = 512;
const STRIDE: usize = 64;
const CONFIDENCE_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EntitySpan {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
}

static LABEL_MAP: &[&str] = &[
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
];

impl NerModel {
    fn run_chunk(
        &self,
        text: &str,
        ids: &[u32],
        mask: &[u32],
        offsets: &[(usize, usize)],
    ) -> anyhow::Result<Vec<EntitySpan>> {
        let seq_len = ids.len();

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

        let mut session_guard = self.session.lock().unwrap();
        let outputs = session_guard
            .run(ort::inputs![
                "input_ids" => TensorRef::<i64>::from_array_view(&input_ids)?,
                "attention_mask" => TensorRef::<i64>::from_array_view(&attention_mask)?
            ])
            .context("ORT session run failed")?;

        let (shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract logits tensor")?;

        let num_labels = shape[2] as usize;
        let mut spans: Vec<EntitySpan> = Vec::new();
        let mut current: Option<(String, usize, usize)> = None;

        for i in 0..seq_len {
            let token_scores: &[f32] = &data[i * num_labels..(i + 1) * num_labels];
            let max_val: f32 = token_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = token_scores.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

            let (best_idx, &confidence) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a): &(_, &f32), (_, b): &(_, &f32)| a.total_cmp(b))
                .unwrap();

            let label = LABEL_MAP.get(best_idx).copied().unwrap_or("O");
            let (char_start, char_end) = offsets.get(i).copied().unwrap_or((0, 0));

            if label == "O" || confidence < CONFIDENCE_THRESHOLD {
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
                if let Some((ref lbl, s, _)) = current {
                    let entity_type = &label[2..];
                    if lbl == entity_type {
                        current = Some((lbl.clone(), s, char_end));
                    }
                }
            }
        }

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

    pub fn predict(&self, text: &str) -> anyhow::Result<Vec<EntitySpan>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let offsets = encoding.get_offsets();

        if ids.len() <= CHUNK_SIZE {
            let mut spans = self.run_chunk(text, ids, mask, offsets)?;
            spans.sort_by_key(|s| s.start);
            return Ok(spans);
        }

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
