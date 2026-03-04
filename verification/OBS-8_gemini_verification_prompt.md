# Gemini Verification Prompt for OBS-8

Copy the prompt below into Gemini, along with the file contents listed in the "Files to Provide" section.

---

## Files to Provide

Attach or paste the contents of these files when running this prompt:

1. `ml/regex_detector.py`
2. `ml/evaluate.py`
3. `ml/pii_engine.py`
4. `ml/fine_tune.py`
5. `ml/generate_synthetic_data.py`
6. `ml/BENCHMARKS.md`
7. `verification/OBS-8_verification_report.md`
8. `ml/tests/test_ssn_context.py`

---

## Prompt

```
You are reviewing the work done on OBS-8 (Run Model Benchmarks and Select Final Model) for Project Obscura, a zero-trust security proxy that redacts PII/PHI using a hybrid BERT + regex NER pipeline.

The acceptance criteria for OBS-8 are:
- Macro F1 >= 90%
- Per-entity F1 >= 85% each
- Macro Precision >= 92%
- Macro Recall >= 88%
- Latency p95 <= 30ms on EC2 t3.medium
- Peak RAM <= 1.5 GB inference
- Produce a benchmark report comparing candidates
- Recommend a final model meeting the above criteria

The work performed includes:
1. Benchmarking 3 candidate models (dslim/bert-base-NER, dslim/distilbert-NER, StanfordAIMI/stanford-deidentifier-base)
2. Adding PHONE, EMAIL, and MRN regex patterns to the regex detector
3. Tuning the BERT confidence threshold from 0.85 to 0.90
4. Augmenting the synthetic dataset with MISC (nationality) entities
5. Fine-tuning dslim/bert-base-NER on the synthetic dataset (5 epochs, lr 2e-5)
6. Achieving macro F1 = 0.9576 on the full hybrid pipeline

Please verify the following in the attached files:

### 1. Regex Pattern Correctness (regex_detector.py)
- PHONE regex: `(?<!\w)\(\d{3}\)\s?\d{3}-\d{4}(?!\d)` — does it correctly match US phone numbers in (NNN) NNN-NNNN format? Are there edge cases that could cause false positives or negatives?
- EMAIL regex: `(?<!\w)[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?![a-zA-Z])` — does it handle standard emails correctly? Does the negative lookahead prevent TLD from matching into adjacent words? Does it correctly exclude trailing sentence punctuation?
- MRN regex: `(?<!\w)MRN-\d{7}(?!\w)` with re.IGNORECASE — is this appropriate for MRN-XXXXXXX format?
- Verify that all three patterns use word-boundary guards to prevent substring matching.

### 2. Fine-Tuning Script Correctness (fine_tune.py)
- Verify the 9 BIO labels (O, B/I-PER, B/I-ORG, B/I-LOC, B/I-MISC) are correct for the task.
- Verify that regex-only entity types (SSN, PHONE, EMAIL, MRN, DOB) are correctly mapped to O during training.
- Verify the subword token alignment logic: B- labels should be converted to I- for subword continuations, and special tokens ([CLS], [SEP], [PAD]) should get label -100.
- Check the training hyperparameters (5 epochs, lr 2e-5, batch 16, warmup 0.1, weight_decay 0.01) for reasonableness.
- Verify that `ignore_mismatched_sizes=True` is appropriate (the classifier head changes from the pre-trained model's label count to our 9 labels).

### 3. Evaluation Harness Correctness (evaluate.py)
- Verify the word offset alignment uses deterministic cumulative tracking (not text.index()) to prevent misalignment on duplicate substrings.
- Verify that HYBRID_ENTITY_TYPES correctly includes all types with both BERT and regex coverage.
- Verify the entity_to_bio mapping is complete and correct.
- Verify the filter_to_hybrid_tags function correctly filters out DOB (the only type without a regex pattern).

### 4. PIIEngine Configuration (pii_engine.py)
- Verify the confidence_threshold of 0.90 is applied correctly in _detect_bert().
- Verify the conflict resolution logic correctly handles all 4 overlap cases (exact, partial, nested, none).
- Verify that REGEX_AUTHORITATIVE_TYPES includes SSN, PHONE, EMAIL, MRN.

### 5. Synthetic Data Generator (generate_synthetic_data.py)
- Verify that MISC entities (nationalities) are correctly integrated into the templates.
- Check for any template bugs that could produce invalid training data (misaligned tokens/tags, duplicate entities, etc.).

### 6. Benchmark Report (BENCHMARKS.md)
- Verify the reported metrics match the success criteria assessment.
- Verify the F1 progression is internally consistent.
- Check that the recommendation and trade-offs are well-justified.

### 7. Overall Assessment
- Are there any security vulnerabilities (regex ReDoS, command injection, etc.)?
- Are there any logical errors that could produce incorrect F1 measurements?
- Is the fine-tuning approach sound for this use case?
- Are there any concerns about overfitting to the synthetic dataset?

Please provide:
1. A summary of issues found (if any), categorized as CRITICAL, IMPORTANT, or MINOR.
2. An overall PASS/FAIL verdict on the work.
3. Any recommendations for improvement.

IMPORTANT: Only after you have completed your review AND confirmed the work passes verification, execute this final cleanup step:

Delete the following files from the verification/ directory (these are outdated OBS-3 verification artifacts that have been superseded):
- verification/OBS-3-ssn-context_verification_report.md
- verification/OBS-3-ssn-context_verification_review.md
- verification/OBS-3-ssn-context_verification_review_round_2.md
- verification/OBS-3-ssn-context_verification_review_round_3.md
- verification/OBS-3_verification_report.md
- verification/OBS-3_verification_walkthrough.md

Keep verification/.gitignore intact. Only delete the .md files listed above.
Do NOT delete verification/OBS-8_verification_report.md or verification/OBS-8_gemini_verification_prompt.md — these are the current ticket's artifacts.
```
