from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

SYSTEM_PROMPT: str = (
    "You are a synthetic code dataset generator for Named Entity Recognition training. "
    "Generate a short code snippet in the specified language that contains identifiers "
    "a company would want to redact (function names, class names, variable names, "
    "hardcoded secrets). Output ONLY a valid JSON object with exactly two keys:\n"
    '  "tokens": list of string tokens (split on whitespace and punctuation)\n'
    '  "ner_tag_labels": parallel list of BIO tags using the schema: '
    "O, B-VAR, I-VAR, B-FUNC, I-FUNC, B-CLASS, I-CLASS, B-SECRET, I-SECRET\n"
    "Ensure len(tokens) == len(ner_tag_labels). At least one label must be non-O."
)

USER_PROMPT_TEMPLATE: str = (
    "Generate a {language} code snippet for a {company_domain} application. "
    "Include at least one proprietary identifier"
    " (function, class, variable, or secret). "
    "Output JSON only — no markdown, no explanation."
)

COMPANY_DOMAINS: list[str] = [
    "fintech",
    "healthcare",
    "e-commerce",
    "logistics",
    "enterprise SaaS",
    "cybersecurity",
    "real estate",
    "insurance",
    "manufacturing",
    "analytics",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic code dataset for CodeBERT fine-tuning"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml/data/code_synthetic.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--count", type=int, default=5000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["python", "javascript", "typescript", "java", "cpp"],
        help="Languages to generate",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        anthropic_key: str | None = os.environ["ANTHROPIC_API_KEY"]
    except KeyError:
        anthropic_key = None
    try:
        openai_key: str | None = os.environ["OPENAI_API_KEY"]
    except KeyError:
        openai_key = None

    if anthropic_key is None and openai_key is None:
        raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY before running.")

    print(f"Generating {args.count} samples across {args.languages}...")
    print(f"Output: {args.output}")

    generated = 0
    with open(args.output, "a") as out_f:
        while generated < args.count:
            language = random.choice(args.languages)
            domain = random.choice(COMPANY_DOMAINS)
            user_msg = USER_PROMPT_TEMPLATE.format(
                language=language, company_domain=domain
            )
            try:
                if anthropic_key:
                    import anthropic as _anthropic

                    client = _anthropic.Anthropic(api_key=anthropic_key)
                    resp = client.messages.create(
                        model="claude-opus-4-6",
                        max_tokens=512,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": user_msg}],
                    )
                    raw = resp.content[0].text.strip()
                else:
                    import openai as _openai

                    client = _openai.OpenAI(api_key=openai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                    )
                    raw = resp.choices[0].message.content.strip()

                sample = json.loads(raw)
                tokens = sample.get("tokens", [])
                labels = sample.get("ner_tag_labels", [])
                if len(tokens) != len(labels):
                    continue
                if all(lbl == "O" for lbl in labels):
                    continue
                record = json.dumps({"tokens": tokens, "ner_tag_labels": labels})
                out_f.write(record + "\n")
                generated += 1
                if generated % 10 == 0:
                    out_f.flush()
                if generated % 100 == 0:
                    print(f"Progress: {generated}/{args.count}")
            except (json.JSONDecodeError, KeyError, Exception):
                continue

    print(f"Done: {generated} samples written to {args.output}")


if __name__ == "__main__":
    main()
