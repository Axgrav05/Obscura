from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional

import requests

PROXY_BASE_URL = os.getenv("PROXY_URL", "http://localhost:8080")
PROXY_PATH = os.getenv("PROXY_PATH", "")

try:
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
except ValueError:
    REQUEST_TIMEOUT = 60


def build_payload() -> Dict[str, Any]:
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Hi, my name is John Carter. "
                    "My phone number is 214-555-0198, "
                    "my email is john.carter@example.com, "
                    "my SSN is 123-45-6789, "
                    "and I live at 742 Evergreen Terrace, Springfield. "
                    "Please summarize my request in 2 sentences."
                ),
            }
        ],
        "temperature": 0.2,
    }


def build_proxy_url() -> str:
    base = PROXY_BASE_URL.rstrip("/")
    path = PROXY_PATH.strip()

    if not path:
        return base

    if not path.startswith("/"):
        path = f"/{path}"

    return f"{base}{path}"


def print_header() -> None:
    print("=" * 48)
    print("OBS-16 MIDTERM DEMO — END-TO-END TEST")
    print("=" * 48)


def print_section(title: str, content: str) -> None:
    print(f"\n[{title}]")
    print(content)


def print_pipeline_status(
    redacted_available: bool,
    llm_available: bool,
    restored_available: bool,
) -> None:
    print("\n[PIPELINE STAGES]")
    print(f"Original Input   : {'available'}")
    print(f"Redacted Input   : {'available' if redacted_available else 'not exposed'}")
    print(f"LLM Response     : {'available' if llm_available else 'not exposed'}")
    print(f"Restored Response: {'available' if restored_available else 'not exposed'}")


def safe_json(response: requests.Response) -> Optional[Dict[str, Any]]:
    try:
        data = response.json()
        if isinstance(data, dict):
            return data
        return None
    except ValueError:
        return None


def extract_openai_text(data: Dict[str, Any]) -> Optional[str]:
    try:
        choices = data.get("choices", [])
        if not choices:
            return None

        message = choices[0].get("message", {})
        content = message.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
            if text_parts:
                return "\n".join(text_parts)

        return None
    except (AttributeError, IndexError, TypeError):
        return None


def get_stage(data: Dict[str, Any], *possible_keys: str) -> Optional[str]:
    for key in possible_keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def main() -> None:
    payload = build_payload()
    proxy_url = build_proxy_url()

    print_header()

    original_input = payload["messages"][0]["content"]
    print_section("ORIGINAL INPUT", original_input)
    print_section("TARGET PROXY URL", proxy_url)
    print("\n[REQUEST FLOW]")
    print("Client -> Obscura Proxy -> OpenAI -> Obscura Proxy -> Client")

    try:
        response = requests.post(proxy_url, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        print_section("ERROR", f"Request failed: {exc}")
        print_section(
            "SUMMARY",
            "Demo script is ready, but the proxy is not reachable yet. "
            "Waiting on local proxy startup or deployed EC2 endpoint.",
        )
        return

    print_section("RESPONSE STATUS", str(response.status_code))

    if not response.ok:
        print_section("ERROR RESPONSE BODY", response.text[:1000])
        print_section(
            "SUMMARY",
            "Proxy responded with an HTTP error. Check route/path, proxy status, or deployment config.",
        )
        return

    data = safe_json(response)

    if data is None:
        print_section("RAW RESPONSE BODY", response.text)
        print_section(
            "SUMMARY",
            "Received a non-JSON response from the proxy.",
        )
        return

    redacted_input = get_stage(data, "redacted", "redacted_input", "sanitized_input")
    llm_response = get_stage(data, "llm_response", "raw_llm_response", "model_response")
    restored_response = get_stage(
        data, "restored", "restored_response", "final_response"
    )

    print_pipeline_status(
        redacted_available=redacted_input is not None,
        llm_available=llm_response is not None,
        restored_available=restored_response is not None,
    )

    if redacted_input is not None:
        print_section("REDACTED INPUT", redacted_input)

    if llm_response is not None:
        print_section("LLM RESPONSE", llm_response)

    if restored_response is not None:
        print_section("RESTORED RESPONSE", restored_response)

    extracted_text = extract_openai_text(data)
    if extracted_text is not None and restored_response is None:
        print_section("MODEL OUTPUT", extracted_text)

    print_section("RAW RESPONSE JSON", json.dumps(data, indent=2))

    if redacted_input or llm_response or restored_response:
        print_section(
            "SUMMARY",
            "Demo completed. Proxy exposed one or more internal pipeline stages.",
        )
    else:
        print_section(
            "SUMMARY",
            "Demo completed. Proxy returned a standard response, but did not expose internal pipeline stages.",
        )


if __name__ == "__main__":
    main()
