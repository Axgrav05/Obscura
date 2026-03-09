from __future__ import annotations

import json
import os
import urllib.request


TEXT = """**Episode Title:** The Evolution of Streaming Media

**Guest:** Mario Vizcay

**Company:** VistaStream Media

**Transcript URL:** https://npr.org/podcasts/510312/throughline#transcript

**Birth Day:** 09/09/1752

**Timestamp:** 18:23:45.123

**Host:** Today we have Mario Vizcay, who manages VistaStream Media. Welcome, Mario!

**Mario Vizcay:** Thank you for having me. It's great to be here.

**Host:** Let's dive into the world of streaming media. Can you tell us more about your role at VistaStream Media?

**Mario Vizcay:** Sure. As a person who manages at VistaStream Media, I oversee the day-to-day operations and ensure that our content delivery is seamless. We strive to provide the best streaming experience for our users. Ron is actually Bob but Bob goes by Kyle is Kyle he means Zack and thats because he had a friend called Ron but not Ron because Ron Isn't Ron and Ron works at Microsoft.

**Host:** That sounds like a challenging but rewarding role. How do you handle the technical aspects of streaming? Including SSN like 231-22-3123. Or an IP address of 42.10.41.31. Maybe i'm located in Dublin, Ireland. Lets throw in someone named Ron.

**Mario Vizcay:** Well, it involves a lot of coordination and planning. We have to make sure that our health plan beneficiary system is always updated and secure. For instance, our PIN 164091 is crucial for accessing certain administrative tools.

Mario, you work at State Farm Insurance correct? What is it like to work there?"""


def _print_summary(data: dict) -> None:
    entities = data.get("entities", [])
    print(
        "state_farm=",
        [entity for entity in entities if "State Farm Insurance" in entity.get("text", "")],
    )
    print(
        "microsoft=",
        [entity for entity in entities if entity.get("text") == "Microsoft"],
    )
    print("total=", data.get("total_count"))


def verify_via_test_client() -> None:
    from ml.web_demo.app import app, load_model

    load_model()
    with app.test_client() as client:
        response = client.post("/api/redact", json={"text": TEXT})
        print("status=", response.status_code)
        _print_summary(response.get_json())


def verify_via_http(base_url: str) -> None:
    payload = json.dumps({"text": TEXT}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/redact",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
        print("status=", response.status)
        _print_summary(body)


if __name__ == "__main__":
    base_url = os.environ.get("OBSCURA_VERIFY_URL")
    if base_url:
        verify_via_http(base_url)
    else:
        verify_via_test_client()