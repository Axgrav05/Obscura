try:
    from ml.web_demo.app import app
except ModuleNotFoundError:
    from ml.app import app

text = """**Episode Title:** The Evolution of Streaming Media

**Guest:** Mario Vizcay

**Company:** VistaStream Media

**Transcript URL:** https://npr.org/podcasts/510312/throughline#transcript

**Birth Day:** 09/09/1752

**Timestamp:** 18:23:45.123

**Host:** Today we have Mario Vizcay, who manages VistaStream Media. Welcome, Mario!

**Mario Vizcay:** Thank you for having me. It's great to be here.

**Host:** Let's dive into the world of streaming media. Can you tell us more about your role at VistaStream Media?

**Mario Vizcay:** Sure. As a person who manages at VistaStream Media, I oversee the day-to-day operations and ensure that our content delivery is seamless. We strive to provide the best streaming experience for our users.

**Host:** That sounds like a challenging but rewarding role. How do you handle the technical aspects of streaming? Including SSN like 231-22-3123. Or an IP address of 42.10.41.31. Maybe i'm located in Dublin, Ireland. Lets throw in someone named Ron.

**Mario Vizcay:** Well, it involves a lot of coordination and planning. We have to make sure that our health plan beneficiary system is always updated and secure. For instance, our PIN 164091 is crucial for accessing certain administrative tools.

Mario, you work at State Farm Insurance correct? What is it like to work there?"""

with app.test_client() as client:
    response = client.post('/api/redact', json={'text': text})
    data = response.get_json()
    print('status=', response.status_code)
    print('state_farm_entities=', [entity for entity in data.get('entities', []) if 'State Farm Insurance' in entity.get('text', '')])
    print('org_entities=', [entity for entity in data.get('entities', []) if entity.get('type') == 'ORG'])
