import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 10,
  duration: '30s',
};

export default function () {
  const url = 'http://localhost:8080/v1/chat/completions';
  const payload = JSON.stringify({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "user",
        content: "Hi, my name is John Carter. My phone number is 214-555-0198, my email is john.carter@example.com, my SSN is 123-45-6789, and I live at 742 Evergreen Terrace, Springfield. Please summarize my request in 2 sentences."
      }
    ],
    temperature: 0.2
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const res = http.post(url, payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  sleep(1);
}
