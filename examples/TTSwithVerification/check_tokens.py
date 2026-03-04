#!/usr/bin/env python3
"""Check what payload is actually being sent."""
import requests
import json

url = "http://localhost:8000/v1/completions"
payload = {
    "model": "Qwen/QwQ-32B",
    "prompt": "What is 2+2?",
    "max_tokens": 50,
    "temperature": 0.0,
    "stream": False,
}

print("Payload being sent:")
print(json.dumps(payload, indent=2))

resp = requests.post(url, json=payload, timeout=10)
result = resp.json()
output = result["choices"][0]["text"]

print(f"\nResponse usage:")
print(f"  Tokens generated: {result['usage'].get('completion_tokens', '?')}")
print(f"  Total tokens: {result['usage'].get('total_tokens', '?')}")

print(f"\nOutput ({len(output)} chars):")
print(repr(output))
