#!/usr/bin/env python3
"""Test vLLM API responses directly."""
import requests
import json

# Test each port
for port in [8000, 8001, 8002, 8003]:
    print(f"\n{'='*60}")
    print(f"Testing port {port}...")
    print('='*60)
    
    url = f"http://localhost:{port}/v1/completions"
    
    # Test: simple completion
    payload = {
        "model": "Qwen/QwQ-32B",
        "prompt": "What is 2+2? Answer:",
        "max_tokens": 50,
        "temperature": 0.0,
        "stream": False,
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"Response keys: {result.keys()}")
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                text = choice.get("text", "")
                print(f"Generated text: {repr(text[:100])}")
                print(f"Text length: {len(text)}")
            else:
                print(f"No choices in response: {result}")
        else:
            print(f"Error response: {resp.text[:200]}")
    except Exception as e:
        print(f"Request failed: {e}")
