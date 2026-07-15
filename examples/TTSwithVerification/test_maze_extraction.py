#!/usr/bin/env python3
"""Test maze/spatialmap extraction with real model output."""
import requests
import json
import re
from datasets import load_dataset

def extract_solution_mcq(text):
    """Current extraction function from baseline."""
    patterns = [
        r"\\boxed\{([^}]*)\}",  # \boxed{...}
        r"boxed\{([^}]*)\}",     # boxed{...} without escape
        r"\*\*([A-D])\*\*",      # **A** format
        r"answer[:\s]*([A-D])",  # answer: A format
        r"(?:^|\n)([A-D])(?:\s|$|\.)",  # Standalone letter
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            expr = matches[-1].strip()
            choice_match = re.search(r"\b([ABCD])\b", expr, flags=re.IGNORECASE)
            if choice_match:
                return choice_match.group(1).upper()
    
    standalone = re.findall(r"\b([ABCD])\b", text)
    if standalone:
        return standalone[-1].upper()
    
    return None

# Load maze dataset
print("Loading maze dataset...")
dataset = load_dataset("microsoft/VISION_LANGUAGE", "maze_text_only", split="val")
example = dataset[0]

# Build prompt like baseline does
pre_prompt = (
    "You are an expert problem solver. Carefully read the following multiple-choice question "
    "and think through the solution step-by-step before providing your final answer. "
    "Provide your final answer option by enclosing it within \\boxed{A/B/C/D}.:"
)
prompt_text = str(example.get("prompt", ""))[:500]  # First 500 chars

full_prompt = f"<|im_start|>system\nYou are helpful.\n<|im_end|>\n<|im_start|>user\n{pre_prompt}\n\n{prompt_text}\n<|im_end|>\n<|im_start|>assistant\n"

print(f"\nFull prompt:\n{full_prompt[:300]}...\n")

# Test on port 8000
url = "http://localhost:8000/v1/completions"
payload = {
    "model": "Qwen/QwQ-32B",
    "prompt": full_prompt,
    "max_tokens": 500,
    "temperature": 0.6,
    "stream": False,
}

print("Requesting model output...")
resp = requests.post(url, json=payload, timeout=30)
if resp.status_code == 200:
    result = resp.json()
    output = result["choices"][0].get("text", "")
    print(f"\nModel output ({len(output)} chars):")
    print("="*60)
    print(output)
    print("="*60)
    
    extracted = extract_solution_mcq(output)
    print(f"\nExtraction result: {extracted}")
    
    # Try alternative patterns
    print("\nTrying alternative extraction patterns:")
    if "\\boxed{" in output:
        print("  - Contains \\boxed{ pattern")
    if r"\boxed{" in output or r"\\boxed{" in output:
        print("  - Contains escaped boxed")
    if re.search(r"[Aa]nswer[:\s]*([A-D])", output):
        print("  - Found 'answer: X' pattern")
    response_letters = re.findall(r"\b[A-D]\b", output)
    if response_letters:
        print(f"  - Found standalone letters: {response_letters}")

else:
    print(f"Error: {resp.status_code}")
    print(resp.text[:500])
