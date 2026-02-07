import math
import httpx
import json

def extract_max_prob(token_json):
    top_logprobs = token_json["choices"][0]["logprobs"]["top_logprobs"][0]
    max_logprob = max(top_logprobs.values())
    return math.exp(max_logprob)

class GeometricMeanTracker:
    def __init__(self):
        self.sum_log = 0.0
        self.count = 0

    def add(self, prob):
        """prob is normal probability (0–1)."""
        if prob <= 0:
            return
        self.sum_log += math.log(prob)
        self.count += 1

    def get(self):
        """returns geometric mean"""
        if self.count == 0:
            return 0.0
        return math.exp(self.sum_log / self.count)

def stream_and_compute_geom_mean(llm_server):
    gm = GeometricMeanTracker()
    generated_text = []

    with httpx.Client(timeout=None) as client:
        with client.stream("POST", llm_server["url"],
                           headers=llm_server["headers"],
                           json=llm_server["payload"]) as response:

            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: "):].strip()
                    if data == "[DONE]":
                        break
                    token = json.loads(data)
                    # print(token["choices"][0]["text"], end="", flush=True)
                    p_max = extract_max_prob(token)
                    gm.add(p_max)
                if token["choices"][0]["text"] in ["."]:
                    break

    return gm.get()

def just_stream(llm_server):
    generated_text = []

    with httpx.Client(timeout=None) as client:
        with client.stream("POST", llm_server["url"],
                           headers=llm_server["headers"],
                           json=llm_server["payload"]) as response:

            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: "):].strip()
                    if data == "[DONE]":
                        break
                    token = json.loads(data)
                    generated_text.append(token["choices"][0]["text"])
                    # print(token["choices"][0]["text"], end="", flush=True)

    return "".join(generated_text   )