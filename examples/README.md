## Verifier-guided Reasoning in Three Lines 
Running verifier-guided inference requires only a few lines of code: just specify the list of monitors to be used with a target LLM. Each monitor requires specifying the kind of verifier, when it should be invoked (e.g., each step or after a reflection token like 'Wait'), and the text pattern to intervene with. 

**Set up target LLM server**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --max-model-len 65536 \
  --port 8000 \
  --tensor-parallel-size 8
```

**Generate answer enabled with given monitors**
```python
llm_server = init_llm_server("Qwen/Qwen3-30B-A3B-Thinking-2507", max_tokens=32768, port=8000)
stream_completion(
    prompt,
    llm_server=llm_server,
    monitors=(SimpleTextReplaceMonitor("IsCheck", "</think>", async_execution=True),),
    async_execution=True
)
```
The above code implements a simple monitor that watches the model's output stream and replaces all occurences of "is" with "isn't". It can be replaced with your custom monitor, e.g., for checking logical correctness or domain-specific constraints.  You can run the full example 
```bash
python ./examples/text_replacement_example.py
```


https://github.com/user-attachments/assets/a90a829d-99a4-4640-9e00-6c9511c64fa1



The table below shows the latency impact of the monitor. When the stream contains the target word ("is"), the monitor activates and performs the replacement, adding some overhead. When the target word is absent, the monitor has negligible impact on latency.

| Stream content | Monitor | Latency (s) |
|----------------|---------|-------------|
| Contains "is" (monitor activates) | enabled | 12.97 ± 2.97 |
| Contains "is" (monitor activates) | disabled | 8.36 ± 0.01 |
| Does not contain "is" (monitor idle) | enabled | 7.31 ± 1.16 |
| Does not contain "is" (monitor idle) | disabled | 7.35 ± 1.17 |
