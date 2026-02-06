## interwhen : Test time Verification framework


### Clone repo
```bash
git clone https://github.com/microsoft/interwhen.git
cd interwhen
```
### setup env
```bash
conda env create -f environment.yml
conda activate inter
pip install -e .
```
### Deploy the server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --max-model-len 65536 \
  --port 8000 \
  --tensor-parallel-size 8
  ```

### Monitors

#### Simple text replacement monitor
```bash
(SimpleTextReplaceMonitor("IsCheck", "</think>", async_execution=False),)
```

### Run scripts

Simple text replacement
```bash
python ./examples/text_replacement_example.py
```
```bash
python ./examples/maze_example.py