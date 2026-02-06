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

#### K stable answer monitors
```bash
KstableAnswerMCQMonitor(
                name="maze_kstable",
                k=3,
                options=options,  # Validate equations use exactly these numbers
                answer_start_token="</think>"
            )

KstableAnswerGame24Monitor(
                name="game24_kstable",
                k=3,
                expected_nums=nums,  # Validate equations use exactly these numbers
                answer_start_token="</think>"
            )
```

### Run scripts

Simple text replacement
```bash
python ./examples/text_replacement_example.py
```
Various datasets
```bash
python ./examples/maze_example.py -n 1
python ./examples/game24_example.py -n 1
python ./examples/spatialmap_example.py -n 1
```