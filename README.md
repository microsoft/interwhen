# interwhen: Verifiable Reasoning

interwhen is a Python library that enables interjecting the output tokens of any language model during inference. Instead of passively waiting for the model to finish generating, interwhen analyzes intermediate reasoning steps and provides corrective feedback from external verifiers to guide the model’s reasoning to be more accurate and efficient. 

interwhen changes the inference pipeline of a language model by creating an auxiliary Monitor model that runs alongside the model and interacts with the model’s output to improve its quality. The Monitor agent reads the output of a language model in real time and calls necessary verifiers to check its validity. Based on the objectivity of a domain, verifiers can be symbolic, neuro-symbolic or even fully neural verifiers.

A detailed discussion of interwhen, including how it was developed and tested, can be found in our paper at: [link]().


## Installation

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

## Getting Started
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

## Examples

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

## Intended Uses
- interwhen was developed to improve the quality of a reasoning model’s outputs without requiring finetuning.
- interwhen is best suited for tasks where verification is feasible, such as math, code reasoning, or structured document generation—not highly subjective tasks like creative writing or open-ended opinion pieces where correctness cannot be formally defined. 
- interwhen is being shared with the research community to facilitate reproduction of our results and foster further research in this area.
- interwhen is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them.

## Out-of-scope Uses
- interwhen is not well suited for subjective tasks where answer verification is harder (or more complex).
We do not recommend using interwhen in commercial or real-world applications without further testing and development. It is being released for research purposes.
- interwhen was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use.
- Without further testing and development, interwhen should not be used in sensitive domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities.
- We do not recommend using interwhen in the context of high-risk decision making (e.g. law enforcement, legal, finance, or healthcare).

## Evaluation
interwhen was evaluated on its ability to improve the reasoning quality of existing language models on benchmarks spanning planning, math, logic, and deep research.
A detailed discussion of our evaluation methods and results can be found in our paper at: [link][BB2.1]

### Evaluation Methods
We used accuracy and efficiency metrics to measure interwhen’s performance.
We compared the performance of interwhen against baseline methods such as tree-of-thought and tool calling using benchmarks such as Maze, SpatialEval, Game of 24, Verina and others.
The target model (the model whose reasoning performance was improved) used in our experiments varied by task and included models from Qwen2, Qwen3 and Phi-4 series.
In our experiments, we used models from Qwen2 and Qwen3 series as the auxiliary monitor model. Results may vary if interwhen is used with a different monitor model, based on its unique design, configuration and training.

### Evaluation Results
At a high level, we found that interwhen allows a plug-and-play solution for improving the accuracy (and/or) efficiency of language models. The accuracy improvement on various benchmarks is shown below. Depending on the goal, interwhen can improve the accuracy of a language model given a compute budget or improve the efficiency of the model at a given accuracy.

## Limitations
- interwhen was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios.
- interwhen supports a human-feedback based monitor, however, such a monitor may not be feasible in situations where latency of the model’s output is a key consideration.
- interwhen was designed and tested using the English language. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native speaker of that language.
- Outputs generated by AI may include factual errors, fabrication, or speculation. Users are responsible for assessing the accuracy of generated content. All decisions leveraging outputs of the system should be made with human oversight and not be based solely on system outputs.
- interwhen inherits any biases, errors, or omissions produced by the auxiliary monitor model, as chosen by the developer. Developers are advised to choose appropriate target and auxiliary LLMs carefully, depending on the intended use case.
- interwhen is a framework which can run with any language model preferred by a user. Users can specify the language model whose reasoning they want to improve (“target” model) and an auxiliary model that monitors the target model’s reasoning trace.
- interwhen inherits any biases, errors, or omissions characteristic of the training data of the language models used, which may be amplified by any AI-generated interpretations.
- There has not been a systematic effort to ensure that systems using interwhen are protected from security vulnerabilities such as indirect prompt injection attacks. Any systems using it should take proactive measures to harden their systems as appropriate.

## Best Practices
Better performance can be achieved by assessing the utility of included verifiers for your task and activating only the necessary ones.
We strongly encourage users to use LLMs/MLLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services. Such services continually update their safety and RAI mitigations with the latest industry standards for responsible use. For more on AOAI’s best practices when employing foundations models for scripts and applications:

- [What is Azure AI Content Safety?] (https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview)
- [Overview of Responsible AI practices for Azure OpenAI models] (https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview)
- [Azure OpenAI Transparency Note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note)
- [OpenAI’s Usage policies] (https://openai.com/policies/usage-policies)
- [Azure OpenAI’s Code of Conduct] (https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct)

## License
MIT License

Nothing disclosed here, including the Out of Scope Uses section, should be interpreted as or deemed a restriction or modification to the license the code is released under.

## Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## Contact
This research was conducted by members of Microsoft Research.  We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact us by posting an issue on Github or at interwhen@microsoft.com.

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.


