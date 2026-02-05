import asyncio
import logging
from transformers import AutoTokenizer
from interwhen.monitors import SimpleTextReplaceMonitor
from interwhen import stream_completion

logger = logging.getLogger(__name__)


def init_llm_server(modelname, max_tokens=200, port=8000):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": modelname,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": True,
        "use_beam_search": False,
        "prompt_cache": True
    }

    headers = {"Content-Type": "application/json"}
    return {'url': url,
            'payload': payload,
            'headers': headers
            }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Override any existing configuration
    )
    model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    llm_server = init_llm_server(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # prepare the model input
    prompt = "Explain quantum computing in simple terms."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    result = asyncio.run(stream_completion(
        text,
        llm_server=llm_server,
        monitors=(SimpleTextReplaceMonitor("IsCheck", "</think>", async_execution=True),),
        add_delay=False,
        termination_requires_validation=False,
        async_execution=True
    ))
    
    # Save output to file
    output_file = "../output.txt"
    with open(output_file, "w") as f:
        f.write(result)
    print(f"\n\nOutput saved to {output_file}")
