import asyncio

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import VerifyMonitor
from interwhen.utils.EAT_helper import compute_entropy, exponential_moving_average, exponential_moving_variance
from interwhen.utils.DEER_helper import stream_and_compute_geom_mean


class EATMonitor(VerifyMonitor):
    """
    Entropy After Think (EAT) Monitor.
    
    Uses entropy-based early stopping: when EM variance of entropy drops below delta,
    we stop generation and force an answer
    """
    
    def __init__(self, name, model_name, alpha=0.2, delta=0.0001, 
                 min_steps=4, answer_start_token="</think>", async_execution=True):
        super().__init__(name)
        self.model_name = model_name
        self.alpha = alpha  # smoothing factor for ema
        self.delta = delta  # variance threshold for early stop
        self.min_steps = min_steps  # minimum steps before allowing early stop
        self.answer_start_token = answer_start_token
        self.async_execution = async_execution
        
        # Load model for entropy computation
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # State tracking
        self.entropy = []
        self.ema_means = []
        self.ema_vars = []
        self.exit_point = None

    async def _verify(self, generated_text, token_index):
        """
        Core verification logic using entropy.
        Returns (is_valid, feedback, correction_index)
        """
        
        # We append this tail so that we can compute entropy for next token (answer)
        partial_answer = (generated_text + "\n\n</think>" + "\n\n" + 'Final answer is \\boxed{')
        entropy_2 = compute_entropy(
            self.hf_model,
            self.tokenizer,
            partial_answer,
        )

        self.entropy.append(entropy_2)
        ema_average = exponential_moving_average(self.entropy, self.alpha)
        ema_variance = exponential_moving_variance(self.entropy, self.alpha, 0.0)

        self.ema_means.append(ema_average[-1])
        self.ema_vars.append(ema_variance[-1])
        
        # Early stopping not triggered unless min_steps number of steps have been processed
        if len(self.entropy) < self.min_steps:
            return (True, None, token_index)
        
        # Intervene if variance is below threshold
        if ema_variance[-1] < self.delta:
            self.exit_point = len(self.entropy)
            # Return False to trigger early stop
            return (False, generated_text, token_index)
        
        return (True, None, token_index)

    async def verify(self, step, token_index, event, event_info):
        """
        Verify the step. Sets event if early stop should be triggered.
        """
        is_valid, feedback, correction_index = await self._verify(step, token_index)
        
        if is_valid:
            return step, None
        
        # Early stop triggered
        if not event.is_set():
            event_info["generated_text"] = step
            event_info["feedback"] = feedback
            event_info["correction_index"] = len(step) # so we know where to slice the gen text during fix
            event_info["entropy_history"] = self.entropy.copy()
            event_info["ema_variance"] = self.ema_vars[-1] if self.ema_vars else None
            event.set()

    async def fix(self, generated_text, event_info, fix_method=None):
        """
        Appending the </think> to force the thinking process to conclude.
        """
        fixed_text = generated_text[:event_info['correction_index']] + "\n\n</think>"
        print("VISHAAAAAAAAAAAAAAAK"*100)
        return fixed_text
    
    def step_extractor(self, chunk, generated_text):
        """
        Extract steps to verify. For EAT, we verify each time a '\n\nWait' is found
        Returns the entire generated text up to last '\n\nWait' found".
        From our experiments we found that the wait tokens only appear at the end of a chunk,
        and hence at the end of the latest generated text. 
        """
        if self.answer_start_token in generated_text:
            return False, None
        
        if(generated_text.endswith("\n\nWait")):
            return (True, generated_text[:-len("\n\nWait")])

        return False, None


class DEERMonitor(VerifyMonitor):
    """
    Dynamic Early Exit in Reasoning (DEER) Monitor.

    Uses answer confidence after wait tokens to know when to stop generation and force an answer.
    """
    def __init__(self, name, llm_server, delta=0.995, answer_start_token="</think>", async_execution=True, max_probe_steps=20):
        super().__init__(name)
        self.llm_server = llm_server
        self.delta = delta
        self.async_execution = async_execution
        self.max_probe_steps = max_probe_steps
        self.answer_start_token = answer_start_token
        self.confidence = []

    async def _verify(self, generated_text, token_index):
        """
        Core verification logic using confidence.
        Returns (is_valid, feedback, correction_index)
        """
        
        # We apppend this tail so that we can compute confidence for the answer
        partial_answer = (generated_text + "\n\n</think>" + "\n\n" + 'Final answer is \\boxed{')
        self.llm_server["payload"]["prompt"] = partial_answer
        confidence = stream_and_compute_geom_mean(self.llm_server)
        self.confidence.append(confidence)

        if confidence > self.delta:
            return False, generated_text, token_index

        return (True, None, token_index)

    async def verify(self, step, token_index, event, event_info):
        """
        Verify the step. Sets event if early stop should be triggered.
        """
        is_valid, feedback, correction_index = await self._verify(step, token_index)
        
        if is_valid:
            return step, None
        
        # Early stop triggered
        if not event.is_set():
            event_info["generated_text"] = step
            event_info["feedback"] = feedback
            event_info["correction_index"] = len(step) # so we know where to slice the gen text during fix
            event_info["confidence_history"] = self.confidence.copy()
            event.set()

    async def fix(self, generated_text, event_info, fix_method=None):
        """
        Appending </think> to force the thinking process to conclude.
        """
        # Append answer prompt to conclude
        fixed_text = generated_text[:event_info['correction_index']] + "\n\n</think>"
        return fixed_text 

    def step_extractor(self, chunk, generated_text):
        """
        Extract steps to verify. For Deer, we verify after each \n\nwait
        """

        if self.answer_start_token in generated_text:
            return False, None
        
        if(generated_text.endswith("\n\nWait")):
            return (True, generated_text[:-len("\n\nWait")])

        return False, None  