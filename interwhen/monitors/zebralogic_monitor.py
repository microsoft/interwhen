import re
import json
import logging
from openai import OpenAI
from transformers import PreTrainedTokenizerBase
from .base import VerifyMonitor
from ..utils.zebralogic_helper import SYSTEM_PROMPT_STATEEXTRACT, USER_PROMPT_TEMPLATE
from ..utils.zebralogic_verifier import ZebraLogicProblem

logger = logging.getLogger(__name__)

class ZebraLogicMonitor(VerifyMonitor):
    """
    ZebraLogic forking monitor.

    Args:
        name: Monitor name.
        instance: Processed ZebraLogic problem dict (from get_zebralogic_dataset).
        llm: Model name for the LLM server.
        tokenizer: HuggingFace tokenizer (must support apply_chat_template).
        step_interval: Verify every N occurrences of the step token.
        step_token: Token string that delimits steps.
        port: vLLM server port.
        max_corrections: Maximum number of feedback injections before stopping.
    """

    def __init__(
        self,
        name: str,
        instance: dict,
        llm: str,
        tokenizer: PreTrainedTokenizerBase,
        step_token: str,
        step_interval: int,
        port: int = 8000,
        max_corrections: int = 50,
        async_execution: bool = True,
        priority: int = 0,
    ):
        super().__init__(name)
        self.instance = instance
        self.llm = llm
        self.tokenizer = tokenizer
        self.step_token = step_token
        self.step_interval = step_interval
        self.max_corrections = max_corrections
        self.async_execution = async_execution
        self.priority = priority

        self.client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key='EMPTY'
        )

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the prompt used for extracting the current house assignments from the LLM."""
        problem_text = self.instance['puzzle_clean']
        system_prompt = SYSTEM_PROMPT_STATEEXTRACT
        user_prompt = USER_PROMPT_TEMPLATE.format(problem_text=problem_text)

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False, add_generation_prompt=True
        )
        return prompt

    def _get_z3_solver(self):
        """Create a fresh Z3 solver for this problem instance."""
        return ZebraLogicProblem(self.instance)

    def _llm_call(self, input_text: str):
        """Make an LLM completion call to extract for current state."""
        response = self.client.completions.create(
            model=self.llm,
            prompt=input_text,
            temperature=0.3,
            max_tokens=2000,
            stop='```'
        )
        text = response.choices[0].text + '```'
        usage = response.usage.to_dict() if response.usage else {}
        return text, usage
    
    def _extract_house_assignments(self, chunk: str) -> dict:
        """Extract the last JSON block of house assignments from text."""
        pattern = re.compile(r'```json(.*?)```', re.DOTALL)
        matches = re.findall(pattern, chunk)
        if not matches:
            raise ValueError("No JSON block found in the chunk.")
        return json.loads(matches[-1].strip())

    def _is_valid_assignment(self, house_no: int, feature: str, value: str) -> bool:
        """Check if a single (house, feature, value) assignment is consistent with all clues."""
        z3_solver = self._get_z3_solver()
        ir = {
            'type': 'place',
            'entity': (feature, value),
            'pos': house_no
        }
        z3_solver.apply_ir([[ir]])
        return z3_solver.is_satisfiable

    def _get_contradictions(self, house_asgns: dict):
        """Find assignments that contradict the problem constraints.

        Returns:
            (contradictions, errors) where:
                contradictions: list of (house_no, feature, value) tuples
                errors: list of (house_no, feature, value, error_msg) tuples
        """
        contradictions = []
        errors = []

        for house, asgns in house_asgns.items():
            house_no = int(re.search(r'House (\d+)', house).group(1))
            for feature, value in asgns.items():
                try:
                    if not self._is_valid_assignment(house_no, feature, value):
                        contradictions.append((house_no, feature, value))
                except Exception as e:
                    errors.append((house_no, feature, value, str(e)))

        return contradictions, errors
    
    def _count_feedback_blocks(self, chunk: str) -> int:
        """Count how many times feedback has been injected into the generation."""
        return len(re.findall(r'\[FEEDBACK\]', chunk))
    
    def step_extractor(self, chunk: str, generated_text: str):
        """Trigger verification at regular step_token intervals.

        Stops every ``step_interval`` occurrences of ``step_token`` in the generated text, within a think block.

        Returns:
            (trigger_flag, text_to_verify) - text_to_verify is the full generated text.
        """
        last_open_think = generated_text.rfind('<think>')
        last_close_think = generated_text.rfind('</think>')
        if last_close_think != -1 and last_close_think > last_open_think: # we are outside of a think block
            return False, None
        
        return generated_text.endswith(self.step_token) and generated_text.count(self.step_token) % self.step_interval == 0, generated_text

    async def verify(self, chunk: str, token_index: int, event, event_info: dict):
        """Extract the current state from the LLM and verify assignments against Z3.

        Appends a state-extract prompt suffix to the current generation, makes an LLM
        call to produce a JSON state snapshot, then validates each assignment.

        Args:
            chunk: The generated text so far.
            token_index: Current token index.
            event: asyncio.Event to signal when correction is needed.
            event_info: Dict to store correction info.
        """
        
        # check with max corrections limit
        num_corrections = self._count_feedback_blocks(chunk)
        if num_corrections >= self.max_corrections:
            max_feedback = "\nthe answer is \\boxed{no solution}"
            if not event.is_set():
                event_info["generated_text"] = chunk
                event_info["feedback"] = max_feedback
                event_info["correction_index"] = token_index
                event.set()
            return

        # Append state extraction suffix to elicit state JSON from the LLM
        suffix = "\nOk let me note down the current partial assignments that I'm sure of, for reference.</think>\n```json\n{\n\"House "
        input = self.system_prompt + chunk + suffix
        
        # fail silently if LLM call or JSON parsing fails, to avoid blocking the main generation loop
        try:
            text, usage = self._llm_call(input)
        except Exception as e:
            logger.warning("LLM state-extract call failed at token_index=%d: %s", token_index, e)
            return

        output = chunk + suffix + text
        
        try:
            house_asgns = self._extract_house_assignments(output)
        except Exception as e:
            logger.warning("Failed to extract house assignments at token_index=%d (usage=%s): %s", token_index, usage, e)
            return

        try:
            contradictions, errors = self._get_contradictions(house_asgns)
        except Exception as e:
            logger.warning("Failed to get contradictions at token_index=%d (usage=%s): %s", token_index, usage, e)
            return

        logger.info(
            "ZebraLogic evaluate at token_index=%d (usage=%s): contradictions=%s, errors=%s",
            token_index, usage, contradictions, errors
        )

        if not contradictions:
            return # No issues found

        # Build feedback string
        feedback = ''
        if contradictions:
            feedback += "The following assignments contradict with the problem's clues:\n"
            for house_no, feature, value in contradictions:
                feedback += f"- House {house_no}: {feature} = {value}\n"
        feedback += '\nUse this feedback and continue solving.'
        feedback = f"\n\n[FEEDBACK]\n{feedback}\n[/FEEDBACK]\n\n"
        
        if not event.is_set():
            event_info["generated_text"] = chunk
            event_info["feedback"] = feedback
            event_info["correction_index"] = token_index
            event.set()

        return chunk, feedback

    async def fix(self, generated_text: str, event_info: dict, fix_method=None) -> str:
        """Inject feedback into the generation to steer the solver.
        """
        return event_info["generated_text"] + event_info["feedback"]
