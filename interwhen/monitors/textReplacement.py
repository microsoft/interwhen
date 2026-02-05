import asyncio
import re
from .base import VerifyMonitor
import time

    
class SimpleTextReplaceMonitor(VerifyMonitor):
    def __init__(self, name, answer_start_token, async_execution=True):
        super().__init__(name)
        self.async_execution = async_execution
        self.answer_start_token = answer_start_token

    async def _verify(self, chunk, token_index):
        if re.search(r'\bis\b', chunk):
            corrected_text = re.sub(r'\bis\b', "isn't", chunk)
            return (False, corrected_text, token_index)
        return (True, chunk, token_index)

    async def verify(self, chunk, token_index, event, event_info):
        is_valid, updated_text, correction_index = await self._verify(chunk, token_index)
        if is_valid:
            return chunk, None
        
        if not event.is_set():
            event_info["generated_text"] = chunk
            event_info["feedback"] = updated_text  # Store the corrected text
            event_info["correction_index"] = correction_index
            event.set()
    
    async def fix(self, generated_text, event_info, fix_method = None):
        return generated_text[:event_info["correction_index"]] + event_info["feedback"]
        
    def step_extractor(self, chunk, generated_text):
        if self.answer_start_token in generated_text:
            return False, None
        return True, chunk
