"""
Base Monitor protocol and related data structures.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union, List
from pydantic import BaseModel


class Monitor(ABC):
    """
    Abstract base class for all monitors.
    
    A Monitor analyzes partial text output from a reasoning model and suggests
    improvements or corrections based on its specific domain of expertise.
    """
    
    def __init__(self, name: str, priority: int = 0):
        """
        Initialize a monitor.
        
        Args:
            name: Unique identifier for this monitor
            priority: Priority level (higher numbers = higher priority)
        """
        self.name = name
        self.priority = priority
    
    @abstractmethod
    async def evaluate(self, chunk: str, token_index: int, generated_text=None, context: Optional[dict] = None):
        """
        Evaluate the given text and return a result with suggested actions.
        
        Args:
            chunk: The partial text output to evaluate
            token_index: The index of the token in the generated text
            generated_text: The full generated text so far
            context: Additional context (e.g., original prompt, conversation history)
            
        Returns:
            some object (TBD)
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, priority={self.priority})"
    
    def __repr__(self) -> str:
        return self.__str__()


class VerifyMonitor(ABC):
    """
    Abstract base class for monitors that use verify/fix pattern.
    
    A VerifyMonitor analyzes partial text output from a reasoning model and suggests
    improvements or corrections based on its specific domain of expertise.
    """
    
    def __init__(self, name: str, priority: int = 0):
        """
        Initialize a monitor.
        
        Args:
            name: Unique identifier for this monitor
            priority: Priority level (higher numbers = higher priority)
        """
        self.name = name
        self.priority = priority
    
    @abstractmethod
    async def verify(self, chunk, token_index, event, event_info):
        """
        Verify the generated text and signal if correction is needed.
        
        Args:
            chunk: The partial text output to verify
            token_index: The index of the current token
            event: asyncio.Event to signal when correction is needed
            event_info: Dict to store correction info
        """
        pass
    
    @abstractmethod
    async def fix(self, generated_text, event_info, fix_method = None):
        """
        Fix the generated text based on event_info.
        
        Args:
            generated_text: The full generated text so far
            event_info: Dict containing correction info
            fix_method: Optional method to use for fixing
            
        Returns:
            The corrected text
        """
        pass
    
    @abstractmethod
    def step_extractor(self, chunk, generated_text):
        """
        Extract steps from generated text.
        
        Args:
            chunk: The partial text output
            generated_text: The full generated text so far
            
        Returns:
            Boolean indicating if more steps should be generated
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, priority={self.priority})"
    
    def __repr__(self) -> str:
        return self.__str__()

