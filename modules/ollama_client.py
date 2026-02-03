"""
Ollama LLM client wrapper.
Handles chat interactions with Ollama models with proper error handling and logging.
"""

import ollama
from typing import Optional, List, Dict, Any
from .logger import setup_logger

logger = setup_logger(__name__)


class OllamaChat:
    """Wrapper for Ollama chat functionality."""
    
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama chat client.
        
        Args:
            model: Model name to use
            base_url: Ollama server base URL
        """
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        
        logger.info(f"OllamaChat initialized with model: {model}, base_url: {base_url}")
    
    def _check_connection(self) -> bool:
        """Check if Ollama server is running."""
        try:
            # Try to get tags (list models)
            self.client.list()
            logger.debug("Ollama connection verified")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Send a chat message and get a response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt to prepend
            temperature: Sampling temperature (0.0-1.0)
            top_p: Top-p sampling parameter
        
        Returns:
            Response text from the model
        """
        if not self._check_connection():
            logger.error("Ollama server not available")
            raise RuntimeError("Ollama server is not running. Start it with: ollama serve")
        
        # Build final messages with system prompt
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(messages)
        
        logger.debug(f"Sending {len(final_messages)} messages to {self.model}")
        logger.debug(f"Message types: {[m.get('role', 'unknown') for m in final_messages]}")
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=final_messages,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                },
                stream=False
            )
            
            response_text = response.get('message', {}).get('content', '').strip()
            logger.debug(f"Response received: {len(response_text)} chars")
            
            return response_text
        
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise
    
    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Stream chat response token by token.
        
        Args:
            messages: List of message dicts
            system_prompt: Optional system prompt
            temperature: Sampling temperature
        
        Yields:
            Response tokens
        """
        if not self._check_connection():
            logger.error("Ollama server not available")
            raise RuntimeError("Ollama server is not running")
        
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(messages)
        
        logger.debug(f"Streaming response from {self.model}")
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=final_messages,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                },
                stream=True
            )
            
            for chunk in response:
                token = chunk.get('message', {}).get('content', '')
                if token:
                    logger.debug(f"Token: {repr(token)}")
                    yield token
        
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            raise
    
    def set_model(self, model: str):
        """Switch to a different model."""
        logger.info(f"Switching model from {self.model} to {model}")
        self.model = model
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = self.client.list()
            models = [m.get('name', '').split(':')[0] for m in response.get('models', [])]
            logger.debug(f"Available models: {models}")
            return models
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
