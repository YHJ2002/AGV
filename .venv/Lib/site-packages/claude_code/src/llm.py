#!/usr/bin/env python3
"""
LLM Client for Claude Code - Ollama Integration
"""

import json
import asyncio
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import dataclass
import logging

try:
    import ollama
except ImportError:
    ollama = None

logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str
    content: str

@dataclass
class LLMResponse:
    message: Message
    usage: Optional[Dict] = None
    model: Optional[str] = None

class OllamaClient:
    """Client for interacting with Ollama API using official Python library"""
    
    def __init__(self, host: str = "http://192.168.170.76:11434", model: str = "qwen2.5:7b-instruct", timeout: int = 300, num_ctx: int = 4096, enable_thinking: bool = False):
        if ollama is None:
            raise ImportError("Ollama library not installed. Run: pip install ollama>=0.5.1")
        
        self.host = host.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.enable_thinking = enable_thinking
        
        # Create Ollama client with custom host
        self.client = ollama.Client(host=self.host)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed for official Ollama client
        pass
    
    async def chat(self, user_message: str, tools: Optional[List[Dict]] = None, 
                   system_prompt: Optional[str] = None, stream: bool = False,
                   conversation_history: Optional[List[Dict]] = None) -> LLMResponse:
        """Send a chat message to Ollama using official library"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": self.num_ctx,
        }
        
        try:
            # Use official Ollama client with custom host
            response = await asyncio.to_thread(
                self.client.chat,
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                options=options,
                stream=stream,
                think=self.enable_thinking  # Use think parameter for thinking mode
            )
            
            if stream:
                # Handle streaming response
                return response
            else:
                # Extract content from response
                message_content = response.get('message', {}).get('content', '')
                
                # Check if response contains thinking content
                if 'thinking' in response.get('message', {}):
                    thinking_content = response['message'].get('thinking', '')
                    if thinking_content and not message_content:
                        # If thinking content exists but no regular message, use thinking content
                        message_content = thinking_content
                
                return LLMResponse(
                    message=Message(
                        role=response.get('message', {}).get('role', 'assistant'),
                        content=message_content
                    ),
                    model=response.get("model"),
                    usage=response.get("usage")
                )
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise
    
    async def stream_chat(self, user_message: str, tools: Optional[List[Dict]] = None,
                         system_prompt: Optional[str] = None,
                         conversation_history: Optional[List[Dict]] = None) -> AsyncGenerator[str, None]:
        """Stream a chat response from Ollama using official library"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": self.num_ctx,
        }
        
        try:
            # Use official Ollama client for streaming
            stream = await asyncio.to_thread(
                self.client.chat,
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                options=options,
                stream=True,
                think=self.enable_thinking
            )
            
            # Stream the response chunks
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:
                        yield content
                        
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise
    
    async def generate_with_tools(self, user_message: str, tools: List[Dict],
                                 system_prompt: Optional[str] = None,
                                 conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate response with tool calling support using official library"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": self.num_ctx,
        }
        
        try:
            response = await asyncio.to_thread(
                self.client.chat,
                model=self.model,
                messages=messages,
                tools=tools,
                options=options,
                stream=False,
                think=self.enable_thinking
            )
            return response
            
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise
    
    async def list_models(self) -> List[str]:
        """List available models using official client"""
        try:
            models = await asyncio.to_thread(self.client.list)
            return [model["name"] for model in models.get("models", [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if Ollama is healthy using official client"""
        try:
            models = await asyncio.to_thread(self.client.list)
            return True
        except Exception:
            return False