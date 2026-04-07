#!/usr/bin/env python3
"""
Gemini Client for Claude Code - Google Gemini API Integration
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import dataclass
import logging

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

class GeminiClient:
    """Client for interacting with Google Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash-preview-05-20", timeout: int = 300):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter is required")
        
        self.model = model
        self.timeout = timeout
        
        try:
            import google.generativeai as genai
            self.genai = genai
            self.genai.configure(api_key=self.api_key)
            self._client = self.genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def chat(self, user_message: str, tools: Optional[List[Dict]] = None, 
                   system_prompt: Optional[str] = None, stream: bool = False) -> LLMResponse:
        """Send a chat message to Gemini"""
        try:
            # Combine system prompt and user message
            full_message = user_message
            if system_prompt:
                full_message = f"{system_prompt}\n\n{user_message}"
            
            # Generate response
            if stream:
                # For streaming, we'll use the stream_chat method
                response_content = ""
                async for chunk in self.stream_chat(user_message, tools, system_prompt):
                    response_content += chunk
                
                return LLMResponse(
                    message=Message(role="assistant", content=response_content),
                    model=self.model
                )
            else:
                response = await asyncio.to_thread(
                    self._client.generate_content, 
                    full_message
                )
                
                # Safely extract text from response
                response_text = ""
                try:
                    if response.text:
                        response_text = response.text
                    elif response.candidates and response.candidates[0].content.parts:
                        text_parts = []
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                        response_text = "".join(text_parts)
                except Exception:
                    # Keep empty string as fallback
                    pass
                
                return LLMResponse(
                    message=Message(
                        role="assistant",
                        content=response_text
                    ),
                    model=self.model,
                    usage={"prompt_tokens": 0, "completion_tokens": 0}  # Gemini doesn't provide detailed usage
                )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    async def stream_chat(self, user_message: str, tools: Optional[List[Dict]] = None,
                         system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream a chat response from Gemini"""
        try:
            # Combine system prompt and user message
            full_message = user_message
            if system_prompt:
                full_message = f"{system_prompt}\n\n{user_message}"
            
            # Generate streaming response
            response = await asyncio.to_thread(
                self._client.generate_content,
                full_message,
                stream=True
            )
            
            for chunk in response:
                # Safely extract text from streaming chunks
                try:
                    if chunk.text:
                        yield chunk.text
                    elif chunk.candidates and chunk.candidates[0].content.parts:
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                yield part.text
                except Exception:
                    # Skip problematic chunks
                    continue
                    
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise
    
    async def generate_with_tools(self, user_message: str, tools: List[Dict],
                                 system_prompt: Optional[str] = None,
                                 conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate response with tool calling support"""
        try:
            # Use conversation_history if provided, otherwise parse from user_message
            conversation = []
            if conversation_history:
                conversation = conversation_history[:]
                # Add current user message to conversation
                try:
                    import json
                    parsed_message = json.loads(user_message)
                    if isinstance(parsed_message, list):
                        conversation.extend(parsed_message)
                    else:
                        conversation.append({"role": "user", "content": user_message})
                except (json.JSONDecodeError, TypeError):
                    conversation.append({"role": "user", "content": user_message})
            else:
                # Parse the conversation from user_message if it's JSON
                try:
                    import json
                    parsed_conversation = json.loads(user_message)
                    if isinstance(parsed_conversation, list):
                        conversation = parsed_conversation
                    else:
                        conversation = [{"role": "user", "content": user_message}]
                except (json.JSONDecodeError, TypeError):
                    conversation = [{"role": "user", "content": user_message}]
            
            # Convert tools to Gemini format
            gemini_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func_def = tool["function"]
                    gemini_func = self.genai.protos.FunctionDeclaration(
                        name=func_def["name"],
                        description=func_def["description"],
                        parameters=self.genai.protos.Schema(
                            type=self.genai.protos.Type.OBJECT,
                            properties={
                                name: self._convert_schema_property(prop)
                                for name, prop in func_def["parameters"].get("properties", {}).items()
                            },
                            required=func_def["parameters"].get("required", [])
                        )
                    )
                    gemini_tools.append(gemini_func)
            
            # Get the last user message
            last_user_message = ""
            for msg in reversed(conversation):
                if msg.get("role") == "user":
                    last_user_message = msg.get("content", "")
                    break
            
            # Add system prompt if provided
            if system_prompt:
                last_user_message = f"{system_prompt}\n\n{last_user_message}"
            
            # Create tool config if tools are provided
            tool_config = None
            if gemini_tools:
                tool_config = self.genai.protos.Tool(function_declarations=gemini_tools)
            
            # Generate response
            response = await asyncio.to_thread(
                self._client.generate_content,
                last_user_message,
                tools=[tool_config] if tool_config else None
            )
            
            # Check for function calls
            tool_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        tool_calls.append({
                            "function": {
                                "name": part.function_call.name,
                                "arguments": dict(part.function_call.args)
                            }
                        })
            
            # Handle text extraction more safely
            response_text = ""
            if response.text:
                response_text = response.text
            elif not tool_calls:
                # If no function calls and no text, check parts directly
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
            
            return {
                "message": {
                    "role": "assistant",
                    "content": response_text,
                    "tool_calls": tool_calls
                },
                "model": self.model,
                "done": True
            }
            
        except Exception as e:
            logger.error(f"Gemini tools error: {e}")
            # Fallback to basic response without tools
            try:
                last_user_message = user_message
                if system_prompt:
                    last_user_message = f"{system_prompt}\n\n{user_message}"
                
                response = await asyncio.to_thread(
                    self._client.generate_content,
                    last_user_message
                )
                
                # Safely extract text from fallback response
                fallback_text = "I encountered an error processing your request."
                try:
                    if response.text:
                        fallback_text = response.text
                    elif response.candidates and response.candidates[0].content.parts:
                        text_parts = []
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                        if text_parts:
                            fallback_text = "".join(text_parts)
                except Exception:
                    # Keep default fallback text
                    pass
                
                return {
                    "message": {
                        "role": "assistant",
                        "content": fallback_text
                    },
                    "model": self.model,
                    "done": True
                }
            except Exception as fallback_error:
                logger.error(f"Gemini fallback error: {fallback_error}")
                return {
                    "message": {
                        "role": "assistant",
                        "content": "I'm experiencing technical difficulties. Please try again."
                    },
                    "model": self.model,
                    "done": True
                }
    
    async def list_models(self) -> List[str]:
        """List available Gemini models"""
        try:
            models = await asyncio.to_thread(self.genai.list_models)
            return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
        except Exception as e:
            logger.error(f"Error listing Gemini models: {e}")
            return [self.model]
    
    def _convert_type(self, json_type: str):
        """Convert JSON schema type to Gemini protobuf type"""
        type_mapping = {
            "string": self.genai.protos.Type.STRING,
            "number": self.genai.protos.Type.NUMBER,
            "integer": self.genai.protos.Type.INTEGER,
            "boolean": self.genai.protos.Type.BOOLEAN,
            "array": self.genai.protos.Type.ARRAY,
            "object": self.genai.protos.Type.OBJECT
        }
        return type_mapping.get(json_type, self.genai.protos.Type.STRING)
    
    def _convert_schema_property(self, prop: Dict[str, Any]):
        """Convert a JSON schema property to Gemini protobuf Schema"""
        prop_type = prop.get("type", "string")
        description = prop.get("description", "")
        
        # Handle simple types
        if prop_type != "array":
            return self.genai.protos.Schema(
                type=self._convert_type(prop_type),
                description=description
            )
        
        # Handle array types - need to specify items
        items_schema = prop.get("items", {})
        items_type = items_schema.get("type", "string")
        
        if items_type == "object":
            # For object arrays, we need to define the object structure
            items_properties = items_schema.get("properties", {})
            items_required = items_schema.get("required", [])
            
            return self.genai.protos.Schema(
                type=self.genai.protos.Type.ARRAY,
                description=description,
                items=self.genai.protos.Schema(
                    type=self.genai.protos.Type.OBJECT,
                    properties={
                        name: self._convert_schema_property(sub_prop)
                        for name, sub_prop in items_properties.items()
                    },
                    required=items_required
                )
            )
        else:
            # For simple type arrays (string, number, etc.)
            return self.genai.protos.Schema(
                type=self.genai.protos.Type.ARRAY,
                description=description,
                items=self.genai.protos.Schema(
                    type=self._convert_type(items_type),
                    description=items_schema.get("description", "")
                )
            )
    
    async def health_check(self) -> bool:
        """Check if Gemini API is accessible"""
        try:
            response = await asyncio.to_thread(
                self._client.generate_content,
                "Hello"
            )
            # Check if we got any response content
            if response.text:
                return True
            elif response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        return True
            return False
        except Exception:
            return False