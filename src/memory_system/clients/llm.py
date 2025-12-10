"""LLM client with provider-agnostic fallback support."""

import json
import time
import logging
from typing import Dict, Any, Optional

from openai import OpenAI, AsyncOpenAI

from ..exceptions import LLMCallError
from langfuse import observe, get_client

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM model client with configurable primary and fallback providers."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        fallback_api_key: Optional[str] = None,
        fallback_base_url: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        """Initialize LLM client.
        
        Args:
            api_key: API key (default: DeepSeek)
            base_url: Base URL for API (default: DeepSeek)
            model: Model ID for LLM (default: deepseek-chat)
            fallback_api_key: Optional API key for fallback provider
            fallback_base_url: Optional fallback base URL
            fallback_model: Optional fallback model ID
        """
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        
        self._fallback_client = None
        self._async_fallback_client = None
        self._fallback_model = None
        if fallback_api_key and fallback_base_url and fallback_model:
            self._fallback_client = OpenAI(
                api_key=fallback_api_key, base_url=fallback_base_url
            )
            self._async_fallback_client = AsyncOpenAI(
                api_key=fallback_api_key, base_url=fallback_base_url
            )
            self._fallback_model = fallback_model
        
        self._max_retries = 3
        self._base_delay = 1.0
    
    @observe(as_type="generation")
    def chat(self, system_prompt: str, user_message: str) -> str:
        """Call LLM for text response.
        
        Args:
            system_prompt: System prompt for LLM
            user_message: User message to process
            
        Returns:
            LLM response text
            
        Raises:
            LLMCallError: If API call fails after retries
        """
        get_client().update_current_trace(
            tags=["llm_call", "generation"],
            metadata={
                "client": "LLMClient",
                "prompt_length": len(system_prompt) + len(user_message)
            }
        )
        try:
            return self._chat_with_retries(
                client=self._client,
                model=self._model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except LLMCallError as primary_error:
            if not self._fallback_client:
                raise
            
            logger.warning(
                "Primary LLM failed; using fallback: %s",
                primary_error,
            )
            try:
                return self._chat_with_retries(
                    client=self._fallback_client,
                    model=self._fallback_model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                )
            except LLMCallError as fallback_error:
                # Surface combined failure context
                raise LLMCallError(
                    f"{self._model} (primary + fallback {self._fallback_model})",
                    self._max_retries,
                    fallback_error.last_error,
                ) from fallback_error
    
    @observe(as_type="generation")
    def chat_stream(self, system_prompt: str, user_message: str):
        """Call LLM for streaming text response.
        
        Args:
            system_prompt: System prompt for LLM
            user_message: User message to process
            
        Yields:
            Text chunks from LLM response
            
        Raises:
            LLMCallError: If API call fails after retries
        """
        get_client().update_current_trace(
            tags=["llm_call", "streaming", "generation"],
            metadata={
                "client": "LLMClient",
                "streaming": True
            }
        )
        try:
            yield from self._chat_stream_with_retries(
                client=self._client,
                model=self._model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except LLMCallError as primary_error:
            if not self._fallback_client:
                raise
            
            logger.warning(
                "Primary LLM failed; using fallback: %s",
                primary_error,
            )
            try:
                yield from self._chat_stream_with_retries(
                    client=self._fallback_client,
                    model=self._fallback_model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                )
            except LLMCallError as fallback_error:
                # Surface combined failure context
                raise LLMCallError(
                    f"{self._model} (primary + fallback {self._fallback_model})",
                    self._max_retries,
                    fallback_error.last_error,
                ) from fallback_error
    
    @observe(as_type="generation")
    async def chat_stream_async(self, system_prompt: str, user_message: str):
        """Async streaming chat using native AsyncOpenAI client.
        
        This method provides true async streaming without blocking the event loop.
        Much more efficient than chat_stream() wrapped in asyncio.to_thread().
        
        Args:
            system_prompt: System prompt for LLM
            user_message: User message to process
            
        Yields:
            Text chunks from LLM response
            
        Raises:
            LLMCallError: If API call fails after retries
        """
        get_client().update_current_trace(
            tags=["llm_call", "async_streaming", "generation"],
            metadata={
                "client": "LLMClient",
                "streaming": True,
                "async": True
            }
        )
        try:
            async for chunk in self._chat_stream_async_with_retries(
                client=self._async_client,
                model=self._model,
                system_prompt=system_prompt,
                user_message=user_message,
            ):
                yield chunk
        except LLMCallError as primary_error:
            if not self._async_fallback_client:
                raise
            
            logger.warning(
                "Primary async LLM failed; using async fallback: %s",
                primary_error,
            )
            try:
                async for chunk in self._chat_stream_async_with_retries(
                    client=self._async_fallback_client,
                    model=self._fallback_model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                ):
                    yield chunk
            except LLMCallError as fallback_error:
                raise LLMCallError(
                    f"{self._model} (primary + fallback {self._fallback_model})",
                    self._max_retries,
                    fallback_error.last_error,
                ) from fallback_error
    
    async def _chat_stream_async_with_retries(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        user_message: str,
    ):
        """Async streaming with retry logic."""
        last_error: Optional[Exception] = None
        
        for attempt in range(self._max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    stream=True
                )
                
                # Use async for to iterate without blocking
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # Success, exit retry loop
                
            except Exception as e:
                last_error = e
                logger.warning(
                    "Async LLM streaming attempt %s/%s for model %s failed: %s",
                    attempt + 1,
                    self._max_retries,
                    model,
                    e,
                )
                if attempt < self._max_retries - 1:
                    import asyncio
                    delay = self._base_delay * (2**attempt)
                    await asyncio.sleep(delay)
        
        raise LLMCallError(model, self._max_retries, last_error)
    
    def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        default: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Call LLM and parse JSON response with safe fallback.
        
        Args:
            system_prompt: System prompt for the LLM
            user_message: User message to process
            default: Default value to return if JSON parsing fails
            
        Returns:
            Dict containing:
            - parsed_data: Parsed JSON response or default value
            - raw_response: Original response text from LLM
            - model: Model used for the request
            - success: Whether parsing was successful
        """
        if default is None:
            default = {}
        
        try:
            response_text = self.chat(system_prompt, user_message)
            parsed_data = self._safe_parse_json(response_text, default)
            
            return {
                "parsed_data": parsed_data,
                "raw_response": response_text,
                "model": self._model,
                "success": True
            }
        except LLMCallError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chat_json: {e}")
            return {
                "parsed_data": default,
                "raw_response": "",
                "model": self._model,
                "success": False,
                "error": str(e)
            }
    
    def _safe_parse_json(
        self,
        response: str,
        default: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse JSON response with fallback to default.
        
        Args:
            response: Raw response text from LLM
            default: Default value if parsing fails
            
        Returns:
            Parsed JSON or default value
        """
        if not response:
            logger.error("Empty response from LLM")
            return default
        
        # Try to extract JSON from response (handle markdown code blocks)
        text = response.strip()
        
        # Remove markdown code block if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}. Response: {response[:200]}")
            return default

    def _chat_with_retries(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Call a specific client with retries."""
        last_error: Optional[Exception] = None
        
        for attempt in range(self._max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM API attempt %s/%s for model %s failed: %s",
                    attempt + 1,
                    self._max_retries,
                    model,
                    e,
                )
                if attempt < self._max_retries - 1:
                    delay = self._base_delay * (2**attempt)
                    time.sleep(delay)
        
        raise LLMCallError(model, self._max_retries, last_error)
    
    def _chat_stream_with_retries(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        user_message: str,
    ):
        """Call a specific client with retries for streaming."""
        last_error: Optional[Exception] = None
        
        for attempt in range(self._max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    stream=True  # Enable streaming
                )
                
                # Yield content chunks from the stream
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # Success, exit the retry loop
                
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM API streaming attempt %s/%s for model %s failed: %s",
                    attempt + 1,
                    self._max_retries,
                    model,
                    e,
                )
                if attempt < self._max_retries - 1:
                    delay = self._base_delay * (2**attempt)
                    time.sleep(delay)
        
        raise LLMCallError(model, self._max_retries, last_error)
