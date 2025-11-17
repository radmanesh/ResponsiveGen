"""
LLM Debug Logger for tracking all LLM API calls.

Supports configurable log levels (NONE, INFO, DEBUG, TRACE) and dual output:
- Console: Human-readable formatted output
- File: JSON Lines format for parsing and analysis
"""

import json
import os
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


class LogLevel(Enum):
    """Logging levels for LLM debug output."""

    NONE = 0
    INFO = 1
    DEBUG = 2
    TRACE = 3


class LLMLogger:
    """Centralized logger for LLM API calls with configurable levels."""

    _instance: Optional["LLMLogger"] = None

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the logger with configuration from environment."""
        if self._initialized:
            return

        load_dotenv()

        # Get log level from environment
        level_str = os.getenv("LLM_DEBUG_LEVEL", "NONE").upper()
        try:
            self.level = LogLevel[level_str]
        except KeyError:
            self.level = LogLevel.NONE

        # Get file logging configuration
        self.log_to_file = os.getenv("LLM_LOG_TO_FILE", "true").lower() == "true"
        self.log_dir = Path(os.getenv("LLM_LOG_DIR", "outputs"))

        self._initialized = True

    def _should_log(self, min_level: LogLevel) -> bool:
        """Check if we should log at the given level."""
        return self.level.value >= min_level.value

    def _format_timestamp(self) -> str:
        """Get ISO8601 formatted timestamp."""
        return datetime.now().isoformat()

    def _truncate_content(self, content: str, max_len: int = 200) -> str:
        """Truncate content for preview."""
        if len(content) <= max_len:
            return content
        return content[:max_len] + "... [truncated]"

    def _has_image_content(self, content: Any) -> bool:
        """Check if content contains image data."""
        if isinstance(content, str):
            # Check for base64 image data
            if "data:image/" in content and "base64," in content:
                return True
        elif isinstance(content, list):
            # Check for multimodal content with images
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image_url" or "image" in str(item).lower():
                        return True
        elif isinstance(content, dict):
            if content.get("type") == "image_url" or "image" in str(content).lower():
                return True
        return False

    def _truncate_image_content(self, content: Any, for_prompt: bool = True) -> Any:
        """Truncate or replace image content with summary."""
        if isinstance(content, str):
            # Check if it's base64 image data
            if "data:image/" in content and "base64," in content:
                # Extract image type and size
                try:
                    parts = content.split("base64,")
                    if len(parts) == 2:
                        image_type = parts[0].split("image/")[1].split(";")[0]
                        data_size = len(parts[1])
                        if for_prompt:
                            return f"[IMAGE_DATA: {image_type}, base64 encoded, {data_size:,} bytes]"
                        else:
                            # For responses, check if there's text before the image
                            text_before = content.split("data:image/")[0].strip()
                            if text_before:
                                # Keep text, exclude image
                                return text_before
                            # Pure image content, exclude entirely
                            return None
                except:
                    pass
                if for_prompt:
                    return "[IMAGE_DATA: base64 encoded image]"
                # For responses, try to extract any text before image
                text_before = content.split("data:image/")[0].strip() if "data:image/" in content else ""
                return text_before if text_before else None

        elif isinstance(content, list):
            # Handle multimodal content
            filtered_items = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "image_url":
                        if for_prompt:
                            url = item.get("image_url", {})
                            if isinstance(url, dict):
                                url_value = url.get("url", "")
                            else:
                                url_value = str(url)
                            if "data:image/" in url_value and "base64," in url_value:
                                try:
                                    parts = url_value.split("base64,")
                                    if len(parts) == 2:
                                        image_type = parts[0].split("image/")[1].split(";")[0]
                                        data_size = len(parts[1])
                                        filtered_items.append({
                                            "type": "text",
                                            "text": f"[IMAGE_DATA: {image_type}, base64 encoded, {data_size:,} bytes]"
                                        })
                                except:
                                    filtered_items.append({"type": "text", "text": "[IMAGE_DATA: base64 encoded image]"})
                            else:
                                filtered_items.append({"type": "text", "text": f"[IMAGE_URL: {url_value[:100]}...]"})
                        # For responses, skip image items entirely
                    elif item_type == "text":
                        # Keep text content as is
                        filtered_items.append(item)
                    else:
                        # Keep other content types
                        filtered_items.append(item)
                else:
                    filtered_items.append(item)
            # If all items were filtered out (only images in response), return None
            if not filtered_items and not for_prompt:
                return None
            return filtered_items if filtered_items else content

        elif isinstance(content, dict):
            if content.get("type") == "image_url":
                if for_prompt:
                    return {"type": "text", "text": "[IMAGE_DATA: base64 encoded image]"}
                return None

        return content

    def _serialize_message(self, msg: Any, truncate_images: bool = True) -> Dict[str, Any]:
        """Serialize a message object to dict, optionally truncating images."""
        if hasattr(msg, "content"):
            content = msg.content if hasattr(msg, "content") else str(msg)
            # Truncate image content if requested
            if truncate_images and self._has_image_content(content):
                content = self._truncate_image_content(content, for_prompt=True)
            return {
                "type": msg.__class__.__name__,
                "content": content,
            }
        return {"type": type(msg).__name__, "content": str(msg)}

    def _format_console_info(
        self,
        component: str,
        provider: str,
        model: str,
        latency_ms: float,
        token_count: Optional[int] = None,
    ) -> str:
        """Format basic info line for console."""
        parts = [
            f"[{component}]",
            f"{provider}/{model}",
            f"{latency_ms:.1f}ms",
        ]
        if token_count is not None:
            parts.append(f"{token_count} tokens")
        return " | ".join(parts)

    def _format_console_debug(
        self,
        request_messages: List[Any],
        response_content: Optional[str] = None,
        tool_calls: Optional[List] = None,
    ) -> str:
        """Format debug info for console."""
        lines = []
        lines.append(f"  Messages: {len(request_messages)}")
        if request_messages:
            for i, msg in enumerate(request_messages[:3]):  # Show first 3
                msg_content = (
                    msg.content if hasattr(msg, "content") else str(msg)
                )
                # Truncate images in message content for console
                if self._has_image_content(msg_content):
                    msg_content = self._truncate_image_content(msg_content, for_prompt=True)
                    if not isinstance(msg_content, str):
                        if isinstance(msg_content, (list, dict)):
                            msg_content = json.dumps(msg_content, indent=0, ensure_ascii=False)
                        else:
                            msg_content = str(msg_content)
                preview = self._truncate_content(str(msg_content), 150)
                msg_type = msg.__class__.__name__ if hasattr(msg, "__class__") else type(msg).__name__
                lines.append(f"    {i+1}. [{msg_type}] {preview}")
            if len(request_messages) > 3:
                lines.append(f"    ... and {len(request_messages) - 3} more")

        if response_content:
            # response_content should already be truncated (images removed) by log_response
            preview = self._truncate_content(str(response_content), 200)
            lines.append(f"  Response: {preview}")

        if tool_calls:
            lines.append(f"  Tool Calls: {len(tool_calls)}")
            for i, tool_call in enumerate(tool_calls[:2]):
                tool_name = (
                    tool_call.get("name", "unknown")
                    if isinstance(tool_call, dict)
                    else getattr(tool_call, "name", "unknown")
                )
                lines.append(f"    - {tool_name}")
            if len(tool_calls) > 2:
                lines.append(f"    ... and {len(tool_calls) - 2} more")

        return "\n".join(lines)

    def _format_console_trace(
        self,
        request_messages: List[Any],
        response_content: str,
        tool_calls: Optional[List] = None,
        token_usage: Optional[Dict] = None,
    ) -> str:
        """Format full trace info for console."""
        lines = []
        lines.append("  REQUEST MESSAGES:")
        for i, msg in enumerate(request_messages):
            # Serialize with image truncation for prompts
            msg_dict = self._serialize_message(msg, truncate_images=True)
            lines.append(f"    [{i+1}] {msg_dict['type']}:")
            content = msg_dict.get("content", "")
            # Ensure content is a string
            if not isinstance(content, str):
                if isinstance(content, (list, dict)):
                    content = json.dumps(content, indent=2, ensure_ascii=False)
                else:
                    content = str(content)
            # Truncate very long text content (but not image placeholders)
            if isinstance(content, str) and len(content) > 500 and "[IMAGE_DATA:" not in content:
                lines.append(f"      {content[:500]}...")
                lines.append(f"      ... [{len(content) - 500} more chars]")
            else:
                for line in str(content).split("\n"):
                    lines.append(f"      {line}")

        lines.append("\n  RESPONSE:")
        # Ensure response_content is a string
        if not isinstance(response_content, str):
            if isinstance(response_content, (list, dict)):
                response_content = json.dumps(response_content, indent=2, ensure_ascii=False)
            else:
                response_content = str(response_content)
        if len(response_content) > 1000:
            lines.append(f"    {response_content[:1000]}...")
            lines.append(f"    ... [{len(response_content) - 1000} more chars]")
        else:
            for line in response_content.split("\n"):
                lines.append(f"    {line}")

        if tool_calls:
            lines.append("\n  TOOL CALLS:")
            for i, tool_call in enumerate(tool_calls):
                if isinstance(tool_call, dict):
                    lines.append(f"    [{i+1}] {json.dumps(tool_call, indent=6)}")
                else:
                    lines.append(f"    [{i+1}] {str(tool_call)}")

        if token_usage:
            lines.append("\n  TOKEN USAGE:")
            for key, value in token_usage.items():
                lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def _write_to_file(self, sample_id: Optional[str], log_entry: Dict[str, Any]):
        """Write log entry to JSON Lines file."""
        if not self.log_to_file or not sample_id:
            return

        log_file = self.log_dir / sample_id / "logs" / "llm_calls.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Append to file (JSON Lines format)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_invocation(
        self,
        component: str,
        provider: str,
        model: str,
        sample_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log the start of an LLM invocation.

        Returns:
            Invocation ID (UUID string) for tracking this call
        """
        if not self._should_log(LogLevel.INFO):
            return ""

        invocation_id = str(uuid.uuid4())
        timestamp = self._format_timestamp()

        if self._should_log(LogLevel.INFO):
            console_msg = f"[{timestamp}] 🔵 LLM Call: [{component}] {provider}/{model}"
            if sample_id:
                console_msg += f" | sample_id: {sample_id}"
            print(console_msg)

        return invocation_id

    def log_request(
        self,
        invocation_id: str,
        component: str,
        provider: str,
        model: str,
        messages: List[Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List] = None,
        sample_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log LLM request details."""
        if not self._should_log(LogLevel.DEBUG):
            return

        timestamp = self._format_timestamp()

        # Serialize messages for file logging
        serialized_messages = [self._serialize_message(msg) for msg in messages]

        # Console output for DEBUG/TRACE
        if self._should_log(LogLevel.DEBUG):
            console_output = self._format_console_debug(messages, tool_calls=tools)
            print(console_output)

        # Prepare log entry
        log_entry = {
            "timestamp": timestamp,
            "level": "DEBUG" if self.level == LogLevel.DEBUG else "TRACE",
            "component": component,
            "invocation_id": invocation_id,
            "provider": provider,
            "model": model,
            "sample_id": sample_id,
            "request": {
                "messages": serialized_messages if self.level == LogLevel.TRACE else [],
                "message_count": len(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": (
                    [
                        {
                            "name": (
                                tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
                            )
                        }
                        for tool in tools
                    ]
                    if tools
                    else None
                ) if self.level.value < LogLevel.TRACE.value else tools,
            },
            "metadata": metadata or {},
        }

        # For TRACE, include full message content
        if self.level == LogLevel.TRACE:
            log_entry["request"]["messages"] = serialized_messages
            if tools:
                log_entry["request"]["tool_definitions"] = tools

        self._write_to_file(sample_id, log_entry)

    def log_response(
        self,
        invocation_id: str,
        component: str,
        provider: str,
        model: str,
        request_messages: List[Any],
        response: Any,
        start_time: float,
        end_time: float,
        sample_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log LLM response with full details."""
        if not self._should_log(LogLevel.INFO):
            return

        timestamp = self._format_timestamp()
        latency_ms = (end_time - start_time) * 1000

        # Extract response content
        if hasattr(response, "content"):
            response_content = response.content
            # Truncate image content from responses for console output
            response_content_for_console = self._truncate_image_content(response_content, for_prompt=False)
            if response_content_for_console is None:
                # All content was images, replace with placeholder
                response_content_for_console = "[RESPONSE: Contains only image data, excluded from console output]"

            # For console, use truncated version (without images)
            console_response_content = response_content_for_console

            # Ensure console_response_content is a string
            if not isinstance(console_response_content, str):
                if isinstance(console_response_content, (list, dict)):
                    console_response_content = json.dumps(console_response_content, indent=2, ensure_ascii=False)
                else:
                    console_response_content = str(console_response_content)

            # For file logging, keep full content but serialize properly
            if not isinstance(response_content, str):
                if isinstance(response_content, (list, dict)):
                    response_content_str = json.dumps(response_content, indent=2, ensure_ascii=False)
                else:
                    response_content_str = str(response_content)
            else:
                response_content_str = response_content

            # Use truncated version for console, full for file
            response_content = console_response_content
        else:
            response_content = str(response)

        # Extract tool calls
        tool_calls = None
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = response.tool_calls
        elif hasattr(response, "response_metadata"):
            tool_calls = response.response_metadata.get("tool_calls")

        # Extract token usage
        token_usage = {}
        total_tokens = None
        if hasattr(response, "usage_metadata"):
            token_usage = {
                "prompt_tokens": getattr(response.usage_metadata, "input_tokens", None),
                "completion_tokens": getattr(response.usage_metadata, "output_tokens", None),
                "total_tokens": getattr(response.usage_metadata, "total_tokens", None),
            }
            total_tokens = token_usage.get("total_tokens")
        elif hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("usage", {})
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
            total_tokens = token_usage.get("total_tokens")

        # Console output
        if self._should_log(LogLevel.INFO):
            console_msg = f"[{timestamp}] ✅ LLM Response: "
            console_msg += self._format_console_info(
                component, provider, model, latency_ms, total_tokens
            )
            print(console_msg)

        if self._should_log(LogLevel.DEBUG):
            console_output = self._format_console_debug(
                request_messages, response_content, tool_calls
            )
            print(console_output)

        if self._should_log(LogLevel.TRACE):
            # Use truncated response content for console
            console_output = self._format_console_trace(
                request_messages, response_content, tool_calls, token_usage
            )
            print(console_output)

        # Prepare log entry
        # For file logging, serialize messages with image truncation for prompts
        serialized_messages = (
            [self._serialize_message(msg, truncate_images=True) for msg in request_messages]
            if self.level == LogLevel.TRACE
            else []
        )

        # For response, get the original response content (before truncation)
        # We need to re-extract it since we modified response_content for console
        if hasattr(response, "content"):
            original_response_content = response.content
            # For file logging, truncate images but keep structure
            file_response_content = self._truncate_image_content(original_response_content, for_prompt=False)
            if file_response_content is None:
                file_response_content = "[RESPONSE: Contains only image data]"

            # Convert to string for file logging
            if not isinstance(file_response_content, str):
                if isinstance(file_response_content, (list, dict)):
                    file_response_content_str = json.dumps(file_response_content, indent=2, ensure_ascii=False)
                else:
                    file_response_content_str = str(file_response_content)
            else:
                file_response_content_str = file_response_content
        else:
            file_response_content_str = str(response)

        log_entry = {
            "timestamp": timestamp,
            "level": self.level.name,
            "component": component,
            "invocation_id": invocation_id,
            "provider": provider,
            "model": model,
            "sample_id": sample_id,
            "request": {
                "messages": serialized_messages,
                "message_count": len(request_messages),
            },
            "response": {
                "content": file_response_content_str if self.level == LogLevel.TRACE else None,
                "content_preview": (
                    self._truncate_content(file_response_content_str, 200)
                    if self.level.value >= LogLevel.DEBUG.value
                    else None
                ),
                "content_length": len(file_response_content_str),
                "tool_calls": tool_calls if tool_calls else None,
            },
            "timing": {
                "latency_ms": latency_ms,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
            },
            "usage": token_usage if token_usage else None,
            "metadata": metadata or {},
        }

        # For TRACE, always include full content (with images truncated)
        if self.level == LogLevel.TRACE:
            log_entry["response"]["content"] = file_response_content_str
            log_entry["request"]["messages"] = serialized_messages

        self._write_to_file(sample_id, log_entry)


def get_logger() -> LLMLogger:
    """Get the singleton logger instance."""
    return LLMLogger()


class LoggedLLM:
    """
    Wrapper around LangChain LLM instances to add debug logging.

    Intercepts invoke() calls and logs requests, responses, timing, and metadata.
    """

    def __init__(
        self,
        llm_instance: Any,
        component: str,
        provider: str,
        model: str,
        sample_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LoggedLLM wrapper.

        Args:
            llm_instance: The actual LLM instance (ChatOpenAI or ChatAnthropic)
            component: Component name (e.g., "orchestrator", "generator", "judge")
            provider: Provider name ("openai" or "anthropic")
            model: Model name
            sample_id: Optional sample ID for tracking
            metadata: Optional additional metadata to include in logs
        """
        self.llm = llm_instance
        self.component = component
        self.provider = provider
        self.model = model
        self.sample_id = sample_id
        self.metadata = metadata or {}
        self.logger = get_logger()

    def __getattr__(self, name: str):
        """Delegate all other attributes to wrapped LLM instance."""
        return getattr(self.llm, name)

    def invoke(self, messages: List[Any], **kwargs) -> Any:
        """
        Invoke LLM with logging.

        Args:
            messages: List of message objects
            **kwargs: Additional arguments passed to LLM

        Returns:
            LLM response
        """
        # Start logging
        invocation_id = self.logger.log_invocation(
            component=self.component,
            provider=self.provider,
            model=self.model,
            sample_id=self.sample_id,
            metadata=self.metadata,
        )

        if not invocation_id:
            # Logging disabled, just call directly
            return self.llm.invoke(messages, **kwargs)

        # Log request details
        temperature = getattr(self.llm, "temperature", None)
        max_tokens = getattr(self.llm, "max_tokens", None)

        # Check for tools in kwargs or bound tools
        tools = kwargs.get("tools") or getattr(self.llm, "bound_tools", None)

        start_time = time.time()
        self.logger.log_request(
            invocation_id=invocation_id,
            component=self.component,
            provider=self.provider,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            sample_id=self.sample_id,
            metadata=self.metadata,
        )

        # Invoke the actual LLM
        try:
            response = self.llm.invoke(messages, **kwargs)
        except Exception as e:
            # Log error if needed
            if self.logger._should_log(LogLevel.INFO):
                timestamp = self.logger._format_timestamp()
                print(
                    f"[{timestamp}] ❌ LLM Error: [{self.component}] {type(e).__name__}: {str(e)}"
                )
            raise

        end_time = time.time()

        # Log response
        self.logger.log_response(
            invocation_id=invocation_id,
            component=self.component,
            provider=self.provider,
            model=self.model,
            request_messages=messages,
            response=response,
            start_time=start_time,
            end_time=end_time,
            sample_id=self.sample_id,
            metadata=self.metadata,
        )

        return response

    def bind_tools(self, tools: List[Any], **kwargs):
        """
        Bind tools to the LLM (returns a new instance with tools bound).

        Args:
            tools: List of tool objects
            **kwargs: Additional arguments

        Returns:
            New LoggedLLM instance with tools bound
        """
        bound_llm = self.llm.bind_tools(tools, **kwargs)
        # Return new wrapper with same configuration
        return LoggedLLM(
            llm_instance=bound_llm,
            component=self.component,
            provider=self.provider,
            model=self.model,
            sample_id=self.sample_id,
            metadata=self.metadata,
        )

