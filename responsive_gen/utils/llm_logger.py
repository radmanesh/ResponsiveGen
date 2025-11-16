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

    def _serialize_message(self, msg: Any) -> Dict[str, Any]:
        """Serialize a message object to dict."""
        if hasattr(msg, "content"):
            return {
                "type": msg.__class__.__name__,
                "content": msg.content if hasattr(msg, "content") else str(msg),
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
                preview = self._truncate_content(msg_content, 150)
                msg_type = msg.__class__.__name__ if hasattr(msg, "__class__") else type(msg).__name__
                lines.append(f"    {i+1}. [{msg_type}] {preview}")
            if len(request_messages) > 3:
                lines.append(f"    ... and {len(request_messages) - 3} more")

        if response_content:
            preview = self._truncate_content(response_content, 200)
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
            msg_dict = self._serialize_message(msg)
            lines.append(f"    [{i+1}] {msg_dict['type']}:")
            content = msg_dict.get("content", "")
            # Ensure content is a string
            if not isinstance(content, str):
                if isinstance(content, (list, dict)):
                    content = json.dumps(content, indent=2, ensure_ascii=False)
                else:
                    content = str(content)
            # For long content, show with proper indentation
            if len(content) > 500:
                lines.append(f"      {content[:500]}...")
                lines.append(f"      ... [{len(content) - 500} more chars]")
            else:
                for line in content.split("\n"):
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
            # Ensure response_content is a string (could be list/dict for multimodal)
            if not isinstance(response_content, str):
                if isinstance(response_content, (list, dict)):
                    response_content = json.dumps(response_content, indent=2, ensure_ascii=False)
                else:
                    response_content = str(response_content)
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
            console_output = self._format_console_trace(
                request_messages, response_content, tool_calls, token_usage
            )
            print(console_output)

        # Prepare log entry
        serialized_messages = (
            [self._serialize_message(msg) for msg in request_messages]
            if self.level == LogLevel.TRACE
            else []
        )

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
                "content": response_content if self.level == LogLevel.TRACE else None,
                "content_preview": (
                    self._truncate_content(response_content, 200)
                    if self.level.value >= LogLevel.DEBUG.value
                    else None
                ),
                "content_length": len(response_content),
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

        # For TRACE, always include full content
        if self.level == LogLevel.TRACE:
            log_entry["response"]["content"] = response_content
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

