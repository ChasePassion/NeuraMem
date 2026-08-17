"""Domain exceptions (migrated from src/memory_system/exceptions.py)."""


class MilvusConnectionError(Exception):
    """Raised when Milvus connection fails.

    Attributes:
        uri: The Milvus URI that failed to connect
        original_error: The underlying exception that caused the failure
    """

    def __init__(self, uri: str, original_error: Exception):
        self.uri = uri
        self.original_error = original_error
        super().__init__(f"Failed to connect to Milvus at {uri}: {original_error}")


class LLMCallError(Exception):
    """Raised when LLM API call fails after retries.

    Attributes:
        model: The model ID that was being called
        attempts: Number of retry attempts made
        last_error: The last exception that occurred
    """

    def __init__(self, model: str, attempts: int, last_error: Exception):
        self.model = model
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"LLM API call failed for {model} after {attempts} attempts: {last_error}"
        )


class LLMParseError(Exception):
    """Raised when an LLM response cannot be parsed as the expected JSON.

    Raised by callers of chat_json that must not silently proceed with the
    fallback default (architecture_target.md #22): treating a parse failure
    as "the LLM decided to do nothing" loses that turn's intended operation
    set with only a log line. The client already made one corrective repair
    retry before giving up.

    Attributes:
        model: The model ID whose response failed to parse
        raw_response: The unparseable response text (message truncates it)
    """

    def __init__(self, model: str, raw_response: str):
        self.model = model
        self.raw_response = raw_response
        super().__init__(
            f"LLM response from {model} was not valid JSON "
            f"(after one repair retry): {raw_response[:200]!r}"
        )
