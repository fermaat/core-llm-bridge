"""
Token counting utilities for managing context windows.

Provides utilities to estimate token counts for different models.
"""


class TokenCounter:
    """
    Utility for estimating token counts.

    Different models use different tokenizers and token counts.
    This provides reasonable estimates for common models.

    Note: These are approximations. Actual token counts may vary.
    For accurate counts, use the provider's native tokenizer if available.
    """

    # Approximate tokens per word for different models
    TOKENS_PER_WORD = {
        "llama": 1.3,  # Llama models
        "mistral": 1.3,
        "neural": 1.2,
        "gpt": 1.3,  # GPT models
        "claude": 1.4,  # Claude models
        "default": 1.3,
    }

    @staticmethod
    def estimate_tokens(text: str, model: str = "default") -> int:
        """
        Estimate the number of tokens for given text.

        Simple estimation based on word count and model-specific multiplier.

        Args:
            text: The text to count tokens for
            model: The model name (used to select appropriate multiplier)

        Returns:
            Estimated token count

        Example:
            >>> counter = TokenCounter()
            >>> tokens = counter.estimate_tokens("Hello world", model="llama")
            >>> print(tokens)  # Approximately 3
        """
        if not text:
            return 0

        # Get model-specific multiplier
        model_lower = model.lower()
        multiplier = TokenCounter.TOKENS_PER_WORD.get("default", 1.3)

        # Try to find a model-specific multiplier
        for key, value in TokenCounter.TOKENS_PER_WORD.items():
            if key in model_lower:
                multiplier = value
                break

        # Count words (simple split on whitespace)
        word_count = len(text.split())

        # Estimate tokens
        estimated_tokens = int(word_count * multiplier)

        return max(estimated_tokens, 1)  # At least 1 token

    @staticmethod
    def count_messages_tokens(messages: list[dict[str, str]], model: str = "default") -> int:
        """
        Estimate total tokens for a list of messages.

        Accounts for message overhead (role, separators, etc).

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model name

        Returns:
            Estimated total token count

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi there!"}
            ... ]
            >>> tokens = TokenCounter.count_messages_tokens(messages)
            >>> print(tokens)
        """
        total_tokens = 0

        for message in messages:
            # Add tokens for the message content
            content = message.get("content", "")
            total_tokens += TokenCounter.estimate_tokens(content, model)

            # Add overhead for message structure (~4-5 tokens per message)
            total_tokens += 4

        return total_tokens

    @staticmethod
    def will_fit_in_context(
        text: str, max_tokens: int, model: str = "default", safety_margin: int = 0
    ) -> bool:
        """
        Check if text will fit within context window.

        Args:
            text: The text to check
            max_tokens: Maximum tokens allowed
            model: The model name
            safety_margin: Tokens to reserve for output

        Returns:
            True if text fits, False otherwise

        Example:
            >>> text = "This is a long document..." * 100
            >>> if TokenCounter.will_fit_in_context(text, max_tokens=4096):
            ...     print("Text fits!")
        """
        estimated = TokenCounter.estimate_tokens(text, model)
        return estimated + safety_margin <= max_tokens

    @staticmethod
    def truncate_to_fit(
        text: str,
        max_tokens: int,
        model: str = "default",
        safety_margin: int = 100,
    ) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: The text to truncate
            max_tokens: Maximum tokens allowed
            model: The model name
            safety_margin: Tokens to reserve for output

        Returns:
            Truncated text that fits within limits

        Example:
            >>> long_text = "....." * 1000
            >>> truncated = TokenCounter.truncate_to_fit(long_text, 2048)
        """
        if TokenCounter.will_fit_in_context(text, max_tokens, model, safety_margin):
            return text

        # Binary search for the right truncation point
        available_tokens = max_tokens - safety_margin
        words = text.split()

        # Estimate tokens per word for this model
        multiplier = TokenCounter.TOKENS_PER_WORD.get("default", 1.3)
        for key, value in TokenCounter.TOKENS_PER_WORD.items():
            if key in model.lower():
                multiplier = value
                break

        # Calculate how many words we can keep
        max_words = int(available_tokens / multiplier)

        # Ensure the truncation logic never returns a negative slice length
        if max_words <= 0:
            max_words = 1

        truncated = " ".join(words[:max_words])

        return truncated


__all__ = ["TokenCounter"]
