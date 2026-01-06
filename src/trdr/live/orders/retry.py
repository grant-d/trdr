"""Retry policies and error handling for order operations."""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Error that can be retried."""

    pass


class NonRetryableError(Exception):
    """Error that should not be retried."""

    pass


@dataclass
class RetryPolicy:
    """Configuration for retry behavior.

    Args:
        max_attempts: Maximum number of attempts (including first try)
        base_delay_seconds: Initial delay between retries
        max_delay_seconds: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add randomness to delays (0.0-1.0)
    """

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.base_delay_seconds * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay_seconds)

        # Add jitter
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)


@dataclass
class RetryResult:
    """Result of a retry operation.

    Args:
        success: Whether the operation succeeded
        result: Result value if successful
        attempts: Number of attempts made
        last_error: Last error if failed
    """

    success: bool
    result: T | None = None
    attempts: int = 0
    last_error: Exception | None = None
    errors: list[Exception] = field(default_factory=list)


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    retryable_exceptions: tuple[type[Exception], ...] = (RetryableError,),
    operation_name: str = "operation",
) -> RetryResult[T]:
    """Execute async operation with retry logic.

    Args:
        operation: Async callable to execute
        policy: Retry policy configuration
        retryable_exceptions: Exception types that trigger retry
        operation_name: Name for logging

    Returns:
        RetryResult with success status and result/errors
    """
    errors: list[Exception] = []

    for attempt in range(policy.max_attempts):
        try:
            result = await operation()
            return RetryResult(
                success=True,
                result=result,
                attempts=attempt + 1,
                errors=errors,
            )
        except retryable_exceptions as e:
            errors.append(e)
            logger.warning(
                f"{operation_name} attempt {attempt + 1}/{policy.max_attempts} " f"failed: {e}"
            )

            if attempt + 1 < policy.max_attempts:
                delay = policy.get_delay(attempt)
                logger.info(f"Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
        except Exception as e:
            # Non-retryable error
            errors.append(e)
            logger.error(f"{operation_name} failed with non-retryable error: {e}")
            return RetryResult(
                success=False,
                attempts=attempt + 1,
                last_error=e,
                errors=errors,
            )

    # All retries exhausted
    return RetryResult(
        success=False,
        attempts=policy.max_attempts,
        last_error=errors[-1] if errors else None,
        errors=errors,
    )


def retry_sync(
    operation: Callable[[], T],
    policy: RetryPolicy,
    retryable_exceptions: tuple[type[Exception], ...] = (RetryableError,),
    operation_name: str = "operation",
) -> RetryResult[T]:
    """Execute sync operation with retry logic.

    Args:
        operation: Callable to execute
        policy: Retry policy configuration
        retryable_exceptions: Exception types that trigger retry
        operation_name: Name for logging

    Returns:
        RetryResult with success status and result/errors
    """
    import time

    errors: list[Exception] = []

    for attempt in range(policy.max_attempts):
        try:
            result = operation()
            return RetryResult(
                success=True,
                result=result,
                attempts=attempt + 1,
                errors=errors,
            )
        except retryable_exceptions as e:
            errors.append(e)
            logger.warning(
                f"{operation_name} attempt {attempt + 1}/{policy.max_attempts} " f"failed: {e}"
            )

            if attempt + 1 < policy.max_attempts:
                delay = policy.get_delay(attempt)
                logger.info(f"Retrying in {delay:.2f}s...")
                time.sleep(delay)
        except Exception as e:
            # Non-retryable error
            errors.append(e)
            logger.error(f"{operation_name} failed with non-retryable error: {e}")
            return RetryResult(
                success=False,
                attempts=attempt + 1,
                last_error=e,
                errors=errors,
            )

    # All retries exhausted
    return RetryResult(
        success=False,
        attempts=policy.max_attempts,
        last_error=errors[-1] if errors else None,
        errors=errors,
    )
