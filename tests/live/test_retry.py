"""Tests for retry logic."""

import pytest

from trdr.live.orders.retry import (
    NonRetryableError,
    RetryableError,
    RetryPolicy,
    retry_async,
    retry_sync,
)


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.base_delay_seconds == 1.0
        assert policy.max_delay_seconds == 30.0
        assert policy.exponential_base == 2.0

    def test_get_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(
            base_delay_seconds=1.0,
            exponential_base=2.0,
            jitter=0.0,
        )
        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0
        assert policy.get_delay(3) == 8.0

    def test_get_delay_respects_max(self):
        """Test delay respects max limit."""
        policy = RetryPolicy(
            base_delay_seconds=1.0,
            max_delay_seconds=5.0,
            exponential_base=2.0,
            jitter=0.0,
        )
        assert policy.get_delay(10) == 5.0

    def test_get_delay_with_jitter(self):
        """Test delay with jitter."""
        policy = RetryPolicy(
            base_delay_seconds=10.0,
            jitter=0.1,  # +/- 10%
        )
        # With jitter, delay should be within +/- 10% of base
        delay = policy.get_delay(0)
        assert 9.0 <= delay <= 11.0


class TestRetrySyncSuccess:
    """Tests for successful sync retry."""

    def test_immediate_success(self):
        """Test operation succeeds on first try."""
        calls = []

        def operation():
            calls.append(1)
            return "success"

        result = retry_sync(
            operation=operation,
            policy=RetryPolicy(max_attempts=3),
        )

        assert result.success
        assert result.result == "success"
        assert result.attempts == 1
        assert len(calls) == 1

    def test_success_after_retries(self):
        """Test operation succeeds after retries."""
        attempts = [0]

        def operation():
            attempts[0] += 1
            if attempts[0] < 3:
                raise RetryableError("temporary failure")
            return "success"

        result = retry_sync(
            operation=operation,
            policy=RetryPolicy(max_attempts=5, base_delay_seconds=0.01),
        )

        assert result.success
        assert result.result == "success"
        assert result.attempts == 3


class TestRetrySyncFailure:
    """Tests for failed sync retry."""

    def test_max_retries_exhausted(self):
        """Test failure after max retries."""

        def operation():
            raise RetryableError("always fails")

        result = retry_sync(
            operation=operation,
            policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.01),
        )

        assert not result.success
        assert result.attempts == 3
        assert len(result.errors) == 3
        assert isinstance(result.last_error, RetryableError)

    def test_non_retryable_error_stops_immediately(self):
        """Test non-retryable error stops retries."""
        attempts = [0]

        def operation():
            attempts[0] += 1
            raise NonRetryableError("permanent failure")

        result = retry_sync(
            operation=operation,
            policy=RetryPolicy(max_attempts=5, base_delay_seconds=0.01),
        )

        assert not result.success
        assert result.attempts == 1
        assert isinstance(result.last_error, NonRetryableError)


class TestRetryAsync:
    """Tests for async retry."""

    @pytest.mark.asyncio
    async def test_immediate_success(self):
        """Test async operation succeeds on first try."""

        async def operation():
            return "success"

        result = await retry_async(
            operation=operation,
            policy=RetryPolicy(max_attempts=3),
        )

        assert result.success
        assert result.result == "success"
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        """Test async operation succeeds after retries."""
        attempts = [0]

        async def operation():
            attempts[0] += 1
            if attempts[0] < 2:
                raise RetryableError("temporary failure")
            return "success"

        result = await retry_async(
            operation=operation,
            policy=RetryPolicy(max_attempts=5, base_delay_seconds=0.01),
        )

        assert result.success
        assert result.result == "success"
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test async failure after max retries."""

        async def operation():
            raise RetryableError("always fails")

        result = await retry_async(
            operation=operation,
            policy=RetryPolicy(max_attempts=2, base_delay_seconds=0.01),
        )

        assert not result.success
        assert result.attempts == 2
