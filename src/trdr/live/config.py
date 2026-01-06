"""Live trading configuration."""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class AlpacaCredentials:
    """Alpaca API credentials.

    Args:
        api_key: API key
        api_secret: API secret
    """

    api_key: str
    api_secret: str


@dataclass
class RiskLimits:
    """Risk limits for circuit breaker.

    Args:
        max_drawdown_pct: Halt if drawdown exceeds this (%)
        max_daily_loss_pct: Halt if daily loss exceeds this (%)
        max_consecutive_losses: Halt after N consecutive losses
        max_position_pct: Max position as % of equity
        max_orders_per_hour: Rate limit for orders
        max_position_value: Max position value in dollars
    """

    max_drawdown_pct: float = 10.0
    max_daily_loss_pct: float = 5.0
    max_consecutive_losses: int = 5
    max_position_pct: float = 100.0
    max_orders_per_hour: int = 100
    max_position_value: float | None = None


@dataclass
class LiveConfig:
    """Configuration for live trading.

    Args:
        mode: Trading mode ("paper" or "live")
        paper_credentials: Alpaca paper trading credentials
        live_credentials: Alpaca live trading credentials
        poll_interval_seconds: Time between strategy polls
        order_timeout_seconds: Order submission timeout
        max_retries: Max retry attempts for failed operations
        risk_limits: Circuit breaker risk limits
        symbol: Trading symbol
        timeframe: Trading timeframe (e.g., "15m")
        enable_websocket: Use WebSocket for fills (vs polling)
        log_file: Audit log file path
    """

    mode: Literal["paper", "live"]
    paper_credentials: AlpacaCredentials | None = None
    live_credentials: AlpacaCredentials | None = None
    poll_interval_seconds: float = 60.0
    order_timeout_seconds: float = 30.0
    max_retries: int = 3
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    symbol: str = ""
    timeframe: str = "15m"
    enable_websocket: bool = True
    log_file: str = "live_trading.audit.log"

    @property
    def credentials(self) -> AlpacaCredentials:
        """Get credentials for current mode."""
        if self.mode == "paper":
            if not self.paper_credentials:
                raise ValueError("Paper credentials not configured")
            return self.paper_credentials
        else:
            if not self.live_credentials:
                raise ValueError("Live credentials not configured")
            return self.live_credentials

    @property
    def is_paper(self) -> bool:
        """Check if running in paper mode."""
        return self.mode == "paper"

    @classmethod
    def from_env(cls, mode: Literal["paper", "live"] | None = None) -> "LiveConfig":
        """Create config from environment variables.

        Environment variables:
            ALPACA_MODE: "paper" or "live" (default: paper)
            ALPACA_PAPER_API_KEY: Paper trading API key
            ALPACA_PAPER_API_SECRET: Paper trading API secret
            ALPACA_LIVE_API_KEY: Live trading API key
            ALPACA_LIVE_API_SECRET: Live trading API secret
            LIVE_POLL_INTERVAL: Poll interval in seconds (default: 60)
            LIVE_SYMBOL: Trading symbol (default: "")
            LIVE_TIMEFRAME: Trading timeframe (default: "15m")

        Args:
            mode: Override mode from env var

        Returns:
            LiveConfig instance
        """
        if mode is None:
            mode_str = os.getenv("ALPACA_MODE", "paper").lower()
            mode = "live" if mode_str == "live" else "paper"

        paper_key = os.getenv("ALPACA_PAPER_API_KEY", "")
        paper_secret = os.getenv("ALPACA_PAPER_API_SECRET", "")
        live_key = os.getenv("ALPACA_LIVE_API_KEY", "")
        live_secret = os.getenv("ALPACA_LIVE_API_SECRET", "")

        paper_credentials = None
        if paper_key and paper_secret:
            paper_credentials = AlpacaCredentials(paper_key, paper_secret)

        live_credentials = None
        if live_key and live_secret:
            live_credentials = AlpacaCredentials(live_key, live_secret)

        poll_interval = float(os.getenv("LIVE_POLL_INTERVAL", "60"))
        symbol = os.getenv("LIVE_SYMBOL", "")
        timeframe = os.getenv("LIVE_TIMEFRAME", "15m")

        return cls(
            mode=mode,
            paper_credentials=paper_credentials,
            live_credentials=live_credentials,
            poll_interval_seconds=poll_interval,
            symbol=symbol,
            timeframe=timeframe,
        )

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        if self.mode == "paper" and not self.paper_credentials:
            errors.append("Paper credentials required for paper mode")
        if self.mode == "live" and not self.live_credentials:
            errors.append("Live credentials required for live mode")

        if self.poll_interval_seconds < 1.0:
            errors.append("Poll interval must be >= 1 second")
        if self.order_timeout_seconds < 1.0:
            errors.append("Order timeout must be >= 1 second")
        if self.max_retries < 0:
            errors.append("Max retries must be >= 0")

        if self.risk_limits.max_drawdown_pct <= 0:
            errors.append("Max drawdown must be > 0")
        if self.risk_limits.max_daily_loss_pct <= 0:
            errors.append("Max daily loss must be > 0")
        if self.risk_limits.max_position_pct <= 0:
            errors.append("Max position pct must be > 0")

        return errors
