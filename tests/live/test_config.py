"""Tests for live trading config."""

import os
from unittest.mock import patch

import pytest

from trdr.live.config import AlpacaCredentials, LiveConfig, RiskLimits


class TestRiskLimits:
    """Tests for RiskLimits."""

    def test_default_values(self):
        """Test default risk limits."""
        limits = RiskLimits()
        assert limits.max_drawdown_pct == 10.0
        assert limits.max_daily_loss_pct == 5.0
        assert limits.max_consecutive_losses == 5
        assert limits.max_position_pct == 100.0
        assert limits.max_orders_per_hour == 100
        assert limits.max_position_value is None

    def test_custom_values(self):
        """Test custom risk limits."""
        limits = RiskLimits(
            max_drawdown_pct=5.0,
            max_daily_loss_pct=2.0,
            max_consecutive_losses=3,
        )
        assert limits.max_drawdown_pct == 5.0
        assert limits.max_daily_loss_pct == 2.0
        assert limits.max_consecutive_losses == 3


class TestAlpacaCredentials:
    """Tests for AlpacaCredentials."""

    def test_credential_creation(self):
        """Test credential creation."""
        creds = AlpacaCredentials(
            api_key="test_key",
            api_secret="test_secret",
        )
        assert creds.api_key == "test_key"
        assert creds.api_secret == "test_secret"


class TestLiveConfig:
    """Tests for LiveConfig."""

    def test_default_paper_mode(self):
        """Test default is paper mode."""
        paper_creds = AlpacaCredentials("key", "secret")
        config = LiveConfig(
            mode="paper",
            paper_credentials=paper_creds,
        )
        assert config.is_paper
        assert config.mode == "paper"

    def test_live_mode(self):
        """Test live mode."""
        live_creds = AlpacaCredentials("key", "secret")
        config = LiveConfig(
            mode="live",
            live_credentials=live_creds,
        )
        assert not config.is_paper
        assert config.mode == "live"

    def test_credentials_property_paper(self):
        """Test credentials property for paper mode."""
        paper_creds = AlpacaCredentials("paper_key", "paper_secret")
        config = LiveConfig(
            mode="paper",
            paper_credentials=paper_creds,
        )
        assert config.credentials.api_key == "paper_key"

    def test_credentials_property_live(self):
        """Test credentials property for live mode."""
        live_creds = AlpacaCredentials("live_key", "live_secret")
        config = LiveConfig(
            mode="live",
            live_credentials=live_creds,
        )
        assert config.credentials.api_key == "live_key"

    def test_credentials_missing_paper(self):
        """Test error when paper credentials missing."""
        config = LiveConfig(mode="paper")
        with pytest.raises(ValueError, match="Paper credentials"):
            _ = config.credentials

    def test_credentials_missing_live(self):
        """Test error when live credentials missing."""
        config = LiveConfig(mode="live")
        with pytest.raises(ValueError, match="Live credentials"):
            _ = config.credentials

    def test_validation_success(self):
        """Test validation passes with valid config."""
        paper_creds = AlpacaCredentials("key", "secret")
        config = LiveConfig(
            mode="paper",
            paper_credentials=paper_creds,
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_validation_missing_credentials(self):
        """Test validation fails without credentials."""
        config = LiveConfig(mode="paper")
        errors = config.validate()
        assert any("credentials" in e.lower() for e in errors)

    def test_validation_invalid_poll_interval(self):
        """Test validation fails with invalid poll interval."""
        paper_creds = AlpacaCredentials("key", "secret")
        config = LiveConfig(
            mode="paper",
            paper_credentials=paper_creds,
            poll_interval_seconds=0.5,
        )
        errors = config.validate()
        assert any("poll" in e.lower() for e in errors)

    def test_validation_invalid_risk_limits(self):
        """Test validation fails with invalid risk limits."""
        paper_creds = AlpacaCredentials("key", "secret")
        config = LiveConfig(
            mode="paper",
            paper_credentials=paper_creds,
            risk_limits=RiskLimits(max_drawdown_pct=0),
        )
        errors = config.validate()
        assert any("drawdown" in e.lower() for e in errors)

    def test_from_env(self):
        """Test config from environment variables."""
        env = {
            "ALPACA_MODE": "paper",
            "ALPACA_PAPER_API_KEY": "env_paper_key",
            "ALPACA_PAPER_API_SECRET": "env_paper_secret",
            "LIVE_POLL_INTERVAL": "30",
            "LIVE_SYMBOL": "ETH/USD",
            "LIVE_TIMEFRAME": "5m",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LiveConfig.from_env()
            assert config.is_paper
            assert config.paper_credentials.api_key == "env_paper_key"
            assert config.poll_interval_seconds == 30.0
            assert config.symbol == "ETH/USD"
            assert config.timeframe == "5m"

    def test_from_env_live_mode(self):
        """Test config from env with live mode."""
        env = {
            "ALPACA_MODE": "live",
            "ALPACA_LIVE_API_KEY": "env_live_key",
            "ALPACA_LIVE_API_SECRET": "env_live_secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LiveConfig.from_env()
            assert not config.is_paper
            assert config.live_credentials.api_key == "env_live_key"

    def test_from_env_mode_override(self):
        """Test env config with mode override."""
        env = {
            "ALPACA_MODE": "live",
            "ALPACA_PAPER_API_KEY": "paper_key",
            "ALPACA_PAPER_API_SECRET": "paper_secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LiveConfig.from_env(mode="paper")
            assert config.is_paper
