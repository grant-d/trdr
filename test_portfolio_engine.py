"""
Unit and integration tests for portfolio engine
"""

import unittest
import math
from datetime import datetime, timedelta
from portfolio_engine import (
    PortfolioEngine,
    OrderType,
    OrderSide,
    OrderStatus,
    Bar,
    AlpacaStockPortfolioEngine,
    AlpacaCryptoPortfolioEngine,
    CoinbasePortfolioEngine,
)


class TestPortfolioEngine(unittest.TestCase):
    """Unit tests for portfolio engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.engine = PortfolioEngine(
            initial_balance=100000.0,
            commission_rate=0.0,
            commission_pct=0.001,  # 0.1%
            slippage_pct=0.0001,  # 0.01%
            slippage_fixed=0.0,
            allow_shorts=False,
            symbol="TEST",
        )
        self.timestamp = datetime.now()

    def test_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.cash, 100000.0)
        self.assertEqual(self.engine.initial_balance, 100000.0)
        self.assertIsNone(self.engine.position)
        self.assertEqual(len(self.engine.orders), 0)
        self.assertEqual(len(self.engine.trades), 0)

    def test_place_market_order(self):
        """Test placing market orders"""
        # Test valid buy order
        order = self.engine.place_order(
            side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET
        )
        self.assertIsNotNone(order)
        if order:
            self.assertEqual(order.side, OrderSide.BUY)
            self.assertEqual(order.quantity, 100)
            self.assertEqual(order.order_type, OrderType.MARKET)
            self.assertEqual(order.status, OrderStatus.PENDING)

        # Test invalid quantity
        order = self.engine.place_order(
            side=OrderSide.BUY, quantity=-100, order_type=OrderType.MARKET
        )
        self.assertIsNone(order)

    def test_place_limit_order(self):
        """Test placing limit orders"""
        # Test valid limit order
        order = self.engine.place_order(
            side=OrderSide.BUY, quantity=100, order_type=OrderType.LIMIT, price=50.0
        )
        self.assertIsNotNone(order)
        if order:
            self.assertEqual(order.price, 50.0)

        # Test limit order without price
        order = self.engine.place_order(
            side=OrderSide.BUY, quantity=100, order_type=OrderType.LIMIT
        )
        self.assertIsNone(order)

    def test_cancel_order(self):
        """Test order cancellation"""
        order = self.engine.place_order(
            side=OrderSide.BUY, quantity=100, order_type=OrderType.LIMIT, price=50.0
        )
        self.assertIsNotNone(order)

        # Cancel order
        if order:
            success = self.engine.cancel_order(order.order_id)
            self.assertTrue(success)
            self.assertEqual(order.status, OrderStatus.CANCELLED)

        # Try to cancel again
        if order:
            success = self.engine.cancel_order(order.order_id)
            self.assertFalse(success)

    def test_market_order_execution(self):
        """Test market order execution"""
        # Place buy order
        order = self.engine.place_order(
            side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET
        )
        self.assertIsNotNone(order)

        # Process bar
        bar = Bar(
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10000,
            timestamp=self.timestamp,
        )
        trades = self.engine.process_bar(bar)

        # Verify execution
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].quantity, 100)
        self.assertEqual(trades[0].price, 100.01)  # With slippage

        # Verify position
        position = self.engine.get_position()
        self.assertIsNotNone(position)
        if position:
            self.assertEqual(position.quantity, 100)
            self.assertEqual(position.average_price, 100.01)

        # Verify cash
        expected_cash = 100000 - (100 * 100.01) - (100 * 100.01 * 0.001)
        self.assertAlmostEqual(self.engine.get_cash(), expected_cash, places=2)

    def test_limit_order_execution(self):
        """Test limit order execution"""
        # Place limit buy order
        order = self.engine.place_order(
            side=OrderSide.BUY, quantity=100, order_type=OrderType.LIMIT, price=99.5
        )
        self.assertIsNotNone(order)

        # Bar doesn't hit limit - no execution
        bar1 = Bar(
            open=100.0,
            high=101.0,
            low=99.6,
            close=100.0,
            volume=10000,
            timestamp=self.timestamp,
        )
        trades = self.engine.process_bar(bar1)
        self.assertEqual(len(trades), 0)
        if order:
            self.assertEqual(order.status, OrderStatus.PENDING)

        # Bar hits limit - execution
        bar2 = Bar(
            open=100.0,
            high=101.0,
            low=99.0,
            close=99.8,
            volume=10000,
            timestamp=self.timestamp + timedelta(minutes=1),
        )
        trades = self.engine.process_bar(bar2)
        self.assertEqual(len(trades), 1)
        if order:
            self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertAlmostEqual(trades[0].price, 99.51, places=2)  # Limit + slippage

    def test_stop_order_execution(self):
        """Test stop order execution"""
        # First buy some shares
        self.engine.place_order(OrderSide.BUY, 100, OrderType.MARKET)
        bar = Bar(
            open=100,
            high=100,
            low=100,
            close=100,
            volume=10000,
            timestamp=self.timestamp,
        )
        self.engine.process_bar(bar)

        # Place stop loss order
        order = self.engine.place_order(
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=98.0,
        )
        self.assertIsNotNone(order)

        # Bar doesn't hit stop
        bar1 = Bar(
            open=100.0,
            high=101.0,
            low=98.5,
            close=99.0,
            volume=10000,
            timestamp=self.timestamp + timedelta(minutes=1),
        )
        trades = self.engine.process_bar(bar1)
        self.assertEqual(len(trades), 0)

        # Bar hits stop
        bar2 = Bar(
            open=99.0,
            high=99.5,
            low=97.5,
            close=98.0,
            volume=10000,
            timestamp=self.timestamp + timedelta(minutes=2),
        )
        trades = self.engine.process_bar(bar2)
        self.assertEqual(len(trades), 1)
        if order:
            self.assertEqual(order.status, OrderStatus.FILLED)

    def test_trailing_stop_order(self):
        """Test trailing stop order"""
        # First buy some shares
        self.engine.place_order(OrderSide.BUY, 100, OrderType.MARKET)
        bar = Bar(
            open=100,
            high=100,
            low=100,
            close=100,
            volume=10000,
            timestamp=self.timestamp,
        )
        self.engine.process_bar(bar)

        # Place trailing stop order (2% trail)
        order = self.engine.place_order(
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.TRAILING_STOP,
            trail_percent=0.02,
        )
        self.assertIsNotNone(order)

        # Price goes up - trail adjusts
        bar1 = Bar(
            open=100.0,
            high=102.0,
            low=100.0,
            close=101.5,
            volume=10000,
            timestamp=self.timestamp + timedelta(minutes=1),
        )
        trades = self.engine.process_bar(bar1)
        self.assertEqual(len(trades), 0)  # No trigger

        # Price drops but not enough to trigger
        bar2 = Bar(
            open=101.5,
            high=101.5,
            low=100.5,
            close=100.5,
            volume=10000,
            timestamp=self.timestamp + timedelta(minutes=2),
        )
        trades = self.engine.process_bar(bar2)
        self.assertEqual(len(trades), 0)  # Still no trigger

        # Price drops enough to trigger (102 * 0.98 = 99.96)
        bar3 = Bar(
            open=100.5,
            high=100.5,
            low=99.5,
            close=99.8,
            volume=10000,
            timestamp=self.timestamp + timedelta(minutes=3),
        )
        trades = self.engine.process_bar(bar3)
        self.assertEqual(len(trades), 1)
        if order:
            self.assertEqual(order.status, OrderStatus.FILLED)

    def test_insufficient_funds(self):
        """Test order rejection for insufficient funds"""
        # Try to buy more than we can afford
        order = self.engine.place_order(
            side=OrderSide.BUY, quantity=10000, order_type=OrderType.MARKET
        )
        self.assertIsNotNone(order)

        bar = Bar(
            open=100,
            high=100,
            low=100,
            close=100,
            volume=10000,
            timestamp=self.timestamp,
        )
        trades = self.engine.process_bar(bar)

        # Order should be rejected
        self.assertEqual(len(trades), 0)

        if order:
            self.assertEqual(order.status, OrderStatus.REJECTED)

    def test_insufficient_position(self):
        """Test order rejection for insufficient position"""
        # Try to sell without position
        order = self.engine.place_order(
            side=OrderSide.SELL, quantity=100, order_type=OrderType.MARKET
        )
        self.assertIsNotNone(order)

        bar = Bar(
            open=100,
            high=100,
            low=100,
            close=100,
            volume=10000,
            timestamp=self.timestamp,
        )
        trades = self.engine.process_bar(bar)

        # Order should be rejected
        self.assertEqual(len(trades), 0)

        if order:
            self.assertEqual(order.status, OrderStatus.REJECTED)

    def test_commission_calculation(self):
        """Test commission calculation"""
        # Set up engine with specific commission
        engine = PortfolioEngine(
            initial_balance=100000.0,
            commission_rate=1.0,  # $1 flat
            commission_pct=0.001,  # 0.1%
            slippage_pct=0.0,
            symbol="TEST",
        )

        # Buy order
        order = engine.place_order(OrderSide.BUY, 100, OrderType.MARKET)
        self.assertIsNotNone(order)

        bar = Bar(
            open=100,
            high=100,
            low=100,
            close=100,
            volume=10000,
            timestamp=self.timestamp,
        )
        trades = engine.process_bar(bar)

        # Commission should be $1 + 0.1% of $10,000 = $1 + $10 = $11
        self.assertEqual(trades[0].commission, 11.0)

    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Execute some trades
        bars = [
            Bar(
                open=100,
                high=100,
                low=100,
                close=100,
                volume=10000,
                timestamp=self.timestamp,
            ),
            Bar(
                open=100,
                high=102,
                low=100,
                close=101,
                volume=10000,
                timestamp=self.timestamp + timedelta(days=1),
            ),
            Bar(
                open=101,
                high=103,
                low=101,
                close=102,
                volume=10000,
                timestamp=self.timestamp + timedelta(days=2),
            ),
            Bar(
                open=102,
                high=102,
                low=98,
                close=99,
                volume=10000,
                timestamp=self.timestamp + timedelta(days=3),
            ),
            Bar(
                open=99,
                high=101,
                low=99,
                close=100,
                volume=10000,
                timestamp=self.timestamp + timedelta(days=4),
            ),
        ]

        # Buy
        self.engine.place_order(OrderSide.BUY, 100, OrderType.MARKET)
        self.engine.process_bar(bars[0])

        # Hold through price changes
        for bar in bars[1:4]:
            self.engine.process_bar(bar)

        # Sell
        self.engine.place_order(OrderSide.SELL, 100, OrderType.MARKET)
        self.engine.process_bar(bars[4])

        # Get metrics
        metrics = self.engine.get_performance_metrics()

        # Verify some metrics exist and are reasonable
        self.assertIn("total_return", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown_pct", metrics)
        self.assertEqual(metrics["total_trades"], 1)  # One complete round trip

    def test_liquidate(self):
        """Test liquidate method"""
        # Test with no position
        order = self.engine.liquidate()
        self.assertIsNone(order)

        # Buy position
        self.engine.place_order(OrderSide.BUY, 100, OrderType.MARKET)
        bar = Bar(
            open=100,
            high=100,
            low=100,
            close=100,
            volume=10000,
            timestamp=self.timestamp,
        )
        self.engine.process_bar(bar)

        # Liquidate long position
        order = self.engine.liquidate()
        if order:
            self.assertIsNotNone(order)
            self.assertEqual(order.side, OrderSide.SELL)
            self.assertEqual(order.quantity, 100)
            self.assertEqual(order.order_type, OrderType.MARKET)

        # Process liquidation
        bar2 = Bar(
            open=100,
            high=100,
            low=100,
            close=100,
            volume=10000,
            timestamp=self.timestamp + timedelta(minutes=1),
        )
        trades = self.engine.process_bar(bar2)
        self.assertEqual(len(trades), 1)
        self.assertIsNone(self.engine.get_position())

    def test_liquidate_with_pending_orders(self):
        """Test liquidate cancels pending orders"""
        # Place some pending orders
        self.engine.place_order(OrderSide.BUY, 100, OrderType.LIMIT, price=95.0)
        self.engine.place_order(OrderSide.BUY, 50, OrderType.LIMIT, price=94.0)

        # Liquidate (should cancel orders)
        order = self.engine.liquidate()

        # Check all orders are cancelled
        pending_orders = self.engine.get_orders(OrderStatus.PENDING)
        self.assertEqual(len(pending_orders), 0)


class TestAlpacaPortfolioEngine(unittest.TestCase):
    """Test Alpaca-specific portfolio engines"""

    def test_alpaca_stock_configuration(self):
        """Test Alpaca stock engine has correct configuration"""
        engine = AlpacaStockPortfolioEngine(initial_balance=50000.0)

        # Check configuration
        self.assertEqual(engine.commission_rate, 0.0)
        self.assertEqual(engine.commission_pct, 0.0)
        self.assertEqual(engine.slippage_pct, 0.0002)
        self.assertEqual(engine.slippage_fixed, 0.01)
        self.assertFalse(engine.allow_shorts)  # Disabled by default
        self.assertEqual(engine.margin_requirement, 0.5)
        self.assertEqual(engine.symbol, "AAPL")

    def test_alpaca_stock_no_commission(self):
        """Test Alpaca stock has no commission"""
        engine = AlpacaStockPortfolioEngine(initial_balance=10000.0)

        # Execute a trade
        engine.place_order(OrderSide.BUY, 100, OrderType.MARKET)
        bar = Bar(
            open=50, high=50, low=50, close=50, volume=10000, timestamp=datetime.now()
        )
        trades = engine.process_bar(bar)

        # Should have no commission
        self.assertEqual(trades[0].commission, 0.0)

    def test_alpaca_crypto_configuration(self):
        """Test Alpaca crypto engine has correct configuration"""
        engine = AlpacaCryptoPortfolioEngine(initial_balance=50000.0)

        # Check configuration
        self.assertEqual(engine.commission_rate, 0.0)
        self.assertEqual(engine.commission_pct, 0.0025)  # 0.25%
        self.assertEqual(engine.slippage_pct, 0.0005)
        self.assertEqual(engine.slippage_fixed, 0.0)
        self.assertFalse(engine.allow_shorts)
        self.assertEqual(engine.symbol, "BTC/USD")

    def test_alpaca_crypto_commission(self):
        """Test Alpaca crypto commission calculation"""
        engine = AlpacaCryptoPortfolioEngine(initial_balance=10000.0)

        # Execute a trade
        engine.place_order(OrderSide.BUY, 0.1, OrderType.MARKET)  # 0.1 BTC
        bar = Bar(
            open=40000,
            high=40000,
            low=40000,
            close=40000,
            volume=100,
            timestamp=datetime.now(),
        )
        trades = engine.process_bar(bar)

        # Commission should be 0.25% of trade value
        # 0.1 BTC * $40,000 = $4,000 * 0.0025 = $10
        self.assertEqual(trades[0].commission, 10.0)


class TestCoinbasePortfolioEngine(unittest.TestCase):
    """Test Coinbase-specific portfolio engine"""

    def test_coinbase_configuration(self):
        """Test Coinbase engine has correct configuration"""
        engine = CoinbasePortfolioEngine(initial_balance=50000.0)

        # Check configuration
        self.assertEqual(engine.commission_rate, 0.0)
        self.assertEqual(engine.commission_pct, 0.006)
        self.assertEqual(engine.slippage_pct, 0.001)
        self.assertEqual(engine.slippage_fixed, 0.0)
        self.assertFalse(engine.allow_shorts)
        self.assertEqual(engine.symbol, "BTC-USD")

    def test_coinbase_commission(self):
        """Test Coinbase commission calculation"""
        engine = CoinbasePortfolioEngine(initial_balance=10000.0)

        # Execute a trade
        engine.place_order(OrderSide.BUY, 0.1, OrderType.MARKET)  # 0.1 BTC
        bar = Bar(
            open=30000,
            high=30000,
            low=30000,
            close=30000,
            volume=100,
            timestamp=datetime.now(),
        )
        trades = engine.process_bar(bar)

        # Commission should be 0.6% of trade value
        # 0.1 BTC * $30,000 = $3,000 * 0.006 = $18
        self.assertEqual(trades[0].commission, 18.0)

    def test_coinbase_no_shorting(self):
        """Test Coinbase doesn't allow shorting"""
        engine = CoinbasePortfolioEngine(initial_balance=10000.0)

        # Try to place short order
        order = engine.place_order(OrderSide.SHORT, 0.1, OrderType.MARKET)
        self.assertIsNone(order)  # Should be rejected


class TestPortfolioEngineIntegration(unittest.TestCase):
    """Integration test with generated data"""

    def test_trend_following_strategy(self):
        """Test a simple trend following strategy with known data"""
        engine = PortfolioEngine(
            initial_balance=100000.0,
            commission_pct=0.001,
            slippage_pct=0.0001,
            symbol="TEST",
        )

        # Generate trending market data
        start_date = datetime(2023, 1, 1)
        prices = []

        # Uptrend for 50 days
        for i in range(50):
            base_price = 100 + i * 0.5  # $0.50/day uptrend
            noise = (i % 3 - 1) * 0.2  # Small noise
            prices.append(base_price + noise)

        # Downtrend for 30 days
        for i in range(30):
            base_price = 125 - i * 0.8  # $0.80/day downtrend
            noise = (i % 3 - 1) * 0.3
            prices.append(base_price + noise)

        # Create bars
        bars = []
        for i, close_price in enumerate(prices):
            high = close_price + 0.5
            low = close_price - 0.5
            open_price = prices[i - 1] if i > 0 else close_price

            bar = Bar(
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=100000,
                timestamp=start_date + timedelta(days=i),
            )
            bars.append(bar)

        # Simple strategy: Buy after 5 days up, sell after 5 days down
        position_open = False

        for i, bar in enumerate(bars):
            if i < 5:
                engine.process_bar(bar)
                continue

            # Check last 5 days
            last_5_closes = [bars[j].close for j in range(i - 4, i + 1)]

            if not position_open:
                # Look for uptrend
                if all(last_5_closes[j] > last_5_closes[j - 1] for j in range(1, 5)):
                    order = engine.place_order(
                        side=OrderSide.BUY, quantity=500, order_type=OrderType.MARKET
                    )
                    position_open = True
            else:
                # Look for downtrend
                if all(last_5_closes[j] < last_5_closes[j - 1] for j in range(1, 5)):
                    order = engine.place_order(
                        side=OrderSide.SELL, quantity=500, order_type=OrderType.MARKET
                    )
                    position_open = False

            engine.process_bar(bar)

        # Force close any open position
        if position_open:
            engine.place_order(OrderSide.SELL, 500, OrderType.MARKET)
            engine.process_bar(bars[-1])

        # Get final metrics
        metrics = engine.get_performance_metrics()

        # Print detailed results
        print("\n=== Integration Test Results ===")
        print(f"Initial Balance: $100,000")
        print(f"Final Equity: ${engine.get_equity():,.2f}")
        print(
            f"Total Return: ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)"
        )
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Total Commission: ${metrics['total_commission']:,.2f}")

        # Verify the strategy made money in this trending market
        self.assertGreater(metrics["total_return"], 0)
        self.assertGreater(metrics["total_trades"], 0)

        # Verify calculations (with rounding tolerance)
        self.assertAlmostEqual(
            engine.get_equity(),
            engine.initial_balance + metrics["total_return"],
            places=2,
        )

        # Check position is closed
        self.assertIsNone(engine.get_position())

    def test_mean_reversion_strategy(self):
        """Test a mean reversion strategy with oscillating data"""
        engine = PortfolioEngine(
            initial_balance=50000.0,
            commission_pct=0.001,
            slippage_pct=0.0001,
            symbol="TEST",
        )

        # Generate oscillating market data
        start_date = datetime(2023, 1, 1)
        bars = []

        for i in range(100):
            # Sine wave pattern
            base_price = 100.0
            amplitude = 5.0
            period = 20  # 20-day cycle

            price = base_price + amplitude * math.sin(2 * math.pi * i / period)

            bar = Bar(
                open=price - 0.1,
                high=price + 0.2,
                low=price - 0.2,
                close=price,
                volume=50000,
                timestamp=start_date + timedelta(days=i),
            )
            bars.append(bar)

        # Mean reversion strategy
        position_open = False

        for i, bar in enumerate(bars):
            if i < 20:  # Need history for moving average
                engine.process_bar(bar)
                continue

            # Calculate 20-day moving average
            ma20 = sum(bars[j].close for j in range(i - 19, i + 1)) / 20

            # Buy when price is 3% below MA, sell when 3% above
            if not position_open and bar.close < ma20 * 0.97:
                engine.place_order(OrderSide.BUY, 200, OrderType.MARKET)
                position_open = True
            elif position_open and bar.close > ma20 * 1.03:
                engine.place_order(OrderSide.SELL, 200, OrderType.MARKET)
                position_open = False

            engine.process_bar(bar)

        # Close any open position
        if position_open:
            engine.place_order(OrderSide.SELL, 200, OrderType.MARKET)
            engine.process_bar(bars[-1])

        metrics = engine.get_performance_metrics()

        # Verify mean reversion worked in oscillating market
        self.assertGreater(metrics["total_trades"], 3)  # Should have multiple trades
        self.assertGreater(metrics["win_rate"], 0.5)  # Should be profitable


if __name__ == "__main__":
    unittest.main(verbosity=2)
