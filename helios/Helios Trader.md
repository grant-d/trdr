# Helios Trader: Product Requirements Document

| | |
| :--- | :--- |
| **Document Version:** | **3.0 (Final Composite)** |
| **Date:** | July 13, 2025 |
| **Status:** | **Final** |

---

## 1. Introduction

This document outlines the requirements for "Helios Trader," an adaptive, quantitative trading algorithm. The system's primary goal is to achieve sustained profitability by dynamically adapting its strategy to changing market conditions. It is designed to be run as a persistent local application, architected for resilience, multi-instrument capability, and future cloud deployment. The target environment is **Python**.

---

## 2. Goals and Success Metrics

* **Primary Goal**: To develop a fully automated trading algorithm that is resilient to application crashes, system reboots, and network outages.
* **Performance Goal**: Achieve a positive **Sortino Ratio > 1.0** (or **Calmar Ratio > 1.0**) over a walk-forward backtest of at least 24 months.
* **Architectural Goal**: Implement a modular design that supports multiple instruments (e.g., AAPL, BTC), multiple timeframes (e.g., 15m, 4h), and concurrent, isolated strategy experiments.

---

## 3. System Architecture

The system will be a **single application instance** that manages multiple, isolated **Trading Contexts**. To ensure resilience, the application's state is not held in memory but is persisted externally after every change.

* **Trading Context**: A self-contained runtime object holding all information and state for a single, unique strategy.
    * **Context ID Format**: `[instrument_id]_[timeframe]_[experiment_name]`
    * **Examples**: `AAPL_NASDAQ_4h_default`, `BTC_COINBASE_1h_AggressiveTrend`
* **Live Trading Engine**: The master controller that iterates through all active Trading Contexts. On startup, it re-hydrates all contexts from the state store. During runtime, it fetches the latest bar for each context, executes the logic, and immediately persists any state change.
* **Persistent State Store**: A swappable storage module. The initial implementation will use the local filesystem.

---

## 4. Core Feature Specifications

### 4.1. Data Service

* **Requirement 4.1.1: Multi-Source Handling**: Must handle both live brokerage feeds and local **CSV file** inputs. The expected CSV format is: `timestamp`, `open`, `high`, `low`, `close`, `volume`.
* **Requirement 4.1.2: Activity-Based Bars (Dollar Bars)**: Must be able to aggregate raw tick data (`price`, `volume`) into **Dollar Bars**, where a new bar is formed after a fixed amount of dollar value has been traded.
* **Requirement 4.1.3: "Get Next Bar" Logic**: Must handle various operational scenarios for each Trading Context:
    * **CSV Source**: Use a cursor (integer index) to iterate through a pre-loaded DataFrame.
    * **Live Source**: Poll the exchange for bars with a timestamp greater than the last processed bar's timestamp.
    * **Crash Recovery / Catch-up**: On startup, fetch all missing bars since the last processed timestamp and process them sequentially.
    * **New Instrument Backfill**: On creation, fetch the last `N` historical bars (e.g., 200) required for indicators to warm up.

### 4.2. Market State Score (MSS) Engine

* **Requirement 4.2.1: Factor Calculation**: Implement functions to calculate the three core factors, normalizing them to a consistent scale (e.g., -100 to +100):
    * **Trend/Momentum**: The slope of a linear regression line on closing prices.
    * **Volatility**: The Average True Range (ATR).
    * **Exhaustion/Mean Reversion**: `(last_close - SMA) / ATR`.
* **Requirement 4.2.2: MSS Calculation**: Combine factors using a weighted average: `MSS = (w_trend * Trend) + (w_vol * Volatility) + (w_exhaust * Exhaustion)`.

### 4.3. The Regime Playbook & Action Matrix

* **Requirement 4.3.1: Regime Classification**: Classify the market into a pre-defined regime (e.g., `Low-Volatility Bull Trend`, `High-Volatility Chop`) based on broader measures of trend and volatility.
* **Requirement 4.3.2: Instrument-Specific Playbooks**: Use different playbooks for each unique strategy. The playbook contains the optimal parameter sets (weights, lookbacks, thresholds) for each regime.
* **Requirement 4.3.3: Action Matrix**: A hard-coded ruleset maps the calculated MSS score to specific trading actions and risk rules.

| Live MSS Score | Action & Risk Management Rule |
| :--- | :--- |
| **> 60** (Strong Bull) | **Enter/Hold Long.** Set a dynamic trailing stop-loss at **2x ATR**. |
| **20 to 60** (Weak Bull) | **Hold Longs ONLY.** Tighten the trailing stop to **1x ATR**. |
| **-20 to 20** (Neutral) | **EXIT ALL POSITIONS.** |
| **-60 to -20** (Weak Bear) | **Hold Shorts ONLY.** Tighten the trailing stop to **1x ATR** above the price. |
| **< -60** (Strong Bear) | **Enter/Hold Short.** Set a dynamic trailing stop-loss at **2x ATR**. |

### 4.4. State Persistence and Recovery

* **Requirement 4.4.1: State Externalization**: The complete state of every Trading Context **must** be stored externally in a dedicated file named after its context ID (e.g., `./state/AAPL_NASDAQ_4h_default.json`).
* **Requirement 4.4.2: Atomic Write Operations**: File write operations must be atomic (e.g., write to a temp file then rename) to prevent data corruption.

---

## 5. Self-Improvement Specification (Offline Engine)

The system will feature two paths for self-improvement that can be used independently or in concert.

### 5.1. Path A: Algorithmic Optimization

This path uses traditional machine learning to tune the numeric parameters of the algorithm.

* **Requirement 5.1.1: Genetic Algorithm**: An offline GA process is used to find the most robust parameter sets for each regime of each instrument.
* **Requirement 5.1.2: Walk-Forward Optimization**: The GA will be applied in a walk-forward manner, using a rolling window of historical data to prevent curve-fitting.
* **Requirement 5.1.3: Fitness Function**: The GA's fitness function for evaluating parameters will be a risk-adjusted return metric, supporting both the **Sortino Ratio** and the **Calmar Ratio**.

### 5.2. Path B: LLM-Based Qualitative Analysis (Experimental)

This path uses a Large Language Model to introduce high-level reasoning and context that is invisible to price data alone.

* **Requirement 5.2.1: Sentiment Factor Generator**: An LLM can be prompted to analyze daily financial news headlines for a given instrument and return a sentiment score (e.g., -100 to +100). This score can be added as a new, forward-looking factor in the MSS calculation.
* **Requirement 5.2.2: Automated Performance Reviewer**: An LLM can be prompted to analyze the trade log for a given period, identify the single biggest weakness in a specific market state (e.g., "losing money in high-volatility chop"), and propose a human-readable hypothesis for a rule change to fix it.
* **Requirement 5.2.3: Economic Regime Adjuster**: An LLM can be prompted to analyze macroeconomic text (e.g., central bank minutes) and suggest adjustments to the algorithm's risk posture, such as widening the "Neutral / Chop" zone thresholds during times of high uncertainty.

---

## 6. User Interface (UI)

A basic, local web dashboard will be created using a framework like **Streamlit** or **Dash** to provide interactive management.

* **Requirement 6.1: Performance Dashboard**: A main view showing a table of all running strategies (`context_id`) and their key performance metrics (P&L, Sortino Ratio, Max Drawdown, etc.) for at-a-glance comparison.
* **Requirement 6.2: Strategy Drill-Down View**: A detailed view for a selected strategy, showing an equity curve chart, a price chart with trade entry/exit markers, and a viewer for the latest log entries.
* **Requirement 6.3: Interactive Management**: The UI must provide controls to:
    * **Create** a new strategy via a form.
    * **Pause/Resume** an active strategy with a toggle button.
    * **Edit** mutable parameters like capital allocation.
    * **Archive/Delete** a strategy.

---

## 7. Project Plan

### Phase 1: Proof of Concept (PoC)

* **Goal**: Validate the core MSS/Regime logic on a single instrument using a CSV file.
* **Deliverable**: A script or Jupyter Notebook that runs a backtest and proves the calculations and logic work.

### Phase 2: Minimum Viable Product (MVP)

* **Goal**: Build the full, robust local application capable of live trading and interactive management.
* **Core Engine Track**: Implement state management, the trading context manager, a live brokerage service, the GA, and the full live engine.
* **UI Track (Parallel)**: Develop the Streamlit/Dash application for read-only monitoring and interactive strategy management as defined in Section 6.
* **Deliverable**: A robust Python application that can be managed via a web UI and live paper-trade multiple strategies.

### Phase 3: Future Enhancements

* **Database**: Upgrade the persistence layer to a more robust database (e.g., SQLite or Redis).
* **Cloud Deployment**: Package the application for deployment in a cloud environment.
* **LLM Integration**: Fully implement and test the experimental LLM-based analysis methods from Section 5.2.

---

## Appendix A: Refinements & Design Notes

This appendix adds clarifying details, alternative approaches, and design rationale to the main Product Requirements Document.

### Refinement to Section 5.1 (Algorithmic Optimization)

As an alternative or complement to the full "Regime Playbook" model, the system should support a simpler self-improvement cadence based on **Shorter, Overlapping Optimization Cycles**.

* **Concept**: Instead of a complex playbook, the system uses a single parameter set that is re-optimized frequently.
* **Example Implementation**: Run the Genetic Algorithm every weekend on a rolling 90-day window of data. The "fittest" parameter set from this run is then deployed for the following week.
* **Benefit**: This approach is less complex than maintaining a multi-regime playbook and ensures the algorithm is always adapted to the most recent market character.

### Design Note for Section 4.1.2 (Activity-Based Bars)

The primary motivation for using **Dollar Bars** over traditional time-based bars is to standardize the "information content" of each bar.

* **Rationale**: A time-based bar (e.g., 1 hour) during a quiet period contains very little market activity and can create statistical noise. A bar during a high-activity period contains a huge amount of information. By forming a new bar only when a fixed amount of value (e.g., $20 million) has been traded, we ensure that every single bar in the dataset represents an **equal amount of economic commitment**. This leads to more reliable and stable calculations for the Trend, Volatility, and Exhaustion factors.

### Clarification for Section 4.3.2 (Instrument-Specific Playbooks)

The concept of an "Instrument-Specific Playbook" should be interpreted at its most granular level. A unique playbook must be generated for each unique **Trading Context ID**.

* **Explanation**: The optimal parameters are not just different between `BTC` and `AAPL`. They are also different for `AAPL` on a 15-minute chart versus a 4-hour chart.
* **Rule**: The Offline Optimization Engine must be run separately for each `instrument_timeframe` pair. The playbook for `AAPL_NASDAQ_4h_default` and `AAPL_NASDAQ_15m_default` will be entirely separate entities, optimized on different underlying data.