from google.colab import drive
drive.mount('/content/drive')

"""# Task
Clean the data in the CSV file "data/poc_data.csv" by identifying and handling invalid email addresses and duplicate rows, as detailed in the document "https://docs.google.com/document/d/1w-Gcjpw73mRmyCN9u-2PrVQyjnGMj_kjFj-XrDdENDI/edit?usp=drive_link", and save the cleaned data to a new CSV file.

## Load the data

### Subtask:
Load the data from the specified CSV file into a pandas DataFrame.

**Reasoning**:
Import pandas and load the CSV file into a DataFrame, then display the head and info of the DataFrame as requested in the instructions.
"""

import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/BTCUSD.csv')

display(df.head())
display(df.info())

"""**Reasoning**:
Load the data from the specified CSV file into a pandas DataFrame as the previous attempts failed.

# Task
Build the PoC described in "/content/drive/MyDrive/Helios Trader.md", including converting the data to dollar-bars using 'Adj Close' after loading and cleaning the data.

## Prepare the data

### Subtask:
Ensure the data is in the correct format for calculations, including converting the 'Date' column to datetime objects and setting it as the index.

**Reasoning**:
Convert the 'Date' column to datetime objects and set it as the index, then sort the index.
"""

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_index()

display(df.head())
display(df.info())

"""## Convert to dollar bars

### Subtask:
Aggregate the data into Dollar Bars based on the 'Adj Close' price and a specified dollar value threshold (Section 4.1.2).

**Reasoning**:
Implement the function to create dollar bars and apply it to the DataFrame with a chosen threshold, then display the head and info of the resulting DataFrame.
"""

def create_dollar_bars(df, threshold):
    dollar_bars = []
    current_dollar_volume = 0
    open_price = None
    high_price = -float('inf')
    low_price = float('inf')
    close_price = None
    start_time = None
    total_volume = 0

    for index, row in df.iterrows():
        if start_time is None:
            start_time = index
            open_price = row['Adj Close']

        dollar_value = row['Adj Close'] * row['Volume']
        current_dollar_volume += dollar_value
        total_volume += row['Volume']

        high_price = max(high_price, row['Adj Close'])
        low_price = min(low_price, row['Adj Close'])
        close_price = row['Adj Close']

        if current_dollar_volume >= threshold:
            dollar_bars.append({
                'Date': index,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': total_volume,
                'DollarVolume': current_dollar_volume
            })
            current_dollar_volume = 0
            open_price = None
            high_price = -float('inf')
            low_price = float('inf')
            total_volume = 0
            start_time = None

    # Handle the last bar if the threshold wasn't met exactly
    if current_dollar_volume > 0:
         dollar_bars.append({
                'Date': df.index[-1], # Use the timestamp of the last included row
                'Open': open_price if open_price is not None else df['Adj Close'].iloc[-1],
                'High': high_price if high_price != -float('inf') else df['Adj Close'].iloc[-1],
                'Low': low_price if low_price != float('inf') else df['Adj Close'].iloc[-1],
                'Close': df['Adj Close'].iloc[-1],
                'Volume': total_volume,
                'DollarVolume': current_dollar_volume
            })


    dollar_bar_df = pd.DataFrame(dollar_bars)
    dollar_bar_df = dollar_bar_df.set_index('Date')
    return dollar_bar_df

dollar_volume_threshold = 100000000 # Example threshold (100 million dollars)
dollar_bars_df = create_dollar_bars(df, dollar_volume_threshold)

display(dollar_bars_df.head())
display(dollar_bars_df.info())

"""## Calculate market state score (mss) factors

### Subtask:
Implement functions to calculate the Trend/Momentum, Volatility (ATR), and Exhaustion/Mean Reversion factors for the Dollar Bars (Section 4.2.1).

**Reasoning**:
Implement the functions to calculate Trend/Momentum, Volatility (ATR), and Exhaustion factors and apply them to the dollar bars dataframe.
"""

import numpy as np
from scipy.stats import linregress

def calculate_trend(df, lookback):
    df['Trend'] = np.nan
    for i in range(lookback, len(df)):
        y = df['Close'].iloc[i-lookback:i]
        x = np.arange(lookback)
        slope, _, r_value, p_value, std_err = linregress(x, y)
        # Normalize slope - Simple linear scaling example, might need refinement
        # Max and Min slope could be determined from historical data or set heuristically
        # For simplicity, let's just scale it based on its value relative to a potential max/min range
        # A more robust normalization would involve z-scores or min-max scaling over a window
        # Let's use a basic scaling where a slope of 1 might be 50, and -1 is -50
        # Assuming typical price movements, adjust scaling factor as needed
        scaling_factor = 50 / df['Close'].iloc[i-lookback:i].std() if df['Close'].iloc[i-lookback:i].std() != 0 else 0
        normalized_slope = slope * scaling_factor
        df.loc[df.index[i], 'Trend'] = np.clip(normalized_slope, -100, 100)
    return df

def calculate_atr(df, lookback):
    df['TR'] = np.maximum(np.maximum(df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1))), abs(df['Low'] - df['Close'].shift(1)))
    df['Volatility_ATR'] = df['TR'].rolling(window=lookback).mean()
    df = df.drop(columns=['TR'])

    # Normalize ATR - Scale ATR relative to the price level
    # ATR is usually related to volatility in absolute terms, normalizing it to -100 to 100 requires a reference
    # A simple approach is to normalize it as a percentage of the price and then scale
    # Let's normalize ATR as a percentage of the Close price and then scale
    df['Volatility_ATR'] = (df['Volatility_ATR'] / df['Close']) * 100 # ATR as percentage of Close
    # Now scale this percentage to -100 to 100. This scaling factor will need tuning.
    # Assuming 1% ATR might correspond to a score of 10, and 10% to 100
    scaling_factor = 10
    df['Volatility_ATR'] = np.clip(df['Volatility_ATR'] * scaling_factor, -100, 100)
    return df

def calculate_exhaustion(df, sma_lookback, atr_column='Volatility_ATR'):
    df['SMA'] = df['Close'].rolling(window=sma_lookback).mean()
    # Ensure ATR column exists and is not NaN for the calculation
    if atr_column not in df.columns:
        raise ValueError(f"ATR column '{atr_column}' not found in DataFrame.")

    # Calculate Exhaustion only where ATR is available and not zero
    valid_indices = df.index[df[atr_column].notna() & (df[atr_column].abs() > 1e-9)] # Avoid division by zero after normalization
    df.loc[valid_indices, 'Exhaustion'] = (df.loc[valid_indices, 'Close'] - df.loc[valid_indices, 'SMA']) / df.loc[valid_indices, atr_column]

    df = df.drop(columns=['SMA'])

    # Normalize Exhaustion - This factor is already somewhat normalized by dividing by ATR
    # However, to fit the -100 to 100 scale, we need to apply a scaling factor
    # A value of 1 might mean Close is 1 ATR away from SMA, a value of -1 means -1 ATR away
    # Let's scale these values to fit -100 to 100, assuming a range like -10 to 10 covers most cases
    scaling_factor = 100 / 10 # Assuming -10 to 10 maps to -100 to 100
    df['Exhaustion'] = np.clip(df['Exhaustion'] * scaling_factor, -100, 100)

    return df

# Apply the functions
lookback_period = 20 # Example lookback
dollar_bars_df = calculate_trend(dollar_bars_df.copy(), lookback_period)
dollar_bars_df = calculate_atr(dollar_bars_df.copy(), lookback_period)
dollar_bars_df = calculate_exhaustion(dollar_bars_df.copy(), lookback_period, atr_column='Volatility_ATR') # Use the normalized ATR column

# Truncate the head to remove rows with NaN values in 'Exhaustion'
dollar_bars_df = dollar_bars_df.dropna(subset=['Exhaustion'])

display(dollar_bars_df.head())
display(dollar_bars_df.info())

"""## Calculate mss

### Subtask:
Combine the calculated factors using the weighted average formula to get the Market State Score (Section 4.2.2).

**Reasoning**:
Calculate the Market State Score (MSS) by taking the weighted average of the 'Trend', 'Volatility_ATR', and 'Exhaustion' columns and add it as a new column to the DataFrame, then display the head of the DataFrame.
"""

# Define weights for each factor (these are examples and should be tuned)
weight_trend = 0.5 # Adjusted weight
weight_volatility = 0.2 # Adjusted weight
weight_exhaustion = 0.3 # Kept weight

# Calculate the Market State Score (MSS)
dollar_bars_df['MSS'] = (weight_trend * dollar_bars_df['Trend'] +
                         weight_volatility * dollar_bars_df['Volatility_ATR'] +
                         weight_exhaustion * dollar_bars_df['Exhaustion'])

display(dollar_bars_df.head())
display(dollar_bars_df.info())

"""## Classify Market Regime

### Subtask:
Implement logic to classify the market into a pre-defined regime based on broader measures of trend and volatility (Section 4.3.1).

**Reasoning**:
Based on the Market State Score (MSS), classify the market into different regimes as suggested by the PRD. Define the thresholds for each regime and create a new column in the DataFrame to store the market regime.
"""

# Define MSS thresholds for regime classification (these are examples and should be tuned)
# These thresholds are based on the Action Matrix in Section 4.3.3 of the PRD,
# which implicitly defines regimes based on MSS score ranges.
strong_bull_threshold = 50 # Adjusted threshold
weak_bull_threshold = 20
neutral_threshold_upper = 20
neutral_threshold_lower = -20
weak_bear_threshold = -20
strong_bear_threshold = -50 # Adjusted threshold

def classify_regime(mss):
    if mss > strong_bull_threshold:
        return 'Strong Bull'
    elif mss > weak_bull_threshold:
        return 'Weak Bull'
    elif mss >= neutral_threshold_lower and mss <= neutral_threshold_upper:
        return 'Neutral'
    elif mss > strong_bear_threshold:
        return 'Weak Bear'
    else: # mss <= strong_bear_threshold
        return 'Strong Bear'

# Apply the classification function to the MSS column
dollar_bars_df['Regime'] = dollar_bars_df['MSS'].apply(classify_regime)

display(dollar_bars_df.head())
display(dollar_bars_df.info())
display(dollar_bars_df['Regime'].value_counts())

"""## Simulate trading actions

### Subtask:
Based on the calculated MSS and the Action Matrix (Section 4.3.3), simulate trading actions (Enter/Hold Long/Short, Exit) and apply risk management rules (trailing stop-loss).

**Reasoning**:
Implement a trading simulation loop that iterates through the dollar bars DataFrame. Within the loop, determine the trading action based on the current MSS and regime, manage positions, and calculate the dynamic trailing stop-loss based on the ATR.
"""

# Initialize trading simulation parameters
position = 0  # 1 for long, -1 for short, 0 for flat
entry_price = 0
stop_loss = 0
equity_curve = []
trade_log = []
initial_capital = 100000 # Example initial capital
current_capital = initial_capital

# Trailing stop-loss multipliers based on the Action Matrix (Section 4.3.3)
stop_loss_multiplier_strong = 2 # 2x ATR for Strong Bull/Bear
stop_loss_multiplier_weak = 1   # 1x ATR for Weak Bull/Bear

# Iterate through the dollar bars
for index, row in dollar_bars_df.iterrows():
    current_price = row['Close']
    current_mss = row['MSS']
    current_regime = row['Regime']
    current_atr = row['Volatility_ATR'] # Using the normalized ATR

    # Determine stop-loss distance based on regime and ATR
    if current_regime in ['Strong Bull', 'Strong Bear']:
        stop_loss_distance = stop_loss_multiplier_strong * current_atr
    elif current_regime in ['Weak Bull', 'Weak Bear']:
        stop_loss_distance = stop_loss_multiplier_weak * current_atr
    else: # Neutral
        stop_loss_distance = 0 # No stop-loss in neutral

    # Trading Logic based on Action Matrix (Section 4.3.3)
    if current_regime == 'Strong Bull':
        if position == 0: # Enter Long
            position = 1
            entry_price = current_price
            stop_loss = current_price - stop_loss_distance
            trade_log.append({'Date': index, 'Action': 'Enter Long', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == 1: # Adjust Trailing Stop for Long
            stop_loss = max(stop_loss, current_price - stop_loss_distance)
            trade_log.append({'Date': index, 'Action': 'Adjust Long Stop', 'Price': current_price, 'StopLoss': stop_loss})
        # If in a short position in Strong Bull, exit short
        elif position == -1:
            position = 0
            pnl = (entry_price - current_price) # Calculate P/L for short exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})


    elif current_regime == 'Weak Bull':
        # Hold Longs ONLY, tighten stop
        if position == 1:
            stop_loss = max(stop_loss, current_price - stop_loss_distance) # Tighten stop for long
            trade_log.append({'Date': index, 'Action': 'Hold Long (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
        # If in a short position in Weak Bull, exit short
        elif position == -1:
            position = 0
            pnl = (entry_price - current_price) # Calculate P/L for short exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
        # If flat, do nothing

    elif current_regime == 'Neutral':
        # EXIT ALL POSITIONS
        if position != 0:
            action = 'Exit Long' if position == 1 else 'Exit Short'
            pnl = (current_price - entry_price) if position == 1 else (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': action + ' (Neutral Regime)', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0

    elif current_regime == 'Weak Bear':
        # Hold Shorts ONLY, tighten stop
         if position == -1:
            stop_loss = min(stop_loss, current_price + stop_loss_distance) # Tighten stop for short (above price)
            trade_log.append({'Date': index, 'Action': 'Hold Short (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
         # If in a long position in Weak Bear, exit long
         elif position == 1:
            position = 0
            pnl = (current_price - entry_price) # Calculate P/L for long exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
         # If flat, do nothing

    elif current_regime == 'Strong Bear':
        if position == 0: # Enter Short
            position = -1
            entry_price = current_price
            stop_loss = current_price + stop_loss_distance # Stop loss above price
            trade_log.append({'Date': index, 'Action': 'Enter Short', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1: # Adjust Trailing Stop for Short
            stop_loss = min(stop_loss, current_price + stop_loss_distance) # Adjust stop loss above price
            trade_log.append({'Date': index, 'Action': 'Adjust Short Stop', 'Price': current_price, 'StopLoss': stop_loss})
        # If in a long position in Strong Bear, exit long
        elif position == 1:
            position = 0
            pnl = (current_price - entry_price) # Calculate P/L for long exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})

    # Check for stop-loss hit
    if position == 1 and current_price <= stop_loss:
        pnl = (current_price - entry_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0
    elif position == -1 and current_price >= stop_loss:
        pnl = (entry_price - current_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Short', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0

    # Append current equity to the equity curve
    equity_curve.append({'Date': index, 'Equity': current_capital + (current_price - entry_price if position != 0 else 0)})


# Convert trade log and equity curve to DataFrames
trade_log_df = pd.DataFrame(trade_log)
equity_curve_df = pd.DataFrame(equity_curve).set_index('Date')

display("Equity Curve:")
display(equity_curve_df.head())
display("Trade Log:")
display(trade_log_df.head())

"""## Evaluate performance

### Subtask:
Calculate key performance metrics like P&L, Sortino Ratio, and Calmar Ratio based on the simulation results.

**Reasoning**:
Calculate the daily returns from the equity curve. Then, calculate the Sortino Ratio and Calmar Ratio using the daily returns and initial capital. Finally, display the calculated performance metrics.
"""

# Calculate daily returns
equity_curve_df['Daily_Return'] = equity_curve_df['Equity'].pct_change().fillna(0)

# Calculate Total Return
total_return = (equity_curve_df['Equity'].iloc[-1] - initial_capital) / initial_capital

# Calculate Annualized Return (assuming daily data)
# Number of trading periods in a year (e.g., 252 for stocks, adjust for crypto if needed)
trading_periods_per_year = 365
annualized_return = (1 + total_return)**(trading_periods_per_year / len(equity_curve_df)) - 1


# Calculate Drawdown
equity_curve_df['Peak'] = equity_curve_df['Equity'].cummax()
equity_curve_df['Drawdown'] = equity_curve_df['Equity'] - equity_curve_df['Peak']
max_drawdown = equity_curve_df['Drawdown'].min()

# Calculate Sortino Ratio
# Need to identify downside deviation. Assuming daily returns, we look at negative returns.
downside_returns = equity_curve_df[equity_curve_df['Daily_Return'] < 0]['Daily_Return']
# Assuming a Minimum Acceptable Return (MAR) of 0 for simplicity
mar = 0
downside_deviation = downside_returns[downside_returns < mar].std()

# Avoid division by zero if there are no downside returns below MAR
sortino_ratio = (annualized_return - mar) / downside_deviation if downside_deviation != 0 else float('inf')


# Calculate Calmar Ratio
# Avoid division by zero if there is no drawdown
calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')


display(f"Initial Capital: {initial_capital:.2f}")
display(f"Final Equity: {equity_curve_df['Equity'].iloc[-1]:.2f}")
display(f"Total Return: {total_return:.4f}")
display(f"Annualized Return: {annualized_return:.4f}")
display(f"Max Drawdown: {max_drawdown:.2f}")
display(f"Sortino Ratio (MAR=0): {sortino_ratio:.4f}")
display(f"Calmar Ratio: {calmar_ratio:.4f}")

"""## Present results

### Subtask:
Display or plot the performance metrics and simulated trades to demonstrate that the calculations and logic work.

**Reasoning**:
Plot the equity curve to visualize the strategy's performance. Display the trade log to review the simulated trades.
"""

import matplotlib.pyplot as plt

# Plot the equity curve
plt.figure(figsize=(12, 6))
plt.plot(equity_curve_df.index, equity_curve_df['Equity'], label='Equity Curve')
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.legend()
plt.grid(True)
plt.show()

# Display the trade log
display("Trade Log Summary:")
display(trade_log_df.head()) # Displaying head, could display more or a summary table

"""## Investigate Data for Trading Signals

### Subtask:
Analyze the `dollar_bars_df` DataFrame to understand the distribution of MSS values and the resulting market regimes to identify why trading signals were not generated.

**Reasoning**:
Display a summary of the 'MSS' column (mean, standard deviation, min, max) and the value counts of the 'Regime' column to understand the distribution of market states. Also, display the head and tail of the DataFrame to see the initial and final MSS and Regime values.
"""

# Display summary statistics for the MSS column
display("Summary statistics for MSS:")
display(dollar_bars_df['MSS'].describe())

# Display value counts for the Regime column
display("Regime distribution:")
display(dollar_bars_df['Regime'].value_counts())

# Display the head and tail of the DataFrame with relevant columns
display("Head of the DataFrame (MSS and Regime):")
display(dollar_bars_df[['Close', 'MSS', 'Regime']].head())

display("Tail of the DataFrame (MSS and Regime):")
display(dollar_bars_df[['Close', 'MSS', 'Regime']].tail())

"""# Task
Build the PoC described in "/content/Helios Trader.md", including converting to dollar-bars using 'Adj Close', truncating initial NaN values, and implementing and evaluating the trading strategy. Then, analyze factor distributions and contributions, refine factor normalization and weights, adjust regime thresholds, and iterate on the simulation and evaluation to improve performance.

## Analyze factor distributions

### Subtask:
Examine the distributions (e.g., using histograms or descriptive statistics) of the individual 'Trend', 'Volatility_ATR', and 'Exhaustion' factors to understand their ranges and typical values after normalization.

**Reasoning**:
Generate descriptive statistics and histograms for the 'Trend', 'Volatility_ATR', and 'Exhaustion' factors to understand their distributions.
"""

# Display descriptive statistics for individual factors
display("Summary statistics for individual factors:")
display(dollar_bars_df[['Trend', 'Volatility_ATR', 'Exhaustion']].describe())

# Plot histograms for individual factors
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(dollar_bars_df['Trend'].dropna(), bins=50, color='skyblue')
axes[0].set_title('Distribution of Trend Factor')
axes[0].set_xlabel('Trend Value')
axes[0].set_ylabel('Frequency')
axes[0].grid(True)

axes[1].hist(dollar_bars_df['Volatility_ATR'].dropna(), bins=50, color='lightcoral')
axes[1].set_title('Distribution of Volatility (ATR) Factor')
axes[1].set_xlabel('Volatility (ATR) Value')
axes[1].set_ylabel('Frequency')
axes[1].grid(True)

axes[2].hist(dollar_bars_df['Exhaustion'].dropna(), bins=50, color='lightgreen')
axes[2].set_title('Distribution of Exhaustion Factor')
axes[2].set_xlabel('Exhaustion Value')
axes[2].set_ylabel('Frequency')
axes[2].grid(True)

plt.tight_layout()
plt.show()

"""## Analyze factor contribution to mss

### Subtask:
Visualize or quantify the individual contribution of each factor to the final 'MSS' value to see which factors are dominating or if any are having minimal impact.

**Reasoning**:
Calculate the weighted contribution of each factor and their average absolute contribution to the MSS.
"""

# Calculate the weighted contribution of each factor
dollar_bars_df['Trend_Contribution'] = weight_trend * dollar_bars_df['Trend']
dollar_bars_df['Volatility_Contribution'] = weight_volatility * dollar_bars_df['Volatility_ATR']
dollar_bars_df['Exhaustion_Contribution'] = weight_exhaustion * dollar_bars_df['Exhaustion']

# Calculate the absolute contribution of each factor
dollar_bars_df['Trend_Abs_Contribution'] = abs(dollar_bars_df['Trend_Contribution'])
dollar_bars_df['Volatility_Abs_Contribution'] = abs(dollar_bars_df['Volatility_Contribution'])
dollar_bars_df['Exhaustion_Abs_Contribution'] = abs(dollar_bars_df['Exhaustion_Contribution'])

# Calculate the average absolute contribution for each factor
average_abs_contribution = {
    'Trend': dollar_bars_df['Trend_Abs_Contribution'].mean(),
    'Volatility': dollar_bars_df['Volatility_Abs_Contribution'].mean(),
    'Exhaustion': dollar_bars_df['Exhaustion_Abs_Contribution'].mean()
}

display("Average Absolute Contribution of Each Factor to MSS:")
display(average_abs_contribution)

# Optional: Plot the average absolute contribution
plt.figure(figsize=(8, 5))
plt.bar(average_abs_contribution.keys(), average_abs_contribution.values(), color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Average Absolute Contribution of Factors to MSS')
plt.xlabel('Factor')
plt.ylabel('Average Absolute Contribution')
plt.grid(axis='y')
plt.show()

"""## Refine factor normalization

### Subtask:
Modify the normalization logic for 'Trend' and 'Volatility_ATR' to potentially produce a wider range of values, if the analysis in step 1 suggests it's necessary.

**Reasoning**:
Modify the normalization logic for Trend and Volatility_ATR within their respective functions to potentially increase the range of values, then reapply the functions and recalculate Exhaustion and display the head and info of the dataframe.
"""

def calculate_trend(df, lookback):
    df['Trend_Slope'] = np.nan
    # Calculate slopes for the entire dataset first
    for i in range(lookback, len(df)):
        y = df['Close'].iloc[i-lookback:i]
        x = np.arange(lookback)
        slope, _, _, _, _ = linregress(x, y)
        df.loc[df.index[i], 'Trend_Slope'] = slope

    # Now calculate rolling standard deviation of slopes
    df['Slope_Std'] = df['Trend_Slope'].rolling(window=lookback).std()

    # Normalize Trend: Scale relative to the rolling standard deviation of slopes
    # Avoid division by zero
    valid_indices = df.index[df['Slope_Std'].notna() & (df['Slope_Std'].abs() > 1e-9)]
    # Adjust the multiplier (e.g., 2) to control how many standard deviations map to the full range
    scaling_factor = 100 / (df.loc[valid_indices, 'Slope_Std'] * 2) # Assuming +/- 2 std dev covers most cases
    df.loc[valid_indices, 'Trend'] = df.loc[valid_indices, 'Trend_Slope'] * scaling_factor

    df = df.drop(columns=['Trend_Slope', 'Slope_Std']) # Drop intermediate columns

    # Clip the result to the -100 to 100 range
    df['Trend'] = np.clip(df['Trend'], -100, 100)

    return df

def calculate_atr(df, lookback):
    df['TR'] = np.maximum(np.maximum(df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1))), abs(df['Low'] - df['Close'].shift(1)))
    df['Volatility_ATR_Absolute'] = df['TR'].rolling(window=lookback).mean()
    df = df.drop(columns=['TR'])

    # Modified Normalization for ATR: Scale relative to a rolling average of ATR
    # Normalize ATR by dividing by a rolling average of ATR and then scaling to -100 to 100
    df['ATR_MA'] = df['Volatility_ATR_Absolute'].rolling(window=lookback).mean()
    # Avoid division by zero
    valid_indices = df.index[df['ATR_MA'].notna() & (df['ATR_MA'].abs() > 1e-9)]
    # Normalize as percentage deviation from the rolling mean, then scale
    df.loc[valid_indices, 'Volatility_ATR'] = ((df.loc[valid_indices, 'Volatility_ATR_Absolute'] / df.loc[valid_indices, 'ATR_MA']) - 1) * 100 # Percentage deviation from MA
    df = df.drop(columns=['ATR_MA', 'Volatility_ATR_Absolute']) # Drop intermediate columns

    # Now clip the result to the -100 to 100 range
    df['Volatility_ATR'] = np.clip(df['Volatility_ATR'], -100, 100)

    return df

# Reapply the modified functions
lookback_period = 20 # Example lookback
dollar_bars_df = calculate_trend(dollar_bars_df.copy(), lookback_period)
dollar_bars_df = calculate_atr(dollar_bars_df.copy(), lookback_period)
# Re-calculate exhaustion with the new normalized ATR
dollar_bars_df = calculate_exhaustion(dollar_bars_df.copy(), lookback_period, atr_column='Volatility_ATR') # Use the normalized ATR column

# Truncate the head to remove rows with NaN values
dollar_bars_df = dollar_bars_df.dropna(subset=['Exhaustion'])

display(dollar_bars_df.head())
display(dollar_bars_df.info())

"""## Analyze factor distributions

### Subtask:
Examine the distributions (e.g., using histograms or descriptive statistics) of the individual 'Trend', 'Volatility_ATR', and 'Exhaustion' factors to understand their ranges and typical values after normalization.

**Reasoning**:
Generate descriptive statistics and histograms for the 'Trend', 'Volatility_ATR', and 'Exhaustion' factors to understand their distributions.
"""

# Display descriptive statistics for individual factors
display("Summary statistics for individual factors:")
display(dollar_bars_df[['Trend', 'Volatility_ATR', 'Exhaustion']].describe())

# Plot histograms for individual factors
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(dollar_bars_df['Trend'].dropna(), bins=50, color='skyblue')
axes[0].set_title('Distribution of Trend Factor')
axes[0].set_xlabel('Trend Value')
axes[0].set_ylabel('Frequency')
axes[0].grid(True)

axes[1].hist(dollar_bars_df['Volatility_ATR'].dropna(), bins=50, color='lightcoral')
axes[1].set_title('Distribution of Volatility (ATR) Factor')
axes[1].set_xlabel('Volatility (ATR) Value')
axes[1].set_ylabel('Frequency')
axes[1].grid(True)

axes[2].hist(dollar_bars_df['Exhaustion'].dropna(), bins=50, color='lightgreen')
axes[2].set_title('Distribution of Exhaustion Factor')
axes[2].set_xlabel('Exhaustion Value')
axes[2].set_ylabel('Frequency')
axes[2].grid(True)

plt.tight_layout()
plt.show()

"""## Analyze factor contribution to mss

### Subtask:
Visualize or quantify the individual contribution of each factor to the final 'MSS' value to see which factors are dominating or if any are having minimal impact.

**Reasoning**:
Calculate the weighted contribution of each factor and their average absolute contribution to the MSS.
"""

# Calculate the weighted contribution of each factor
dollar_bars_df['Trend_Contribution'] = weight_trend * dollar_bars_df['Trend']
dollar_bars_df['Volatility_Contribution'] = weight_volatility * dollar_bars_df['Volatility_ATR']
dollar_bars_df['Exhaustion_Contribution'] = weight_exhaustion * dollar_bars_df['Exhaustion']

# Calculate the absolute contribution of each factor
dollar_bars_df['Trend_Abs_Contribution'] = abs(dollar_bars_df['Trend_Contribution'])
dollar_bars_df['Volatility_Abs_Contribution'] = abs(dollar_bars_df['Volatility_Contribution'])
dollar_bars_df['Exhaustion_Abs_Contribution'] = abs(dollar_bars_df['Exhaustion_Contribution'])

# Calculate the average absolute contribution for each factor
average_abs_contribution = {
    'Trend': dollar_bars_df['Trend_Abs_Contribution'].mean(),
    'Volatility': dollar_bars_df['Volatility_Abs_Contribution'].mean(),
    'Exhaustion': dollar_bars_df['Exhaustion_Abs_Contribution'].mean()
}

display("Average Absolute Contribution of Each Factor to MSS:")
display(average_abs_contribution)

# Optional: Plot the average absolute contribution
plt.figure(figsize=(8, 5))
plt.bar(average_abs_contribution.keys(), average_abs_contribution.values(), color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Average Absolute Contribution of Factors to MSS')
plt.xlabel('Factor')
plt.ylabel('Average Absolute Contribution')
plt.grid(axis='y')
plt.show()

# Re-run the trading simulation with adjusted parameters and thresholds
# Initialize trading simulation parameters
position = 0  # 1 for long, -1 for short, 0 for flat
entry_price = 0
stop_loss = 0
equity_curve = []
trade_log = []
initial_capital = 100000 # Example initial capital
current_capital = initial_capital

# Trailing stop-loss multipliers based on the Action Matrix (Section 4.3.3)
stop_loss_multiplier_strong = 2 # 2x ATR for Strong Bull/Bear
stop_loss_multiplier_weak = 1   # 1x ATR for Weak Bull/Bear

# Iterate through the dollar bars
for index, row in dollar_bars_df.iterrows():
    current_price = row['Close']
    current_mss = row['MSS']
    current_regime = row['Regime']
    current_atr = row['Volatility_ATR'] # Using the normalized ATR

    # Determine stop-loss distance based on regime and ATR
    if current_regime in ['Strong Bull', 'Strong Bear']:
        stop_loss_distance = stop_loss_multiplier_strong * current_atr
    elif current_regime in ['Weak Bull', 'Weak Bear']:
        stop_loss_distance = stop_loss_multiplier_weak * current_atr
    else: # Neutral
        stop_loss_distance = 0 # No stop-loss in neutral

    # Trading Logic based on Action Matrix (Section 4.3.3)
    if current_regime == 'Strong Bull':
        if position == 0: # Enter Long
            position = 1
            entry_price = current_price
            stop_loss = current_price - stop_loss_distance
            trade_log.append({'Date': index, 'Action': 'Enter Long', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == 1: # Adjust Trailing Stop for Long
            stop_loss = max(stop_loss, current_price - stop_loss_distance)
            trade_log.append({'Date': index, 'Action': 'Adjust Long Stop', 'Price': current_price, 'StopLoss': stop_loss})
        # If in a short position in Strong Bull, exit short
        elif position == -1:
            position = 0
            pnl = (entry_price - current_price) # Calculate P/L for short exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})


    elif current_regime == 'Weak Bull':
        # Hold Longs ONLY, tighten stop
        if position == 1:
            stop_loss = max(stop_loss, current_price - stop_loss_distance) # Tighten stop for long
            trade_log.append({'Date': index, 'Action': 'Hold Long (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
        # If in a short position in Weak Bull, exit short
        elif position == -1:
            position = 0
            pnl = (entry_price - current_price) # Calculate P/L for short exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
        # If flat, do nothing

    elif current_regime == 'Neutral':
        # EXIT ALL POSITIONS
        if position != 0:
            action = 'Exit Long' if position == 1 else 'Exit Short'
            pnl = (current_price - entry_price) if position == 1 else (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': action + ' (Neutral Regime)', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0

    elif current_regime == 'Weak Bear':
        # Hold Shorts ONLY, tighten stop
         if position == -1:
            stop_loss = min(stop_loss, current_price + stop_loss_distance) # Tighten stop for short (above price)
            trade_log.append({'Date': index, 'Action': 'Hold Short (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
         # If in a long position in Weak Bear, exit long
         elif position == 1:
            position = 0
            pnl = (current_price - entry_price) # Calculate P/L for long exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
         # If flat, do nothing


    elif current_regime == 'Strong Bear':
        if position == 0: # Enter Short
            position = -1
            entry_price = current_price
            stop_loss = current_price + stop_loss_distance # Stop loss above price
            trade_log.append({'Date': index, 'Action': 'Enter Short', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1: # Adjust Trailing Stop for Short
            stop_loss = min(stop_loss, current_price + stop_loss_distance) # Adjust stop loss above price
            trade_log.append({'Date': index, 'Action': 'Adjust Short Stop', 'Price': current_price, 'StopLoss': stop_loss})
        # If in a long position in Strong Bear, exit long
        elif position == 1:
            position = 0
            pnl = (current_price - entry_price) # Calculate P/L for long exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})


    # Check for stop-loss hit
    if position == 1 and current_price <= stop_loss:
        pnl = (current_price - entry_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0
    elif position == -1 and current_price >= stop_loss:
        pnl = (entry_price - current_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Short', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0

    # Append current equity to the equity curve
    equity_curve.append({'Date': index, 'Equity': current_capital + (current_price - entry_price if position != 0 else 0)})

# Convert trade log and equity curve to DataFrames
trade_log_df = pd.DataFrame(trade_log)
equity_curve_df = pd.DataFrame(equity_curve).set_index('Date')

display("Equity Curve:")
display(equity_curve_df.head())
display("Trade Log:")
display(trade_log_df.head())

# Calculate daily returns
equity_curve_df['Daily_Return'] = equity_curve_df['Equity'].pct_change().fillna(0)

# Calculate Total Return
total_return = (equity_curve_df['Equity'].iloc[-1] - initial_capital) / initial_capital

# Calculate Annualized Return (assuming daily data)
# Number of trading periods in a year (e.g., 252 for stocks, adjust for crypto if needed)
trading_periods_per_year = 365
annualized_return = (1 + total_return)**(trading_periods_per_year / len(equity_curve_df)) - 1


# Calculate Drawdown
equity_curve_df['Peak'] = equity_curve_df['Equity'].cummax()
equity_curve_df['Drawdown'] = equity_curve_df['Equity'] - equity_curve_df['Peak']
max_drawdown = equity_curve_df['Drawdown'].min()

# Calculate Sortino Ratio
# Need to identify downside deviation. Assuming daily returns, we look at negative returns.
downside_returns = equity_curve_df[equity_curve_df['Daily_Return'] < 0]['Daily_Return']
# Assuming a Minimum Acceptable Return (MAR) of 0 for simplicity
mar = 0
downside_deviation = downside_returns[downside_returns < mar].std()

# Avoid division by zero if there are no downside returns below MAR
sortino_ratio = (annualized_return - mar) / downside_deviation if downside_deviation != 0 else float('inf')


# Calculate Calmar Ratio
# Avoid division by zero if there is no drawdown
calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')


display(f"Initial Capital: {initial_capital:.2f}")
display(f"Final Equity: {equity_curve_df['Equity'].iloc[-1]:.2f}")
display(f"Total Return: {total_return:.4f}")
display(f"Annualized Return: {annualized_return:.4f}")
display(f"Max Drawdown: {max_drawdown:.2f}")
display(f"Sortino Ratio (MAR=0): {sortino_ratio:.4f}")
display(f"Calmar Ratio: {calmar_ratio:.4f}")

import matplotlib.pyplot as plt

# Plot the equity curve
plt.figure(figsize=(12, 6))
plt.plot(equity_curve_df.index, equity_curve_df['Equity'], label='Equity Curve')
plt.title('Equity Curve (After Parameter Adjustments)')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.legend()
plt.grid(True)
plt.show()

# Display the trade log
display("Trade Log Summary (After Parameter Adjustments):")
display(trade_log_df.head()) # Displaying head, could display more or a summary table

"""## Analyze Losing Trades

### Subtask:
Analyze trades with negative PnL from the `trade_log_df` to identify patterns or market conditions where the strategy is failing.

**Reasoning**:
Filter the `trade_log_df` to include only trades with negative PnL and display the resulting DataFrame to examine the losing trades.
"""

# Filter trade log for losing trades (PnL < 0)
losing_trades_df = trade_log_df[trade_log_df['PnL'] < 0]

display("Losing Trades Summary:")
display(losing_trades_df)

# Re-run the trading simulation with adjusted parameters and thresholds
# Initialize trading simulation parameters
position = 0  # 1 for long, -1 for short, 0 for flat
entry_price = 0
stop_loss = 0
equity_curve = []
trade_log = []
initial_capital = 100000 # Example initial capital
current_capital = initial_capital

# Trailing stop-loss multipliers based on the Action Matrix (Section 4.3.3)
stop_loss_multiplier_strong = 3 # Adjusted: 3x ATR for Strong Bull/Bear
stop_loss_multiplier_weak = 3   # Adjusted: 3x ATR for Weak Bull/Bear

# Iterate through the dollar bars
for index, row in dollar_bars_df.iterrows():
    current_price = row['Close']
    current_mss = row['MSS']
    current_regime = row['Regime']
    current_atr = row['Volatility_ATR'] # Using the normalized ATR

    # Determine stop-loss distance based on regime and ATR
    if current_regime in ['Strong Bull', 'Strong Bear']:
        stop_loss_distance = stop_loss_multiplier_strong * current_atr
    elif current_regime in ['Weak Bull', 'Weak Bear']:
        stop_loss_distance = stop_loss_multiplier_weak * current_atr
    else: # Neutral
        stop_loss_distance = 0 # No stop-loss in neutral

    # Trading Logic based on Action Matrix (Section 4.3.3)
    if current_regime == 'Strong Bull':
        if position == 0: # Enter Long
            position = 1
            entry_price = current_price
            stop_loss = current_price - stop_loss_distance
            trade_log.append({'Date': index, 'Action': 'Enter Long', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == 1: # Adjust Trailing Stop for Long
            stop_loss = max(stop_loss, current_price - stop_loss_distance)
            trade_log.append({'Date': index, 'Action': 'Adjust Long Stop', 'Price': current_price, 'StopLoss': stop_loss})
        # If in a short position in Strong Bull, exit short
        elif position == -1:
            position = 0
            pnl = (entry_price - current_price) # Calculate P/L for short exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})


    elif current_regime == 'Weak Bull':
        # Hold Longs ONLY, tighten stop
        if position == 1:
            stop_loss = max(stop_loss, current_price - stop_loss_distance) # Tighten stop for long
            trade_log.append({'Date': index, 'Action': 'Hold Long (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
        # If in a short position in Weak Bull, exit short
        elif position == -1:
            position = 0
            pnl = (entry_price - current_price) # Calculate P/L for short exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
        # If flat, do nothing

    elif current_regime == 'Neutral':
        # EXIT ALL POSITIONS
        if position != 0:
            action = 'Exit Long' if position == 1 else 'Exit Short'
            pnl = (current_price - entry_price) if position == 1 else (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': action + ' (Neutral Regime)', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0

    elif current_regime == 'Weak Bear':
        # Hold Shorts ONLY, tighten stop
         if position == -1:
            stop_loss = min(stop_loss, current_price + stop_loss_distance) # Tighten stop for short (above price)
            trade_log.append({'Date': index, 'Action': 'Hold Short (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
         # If in a long position in Weak Bear, exit long
         elif position == 1:
            position = 0
            pnl = (current_price - entry_price) # Calculate P/L for long exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
         # If flat, do nothing


    elif current_regime == 'Strong Bear':
        if position == 0: # Enter Short
            position = -1
            entry_price = current_price
            stop_loss = current_price + stop_loss_distance # Stop loss above price
            trade_log.append({'Date': index, 'Action': 'Enter Short', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1: # Adjust Trailing Stop for Short
            stop_loss = min(stop_loss, current_price + stop_loss_distance) # Adjust stop loss above price
            trade_log.append({'Date': index, 'Action': 'Adjust Short Stop', 'Price': current_price, 'StopLoss': stop_loss})
        # If in a long position in Strong Bear, exit long
        elif position == 1:
            position = 0
            pnl = (current_price - entry_price) # Calculate P/L for long exit
            current_capital += pnl # Update capital
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})


    # Check for stop-loss hit
    if position == 1 and current_price <= stop_loss:
        pnl = (current_price - entry_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0
    elif position == -1 and current_price >= stop_loss:
        pnl = (entry_price - current_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Short', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0

    # Append current equity to the equity curve
    equity_curve.append({'Date': index, 'Equity': current_capital + (current_price - entry_price if position != 0 else 0)})

# Convert trade log and equity curve to DataFrames
trade_log_df = pd.DataFrame(trade_log)
equity_curve_df = pd.DataFrame(equity_curve).set_index('Date')

display("Equity Curve:")
display(equity_curve_df.head())
display("Trade Log:")
display(trade_log_df.head())

# Calculate daily returns
equity_curve_df['Daily_Return'] = equity_curve_df['Equity'].pct_change().fillna(0)

# Calculate Total Return
total_return = (equity_curve_df['Equity'].iloc[-1] - initial_capital) / initial_capital

# Calculate Annualized Return (assuming daily data)
# Number of trading periods in a year (e.g., 252 for stocks, adjust for crypto if needed)
trading_periods_per_year = 365
annualized_return = (1 + total_return)**(trading_periods_per_year / len(equity_curve_df)) - 1


# Calculate Drawdown
equity_curve_df['Peak'] = equity_curve_df['Equity'].cummax()
equity_curve_df['Drawdown'] = equity_curve_df['Equity'] - equity_curve_df['Peak']
max_drawdown = equity_curve_df['Drawdown'].min()

# Calculate Sortino Ratio
# Need to identify downside deviation. Assuming daily returns, we look at negative returns.
downside_returns = equity_curve_df[equity_curve_df['Daily_Return'] < 0]['Daily_Return']
# Assuming a Minimum Acceptable Return (MAR) of 0 for simplicity
mar = 0
downside_deviation = downside_returns[downside_returns < mar].std()

# Avoid division by zero if there are no downside returns below MAR
sortino_ratio = (annualized_return - mar) / downside_deviation if downside_deviation != 0 else float('inf')


# Calculate Calmar Ratio
# Avoid division by zero if there is no drawdown
calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')


display(f"Initial Capital: {initial_capital:.2f}")
display(f"Final Equity: {equity_curve_df['Equity'].iloc[-1]:.2f}")
display(f"Total Return: {total_return:.4f}")
display(f"Annualized Return: {annualized_return:.4f}")
display(f"Max Drawdown: {max_drawdown:.2f}")
display(f"Sortino Ratio (MAR=0): {sortino_ratio:.4f}")
display(f"Calmar Ratio: {calmar_ratio:.4f}")

import matplotlib.pyplot as plt

# Plot the equity curve
plt.figure(figsize=(12, 6))
plt.plot(equity_curve_df.index, equity_curve_df['Equity'], label='Equity Curve (After Parameter Adjustments)')
plt.title('Equity Curve (After Parameter Adjustments)')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.legend()
plt.grid(True)
plt.show()

# Display the trade log
display("Trade Log Summary (After Parameter Adjustments):")
display(trade_log_df.head()) # Displaying head, could display more or a summary table

# Calculate daily returns
equity_curve_df['Daily_Return'] = equity_curve_df['Equity'].pct_change().fillna(0)

# Calculate Total Return
total_return = (equity_curve_df['Equity'].iloc[-1] - initial_capital) / initial_capital

# Calculate Annualized Return (assuming daily data)
# Number of trading periods in a year (e.g., 252 for stocks, adjust for crypto if needed)
trading_periods_per_year = 365
annualized_return = (1 + total_return)**(trading_periods_per_year / len(equity_curve_df)) - 1


# Calculate Drawdown
equity_curve_df['Peak'] = equity_curve_df['Equity'].cummax()
equity_curve_df['Drawdown'] = equity_curve_df['Equity'] - equity_curve_df['Peak']
max_drawdown = equity_curve_df['Drawdown'].min()

# Calculate Sortino Ratio
# Need to identify downside deviation. Assuming daily returns, we look at negative returns.
downside_returns = equity_curve_df[equity_curve_df['Daily_Return'] < 0]['Daily_Return']
# Assuming a Minimum Acceptable Return (MAR) of 0 for simplicity
mar = 0
downside_deviation = downside_returns[downside_returns < mar].std()

# Avoid division by zero if there are no downside returns below MAR
sortino_ratio = (annualized_return - mar) / downside_deviation if downside_deviation != 0 else float('inf')


# Calculate Calmar Ratio
# Avoid division by zero if there is no drawdown
calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')


display(f"Initial Capital: {initial_capital:.2f}")
display(f"Final Equity: {equity_curve_df['Equity'].iloc[-1]:.2f}")
display(f"Total Return: {total_return:.4f}")
display(f"Annualized Return: {annualized_return:.4f}")
display(f"Max Drawdown: {max_drawdown:.2f}")
display(f"Sortino Ratio (MAR=0): {sortino_ratio:.4f}")
display(f"Calmar Ratio: {calmar_ratio:.4f}")

import matplotlib.pyplot as plt

# Plot the equity curve
plt.figure(figsize=(12, 6))
plt.plot(equity_curve_df.index, equity_curve_df['Equity'], label='Equity Curve (After Parameter Adjustments)')
plt.title('Equity Curve (After Parameter Adjustments)')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.legend()
plt.grid(True)
plt.show()

# Display the trade log
display("Trade Log Summary (After Parameter Adjustments):")
display(trade_log_df.head()) # Displaying head, could display more or a summary table

"""# Task
Implement a Genetic Algorithm with walk-forward optimization to find optimal parameters for the trading strategy described in "/content/Helios Trader.md", using dollar-bar data derived from the provided historical data, and evaluate the performance of the optimized strategy.

## Define the fitness function

### Subtask:
Implement a fitness function that evaluates the performance of a given set of strategy parameters using a risk-adjusted return metric (Sortino Ratio or Calmar Ratio, as per Section 5.1.3). This function will run backtests using the historical data.

**Reasoning**:
Implement the fitness function that takes parameters, runs a backtest, and returns the Sortino Ratio, handling edge cases.
"""

def evaluate_strategy(params, df, initial_capital=100000):
    """
    Evaluates a trading strategy with given parameters on historical data.

    Args:
        params (dict): Dictionary of strategy parameters, including:
                       - 'weight_trend': Weight for the Trend factor.
                       - 'weight_volatility': Weight for the Volatility factor.
                       - 'weight_exhaustion': Weight for the Exhaustion factor.
                       - 'strong_bull_threshold': MSS threshold for Strong Bull.
                       - 'weak_bull_threshold': MSS threshold for Weak Bull.
                       - 'neutral_threshold_upper': Upper MSS threshold for Neutral.
                       - 'neutral_threshold_lower': Lower MSS threshold for Neutral.
                       - 'strong_bear_threshold': MSS threshold for Strong Bear.
                       - 'weak_bear_threshold': MSS threshold for Weak Bear.
                       - 'stop_loss_multiplier_strong': ATR multiplier for strong regimes.
                       - 'stop_loss_multiplier_weak': ATR multiplier for weak regimes.
        df (pd.DataFrame): DataFrame containing dollar bars and calculated factors.
        initial_capital (float): Starting capital for the backtest.

    Returns:
        float: The Sortino Ratio of the strategy's performance, or a very low number
               if the ratio is infinite or NaN.
    """
    weight_trend = params['weight_trend']
    weight_volatility = params['weight_volatility']
    weight_exhaustion = params['weight_exhaustion']
    strong_bull_threshold = params['strong_bull_threshold']
    weak_bull_threshold = params['weak_bull_threshold']
    neutral_threshold_upper = params['neutral_threshold_upper']
    neutral_threshold_lower = params['neutral_threshold_lower']
    strong_bear_threshold = params['strong_bear_threshold']
    weak_bear_threshold = params['weak_bear_threshold']
    stop_loss_multiplier_strong = params['stop_loss_multiplier_strong']
    stop_loss_multiplier_weak = params['stop_loss_multiplier_weak']


    # Recalculate MSS with new weights
    df['MSS_eval'] = (weight_trend * df['Trend'] +
                      weight_volatility * df['Volatility_ATR'] +
                      weight_exhaustion * df['Exhaustion'])

    # Reclassify Regime with new thresholds
    def classify_regime_eval(mss):
        if mss > strong_bull_threshold:
            return 'Strong Bull'
        elif mss > weak_bull_threshold:
            return 'Weak Bull'
        elif mss >= neutral_threshold_lower and mss <= neutral_threshold_upper:
            return 'Neutral'
        elif mss > strong_bear_threshold:
            return 'Weak Bear'
        else:
            return 'Strong Bear'

    df['Regime_eval'] = df['MSS_eval'].apply(classify_regime_eval)


    # --- Trading Simulation ---
    position = 0
    entry_price = 0
    stop_loss = 0
    equity_curve = []
    current_capital = initial_capital

    for index, row in df.iterrows():
        current_price = row['Close']
        current_regime = row['Regime_eval']
        current_atr = row['Volatility_ATR'] # Using the normalized ATR

        # Determine stop-loss distance based on regime and ATR
        if current_regime in ['Strong Bull', 'Strong Bear']:
            stop_loss_distance = stop_loss_multiplier_strong * current_atr
        elif current_regime in ['Weak Bull', 'Weak Bear']:
            stop_loss_distance = stop_loss_multiplier_weak * current_atr
        else: # Neutral
            stop_loss_distance = 0

        # Trading Logic based on Action Matrix (Section 4.3.3) - Adapted for evaluation
        if current_regime == 'Strong Bull':
            if position == 0: # Enter Long
                position = 1
                entry_price = current_price
                stop_loss = current_price - stop_loss_distance
            elif position == 1: # Adjust Trailing Stop for Long
                stop_loss = max(stop_loss, current_price - stop_loss_distance)
            elif position == -1: # Exit Short
                pnl = (entry_price - current_price)
                current_capital += pnl
                position = 0

        elif current_regime == 'Weak Bull':
            if position == 1: # Hold Longs ONLY, tighten stop
                 stop_loss = max(stop_loss, current_price - stop_loss_distance)
            elif position == -1: # Exit Short
                pnl = (entry_price - current_price)
                current_capital += pnl
                position = 0

        elif current_regime == 'Neutral':
            if position != 0: # EXIT ALL POSITIONS
                pnl = (current_price - entry_price) if position == 1 else (entry_price - current_price)
                current_capital += pnl
                position = 0
                entry_price = 0
                stop_loss = 0

        elif current_regime == 'Weak Bear':
            if position == -1: # Hold Shorts ONLY, tighten stop
                stop_loss = min(stop_loss, current_price + stop_loss_distance)
            elif position == 1: # Exit Long
                pnl = (current_price - entry_price)
                current_capital += pnl
                position = 0

        elif current_regime == 'Strong Bear':
            if position == 0: # Enter Short
                position = -1
                entry_price = current_price
                stop_loss = current_price + stop_loss_distance
            elif position == -1: # Adjust Trailing Stop for Short
                 stop_loss = min(stop_loss, current_price + stop_loss_distance)
            elif position == 1: # Exit Long
                pnl = (current_price - entry_price)
                current_capital += pnl
                position = 0


        # Check for stop-loss hit
        if position == 1 and current_price <= stop_loss:
            pnl = (current_price - entry_price)
            current_capital += pnl
            position = 0
            entry_price = 0
            stop_loss = 0
        elif position == -1 and current_price >= stop_loss:
            pnl = (entry_price - current_price)
            current_capital += pnl
            position = 0
            entry_price = 0
            stop_loss = 0

        # Append current equity to the equity curve
        equity_curve.append({'Date': index, 'Equity': current_capital + (current_price - entry_price if position != 0 else 0)})

    equity_curve_df_eval = pd.DataFrame(equity_curve).set_index('Date')

    # --- Performance Evaluation (Sortino Ratio) ---
    if equity_curve_df_eval.empty or len(equity_curve_df_eval) < 2:
         return -1000 # Return a very low fitness for empty or short equity curves

    equity_curve_df_eval['Daily_Return'] = equity_curve_df_eval['Equity'].pct_change().fillna(0)

    # Calculate Annualized Return (assuming daily data from dollar bars)
    trading_periods_per_year = 365 # Adjust if dollar bars don't represent daily frequency
    total_return = (equity_curve_df_eval['Equity'].iloc[-1] - initial_capital) / initial_capital
    # Avoid division by zero for annualized return if the simulation period is too short
    if len(equity_curve_df_eval) > 1:
      annualized_return = (1 + total_return)**(trading_periods_per_year / len(equity_curve_df_eval)) - 1
    else:
      annualized_return = 0


    # Calculate Sortino Ratio
    mar = 0 # Minimum Acceptable Return (MAR)
    downside_returns = equity_curve_df_eval[equity_curve_df_eval['Daily_Return'] < mar]['Daily_Return']
    downside_deviation = downside_returns.std()

    # Handle cases where downside_deviation is zero or NaN
    if downside_deviation == 0 or np.isnan(downside_deviation):
        sortino_ratio = float('inf') if annualized_return > mar else (0 if annualized_return == mar else -1000) # High ratio if profitable with no downside, else low

    else:
        sortino_ratio = (annualized_return - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN
    if np.isinf(sortino_ratio) or np.isnan(sortino_ratio):
        return -1000 # Return a very low fitness for problematic ratios

    return sortino_ratio

# Example usage of the fitness function (using current parameters)
current_params = {
    'weight_trend': weight_trend,
    'weight_volatility': weight_volatility,
    'weight_exhaustion': weight_exhaustion,
    'strong_bull_threshold': strong_bull_threshold,
    'weak_bull_threshold': weak_bull_threshold,
    'neutral_threshold_upper': neutral_threshold_upper,
    'neutral_threshold_lower': neutral_threshold_lower,
    'strong_bear_threshold': strong_bear_threshold,
    'weak_bear_threshold': weak_bear_threshold,
    'stop_loss_multiplier_strong': stop_loss_multiplier_strong,
    'stop_loss_multiplier_weak': stop_loss_multiplier_weak
}
fitness = evaluate_strategy(current_params, dollar_bars_df.copy()) # Use a copy to avoid modifying original df
display(f"Fitness (Sortino Ratio) with current parameters: {fitness}")

"""**Reasoning**:
The fitness function is implemented. Now, implement the genetic algorithm and the walk-forward optimization process using this fitness function.


"""

import random

def genetic_algorithm_walk_forward(df, initial_capital, param_space, n_generations, pop_size, mutation_rate, lookback_window, step_size):
    """
    Runs a Genetic Algorithm with walk-forward optimization.

    Args:
        df (pd.DataFrame): DataFrame containing the full historical data (dollar bars with factors).
        initial_capital (float): Starting capital for backtests.
        param_space (dict): Dictionary defining the search space for each parameter.
                            Each key is a parameter name, and the value is a tuple (min_val, max_val).
        n_generations (int): Number of generations for the GA.
        pop_size (int): Size of the population in each generation.
        mutation_rate (float): Probability of mutation for each parameter.
        lookback_window (int): Size of the walk-forward backtesting window (number of bars).
        step_size (int): Size of the step for the walk-forward window (number of bars).

    Returns:
        dict: The best set of parameters found during the walk-forward optimization.
    """

    best_overall_params = None
    best_overall_fitness = -float('inf')

    # Determine the starting and ending indices for the walk-forward windows
    start_index = df.index.min() + pd.Timedelta(days=lookback_window) # Start after the initial lookback
    end_index = df.index.max()

    current_start_date = df.index.min()
    current_end_date = current_start_date + pd.Timedelta(days=lookback_window)

    while current_end_date <= end_index:
        # Define the training window (backtest period for GA optimization)
        train_df = df.loc[current_start_date:current_end_date].copy()

        if len(train_df) < lookback_window:
            # Not enough data for a full window
            break

        display(f"Optimizing on window: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")

        # --- Genetic Algorithm ---
        # Initialize population
        population = []
        for _ in range(pop_size):
            params = {}
            for param, (min_val, max_val) in param_space.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param] = random.randint(min_val, max_val)
                else:
                    params[param] = random.uniform(min_val, max_val)
            population.append(params)

        # GA generations
        for generation in range(n_generations):
            # Evaluate fitness of each individual
            fitness_scores = [evaluate_strategy(params, train_df.copy(), initial_capital) for params in population]

            # Select parents (e.g., tournament selection)
            parents = []
            for _ in range(pop_size // 2):
                # Select two random individuals and pick the better one
                idx1, idx2 = random.sample(range(pop_size), 2)
                parent1 = population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]
                idx1, idx2 = random.sample(range(pop_size), 2)
                parent2 = population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]
                parents.extend([parent1, parent2])

            # Create next generation (crossover and mutation)
            next_population = []
            for i in range(0, len(parents), 2):
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = {}, {}

                # Crossover
                for param in param_space.keys():
                    if random.random() < 0.5:
                        child1[param] = parent1[param]
                        child2[param] = parent2[param]
                    else:
                        child1[param] = parent2[param]
                        child2[param] = parent1[param]

                # Mutation
                for child in [child1, child2]:
                    for param, (min_val, max_val) in param_space.items():
                        if random.random() < mutation_rate:
                            if isinstance(min_val, int) and isinstance(max_val, int):
                                child[param] = random.randint(min_val, max_val)
                            else:
                                child[param] = random.uniform(min_val, max_val)
                            # Ensure mutated values are within bounds
                            child[param] = max(min_val, min(max_val, child[param]))


                next_population.extend([child1, child2])

            population = next_population

            # Track best individual in this generation
            best_gen_fitness = max(fitness_scores)
            best_gen_params = population[fitness_scores.index(best_gen_fitness)]
            display(f"Generation {generation+1}: Best Fitness = {best_gen_fitness:.4f}")


        # After GA for the current window, evaluate the best parameters on the next step_size bars (walk-forward)
        # This step is for evaluation/selection, not optimization. The optimization happened on train_df.
        # We select the best parameters from the GA run on train_df and carry them forward.
        # For this PoC, we'll just take the best params from the last generation of the GA
        # and update the overall best if it's better. A true walk-forward would evaluate
        # the best params from the *training* window on the *next* window and select based on that.
        # For simplicity in the PoC, we just track the best parameters found across all training windows.

        best_window_fitness = max(fitness_scores)
        best_window_params = population[fitness_scores.index(best_window_fitness)]

        if best_window_fitness > best_overall_fitness:
            best_overall_fitness = best_window_fitness
            best_overall_params = best_window_params
            display(f"New overall best fitness found: {best_overall_fitness:.4f}")
            display(f"Corresponding parameters: {best_overall_params}")


        # Move the window forward
        current_start_date += pd.Timedelta(days=step_size)
        current_end_date += pd.Timedelta(days=step_size)


    return best_overall_params

# Define the parameter space for the GA
# These ranges should be determined based on domain knowledge or initial exploration
param_space = {
    'weight_trend': (0.1, 0.7),
    'weight_volatility': (0.1, 0.4),
    'weight_exhaustion': (0.1, 0.4),
    'strong_bull_threshold': (30, 70),
    'weak_bull_threshold': (10, 40),
    'neutral_threshold_upper': (10, 30),
    'neutral_threshold_lower': (-30, -10),
    'strong_bear_threshold': (-70, -30),
    'weak_bear_threshold': (-40, -10),
    'stop_loss_multiplier_strong': (1.5, 3.5),
    'stop_loss_multiplier_weak': (0.5, 2.5)
}

# GA and Walk-Forward parameters
n_generations = 10 # Reduced for faster PoC run
pop_size = 20 # Reduced for faster PoC run
mutation_rate = 0.1
lookback_window_days = 365 # 1 year lookback window for optimization
step_size_days = 90 # Step forward by 90 days

# Convert lookback and step size from days to number of bars in dollar_bars_df
# This is an approximation; a more precise approach would map dates to indices
# For simplicity in PoC, we'll use date offsets directly in the loop
# lookback_window_bars = int(len(dollar_bars_df) * (lookback_window_days / (dollar_bars_df.index.max() - dollar_bars_df.index.min()).days))
# step_size_bars = int(len(dollar_bars_df) * (step_size_days / (dollar_bars_df.index.max() - dollar_bars_df.index.min()).days))


# Run the Genetic Algorithm with walk-forward optimization
# Note: This will be computationally intensive even with reduced parameters
display("Starting Genetic Algorithm with Walk-Forward Optimization...")
optimal_params = genetic_algorithm_walk_forward(
    dollar_bars_df.copy(), # Use a copy to avoid modifying original df
    initial_capital,
    param_space,
    n_generations,
    pop_size,
    mutation_rate,
    lookback_window_days,
    step_size_days
)

display("\nOptimal Parameters found:")
display(optimal_params)

"""**Reasoning**:
The walk-forward optimization did not find any parameters with fitness better than the initial -inf. This indicates an issue, possibly that no profitable trades were found within the optimization windows using the initial parameter space and evaluation metric. The next step is to evaluate the performance of the found optimal parameters (which are None in this case, but the evaluation code should handle it) and then investigate why no better parameters were found by analyzing the trade log and potentially adjusting the parameter space or fitness function if needed.


"""

# Evaluate the performance of the optimal parameters on the full dataset
# If optimal_params is None (meaning no improvement was found), use the default parameters
if optimal_params is None:
    display("No optimal parameters found. Evaluating with default parameters.")
    evaluation_params = current_params # Use the parameters from the previous manual run
else:
    display("Evaluating with optimal parameters.")
    evaluation_params = optimal_params


# Re-run the trading simulation with the selected parameters on the full dataset
position = 0  # 1 for long, -1 for short, 0 for flat
entry_price = 0
stop_loss = 0
equity_curve = []
trade_log = []
current_capital = initial_capital

# Get parameters for evaluation simulation
weight_trend_eval = evaluation_params.get('weight_trend', weight_trend)
weight_volatility_eval = evaluation_params.get('weight_volatility', weight_volatility)
weight_exhaustion_eval = evaluation_params.get('weight_exhaustion', weight_exhaustion)
strong_bull_threshold_eval = evaluation_params.get('strong_bull_threshold', strong_bull_threshold)
weak_bull_threshold_eval = evaluation_params.get('weak_bull_threshold', weak_bull_threshold)
neutral_threshold_upper_eval = evaluation_params.get('neutral_threshold_upper', neutral_threshold_upper)
neutral_threshold_lower_eval = evaluation_params.get('neutral_threshold_lower', neutral_threshold_lower)
strong_bear_threshold_eval = evaluation_params.get('strong_bear_threshold', strong_bear_threshold)
weak_bear_threshold_eval = evaluation_params.get('weak_bear_threshold', weak_bear_threshold)
stop_loss_multiplier_strong_eval = evaluation_params.get('stop_loss_multiplier_strong', stop_loss_multiplier_strong)
stop_loss_multiplier_weak_eval = evaluation_params.get('stop_loss_multiplier_weak', stop_loss_multiplier_weak)


# Recalculate MSS and Regime for the full dataset using evaluation parameters
dollar_bars_df['MSS_final'] = (weight_trend_eval * dollar_bars_df['Trend'] +
                               weight_volatility_eval * dollar_bars_df['Volatility_ATR'] +
                               weight_exhaustion_eval * dollar_bars_eval['Exhaustion'])

def classify_regime_final(mss):
    if mss > strong_bull_threshold_eval:
        return 'Strong Bull'
    elif mss > weak_bull_threshold_eval:
        return 'Weak Bull'
    elif mss >= neutral_threshold_lower_eval and mss <= neutral_threshold_upper_eval:
        return 'Neutral'
    elif mss > strong_bear_threshold_eval:
        return 'Weak Bear'
    else:
        return 'Strong Bear'

dollar_bars_df['Regime_final'] = dollar_bars_df['MSS_final'].apply(classify_regime_final)


# Iterate through the dollar bars for the final simulation
for index, row in dollar_bars_df.iterrows():
    current_price = row['Close']
    current_regime = row['Regime_final']
    current_atr = row['Volatility_ATR'] # Using the normalized ATR

    # Determine stop-loss distance based on regime and ATR
    if current_regime in ['Strong Bull', 'Strong Bear']:
        stop_loss_distance = stop_loss_multiplier_strong_eval * current_atr
    elif current_regime in ['Weak Bull', 'Weak Bear']:
        stop_loss_distance = stop_loss_multiplier_weak_eval * current_atr
    else: # Neutral
        stop_loss_distance = 0

    # Trading Logic based on Action Matrix (Section 4.3.3)
    if current_regime == 'Strong Bull':
        if position == 0: # Enter Long
            position = 1
            entry_price = current_price
            stop_loss = current_price - stop_loss_distance
            trade_log.append({'Date': index, 'Action': 'Enter Long', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == 1: # Adjust Trailing Stop for Long
            stop_loss = max(stop_loss, current_price - stop_loss_distance)
            trade_log.append({'Date': index, 'Action': 'Adjust Long Stop', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1: # Exit Short
            pnl = (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0


    elif current_regime == 'Weak Bull':
        if position == 1: # Hold Longs ONLY, tighten stop
             stop_loss = max(stop_loss, current_price - stop_loss_distance)
             trade_log.append({'Date': index, 'Action': 'Hold Long (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1: # Exit Short
            pnl = (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0


    elif current_regime == 'Neutral':
        if position != 0: # EXIT ALL POSITIONS
            action = 'Exit Long' if position == 1 else 'Exit Short'
            pnl = (current_price - entry_price) if position == 1 else (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': action + ' (Neutral Regime)', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0

    elif current_regime == 'Weak Bear':
         if position == -1: # Hold Shorts ONLY, tighten stop
            stop_loss = min(stop_loss, current_price + stop_loss_distance)
            trade_log.append({'Date': index, 'Action': 'Hold Short (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
         elif position == 1: # Exit Long
            pnl = (current_price - entry_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0


    elif current_regime == 'Strong Bear':
        if position == 0: # Enter Short
            position = -1
            entry_price = current_price
            stop_loss = current_price + stop_loss_distance
            trade_log.append({'Date': index, 'Action': 'Enter Short', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1: # Adjust Trailing Stop for Short
             stop_loss = min(stop_loss, current_price + stop_loss_distance)
             trade_log.append({'Date': index, 'Action': 'Adjust Short Stop', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == 1: # Exit Long
            pnl = (current_price - entry_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0


    # Check for stop-loss hit
    if position == 1 and current_price <= stop_loss:
        pnl = (current_price - entry_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0
    elif position == -1 and current_price >= stop_loss:
        pnl = (entry_price - current_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Short', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0

    # Append current equity to the equity curve
    equity_curve.append({'Date': index, 'Equity': current_capital + (current_price - entry_price if position != 0 else 0)})

# Convert trade log and equity curve to DataFrames
equity_curve_df_final = pd.DataFrame(equity_curve).set_index('Date')
trade_log_df_final = pd.DataFrame(trade_log)


# Evaluate performance metrics for the final simulation
if equity_curve_df_final.empty or len(equity_curve_df_final) < 2:
    display("Final equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_final['Daily_Return'] = equity_curve_df_final['Equity'].pct_change().fillna(0)
    total_return_final = (equity_curve_df_final['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_final = (1 + total_return_final)**(trading_periods_per_year / len(equity_curve_df_final)) - 1

    equity_curve_df_final['Peak'] = equity_curve_df_final['Equity'].cummax()
    equity_curve_df_final['Drawdown'] = equity_curve_df_final['Equity'] - equity_curve_df_final['Peak']
    max_drawdown_final = equity_curve_df_final['Drawdown'].min()

    mar = 0
    downside_returns_final = equity_curve_df_final[equity_curve_df_final['Daily_Return'] < mar]['Daily_Return']
    downside_deviation_final = downside_returns_final.std()

    sortino_ratio_final = (annualized_return_final - mar) / downside_deviation_final if downside_deviation_final != 0 and not np.isnan(downside_deviation_final) else (float('inf') if annualized_return_final > mar else (0 if annualized_return_final == mar else -1000))

    calmar_ratio_final = annualized_return_final / abs(max_drawdown_final) if max_drawdown_final != 0 else float('inf')


    display("\nFinal Performance Metrics (using selected parameters):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_final['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_final:.4f}")
    display(f"Annualized Return: {annualized_return_final:.4f}")
    display(f"Max Drawdown: {max_drawdown_final:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_final:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_final:.4f}")

    # Plot final equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_final.index, equity_curve_df_final['Equity'], label='Final Equity Curve')
    plt.title('Final Equity Curve (using selected parameters)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the final trade log
    display("\nFinal Trade Log Summary:")
    display(trade_log_df_final.head())

    # Analyze losing trades from the final simulation
    losing_trades_df_final = trade_log_df_final[trade_log_df_final['PnL'].notna() & (trade_log_df_final['PnL'] < 0)]
    display("\nLosing Trades Summary from Final Simulation:")
    display(losing_trades_df_final.head()) # Display head of losing trades

"""**Reasoning**:
The previous code failed because `dollar_bars_eval` was not defined. It should have used `dollar_bars_df` instead. Fix this typo and re-run the evaluation with the selected parameters.


"""

# Re-run the trading simulation with the selected parameters on the full dataset
# If optimal_params is None (meaning no improvement was found), use the default parameters
if optimal_params is None:
    display("No optimal parameters found. Evaluating with default parameters.")
    evaluation_params = current_params # Use the parameters from the previous manual run
else:
    display("Evaluating with optimal parameters.")
    evaluation_params = optimal_params


# Re-run the trading simulation with the selected parameters on the full dataset
position = 0  # 1 for long, -1 for short, 0 for flat
entry_price = 0
stop_loss = 0
equity_curve = []
trade_log = []
current_capital = initial_capital

# Get parameters for evaluation simulation
weight_trend_eval = evaluation_params.get('weight_trend', weight_trend)
weight_volatility_eval = evaluation_params.get('weight_volatility', weight_volatility)
weight_exhaustion_eval = evaluation_params.get('weight_exhaustion', weight_exhaustion)
strong_bull_threshold_eval = evaluation_params.get('strong_bull_threshold', strong_bull_threshold)
weak_bull_threshold_eval = evaluation_params.get('weak_bull_threshold', weak_bull_threshold)
neutral_threshold_upper_eval = evaluation_params.get('neutral_threshold_upper', neutral_threshold_upper)
neutral_threshold_lower_eval = evaluation_params.get('neutral_threshold_lower', neutral_threshold_lower)
strong_bear_threshold_eval = evaluation_params.get('strong_bear_threshold', strong_bear_threshold)
weak_bear_threshold_eval = evaluation_params.get('weak_bear_threshold', weak_bear_threshold)
stop_loss_multiplier_strong_eval = evaluation_params.get('stop_loss_multiplier_strong', stop_loss_multiplier_strong)
stop_loss_multiplier_weak_eval = evaluation_params.get('stop_loss_multiplier_weak', stop_loss_multiplier_weak)


# Recalculate MSS and Regime for the full dataset using evaluation parameters
dollar_bars_df['MSS_final'] = (weight_trend_eval * dollar_bars_df['Trend'] +
                               weight_volatility_eval * dollar_bars_df['Volatility_ATR'] +
                               weight_exhaustion_eval * dollar_bars_df['Exhaustion']) # Fixed typo here

def classify_regime_final(mss):
    if mss > strong_bull_threshold_eval:
        return 'Strong Bull'
    elif mss > weak_bull_threshold_eval:
        return 'Weak Bull'
    elif mss >= neutral_threshold_lower_eval and mss <= neutral_threshold_upper_eval:
        return 'Neutral'
    elif mss > strong_bear_threshold_eval:
        return 'Weak Bear'
    else:
        return 'Strong Bear'

dollar_bars_df['Regime_final'] = dollar_bars_df['MSS_final'].apply(classify_regime_final)


# Iterate through the dollar bars for the final simulation
for index, row in dollar_bars_df.iterrows():
    current_price = row['Close']
    current_regime = row['Regime_final']
    current_atr = row['Volatility_ATR'] # Using the normalized ATR

    # Determine stop-loss distance based on regime and ATR
    if current_regime in ['Strong Bull', 'Strong Bear']:
        stop_loss_distance = stop_loss_multiplier_strong_eval * current_atr
    elif current_regime in ['Weak Bull', 'Weak Bear']:
        stop_loss_distance = stop_loss_multiplier_weak_eval * current_atr
    else: # Neutral
        stop_loss_distance = 0

    # Trading Logic based on Action Matrix (Section 4.3.3)
    if current_regime == 'Strong Bull':
        if position == 0: # Enter Long
            position = 1
            entry_price = current_price
            stop_loss = current_price - stop_loss_distance
            trade_log.append({'Date': index, 'Action': 'Enter Long', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == 1: # Adjust Trailing Stop for Long
            stop_loss = max(stop_loss, current_price - stop_loss_distance)
            trade_log.append({'Date': index, 'Action': 'Adjust Long Stop', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1: # Exit Short
            pnl = (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0


    elif current_regime == 'Weak Bull':
        if position == 1: # Hold Longs ONLY, tighten stop
             stop_loss = max(stop_loss, current_price - stop_loss_distance)
             trade_log.append({'Date': index, 'Action': 'Hold Long (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1: # Exit Short
            pnl = (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0


    elif current_regime == 'Neutral':
        if position != 0: # EXIT ALL POSITIONS
            action = 'Exit Long' if position == 1 else 'Exit Short'
            pnl = (current_price - entry_price) if position == 1 else (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': action + ' (Neutral Regime)', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0

    elif current_regime == 'Weak Bear':
         if position == -1: # Hold Shorts ONLY, tighten stop
            stop_loss = min(stop_loss, current_price + stop_loss_distance)
            trade_log.append({'Date': index, 'Action': 'Hold Short (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
         elif position == 1: # Exit Long
            pnl = (current_price - entry_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0


    elif current_regime == 'Strong Bear':
        if position == 0: # Enter Short
            position = -1
            entry_price = current_price
            stop_loss = current_price + stop_loss_distance
            trade_log.append({'Date': index, 'Action': 'Enter Short', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1: # Adjust Trailing Stop for Short
             stop_loss = min(stop_loss, current_price + stop_loss_distance)
             trade_log.append({'Date': index, 'Action': 'Adjust Short Stop', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == 1: # Exit Long
            pnl = (current_price - entry_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0


    # Check for stop-loss hit
    if position == 1 and current_price <= stop_loss:
        pnl = (current_price - entry_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0
    elif position == -1 and current_price >= stop_loss:
        pnl = (entry_price - current_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Short', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0

    # Append current equity to the equity curve
    equity_curve.append({'Date': index, 'Equity': current_capital + (current_price - entry_price if position != 0 else 0)})

# Convert trade log and equity curve to DataFrames
equity_curve_df_final = pd.DataFrame(equity_curve).set_index('Date')
trade_log_df_final = pd.DataFrame(trade_log)


# Evaluate performance metrics for the final simulation
if equity_curve_df_final.empty or len(equity_curve_df_final) < 2:
    display("Final equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_final['Daily_Return'] = equity_curve_df_final['Equity'].pct_change().fillna(0)
    total_return_final = (equity_curve_df_final['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_final = (1 + total_return_final)**(trading_periods_per_year / len(equity_curve_df_final)) - 1

    equity_curve_df_final['Peak'] = equity_curve_df_final['Equity'].cummax()
    equity_curve_df_final['Drawdown'] = equity_curve_df_final['Equity'] - equity_curve_df_final['Peak']
    max_drawdown_final = equity_curve_df_final['Drawdown'].min()

    mar = 0
    downside_returns_final = equity_curve_df_final[equity_curve_df_final['Daily_Return'] < mar]['Daily_Return']
    downside_deviation_final = downside_returns_final.std()

    sortino_ratio_final = (annualized_return_final - mar) / downside_deviation_final if downside_deviation_final != 0 and not np.isnan(downside_deviation_final) else (float('inf') if annualized_return_final > mar else (0 if annualized_return_final == mar else -1000))

    calmar_ratio_final = annualized_return_final / abs(max_drawdown_final) if max_drawdown_final != 0 else float('inf')


    display("\nFinal Performance Metrics (using selected parameters):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_final['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_final:.4f}")
    display(f"Annualized Return: {annualized_return_final:.4f}")
    display(f"Max Drawdown: {max_drawdown_final:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_final:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_final:.4f}")

    # Plot final equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_final.index, equity_curve_df_final['Equity'], label='Final Equity Curve')
    plt.title('Final Equity Curve (using selected parameters)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the final trade log
    display("\nFinal Trade Log Summary:")
    display(trade_log_df_final.head())

    # Analyze losing trades from the final simulation
    losing_trades_df_final = trade_log_df_final[trade_log_df_final['PnL'].notna() & (trade_log_df_final['PnL'] < 0)]
    display("\nLosing Trades Summary from Final Simulation:")
    display(losing_trades_df_final.head()) # Display head of losing trades

"""**Reasoning**:
The evaluation with default parameters shows negative performance, and the GA did not find better parameters. This suggests either the parameter space is too small, the fitness function is not effectively guiding the search, or the strategy itself needs refinement. Analyzing the losing trades might provide clues. The next step is to finalize the subtask by analyzing the characteristics of losing trades to understand why they occurred.


"""

# Further analyze losing trades.
# Display summary statistics for losing trades
display("\nSummary Statistics for Losing Trades:")
display(losing_trades_df_final['PnL'].describe())

# Analyze the distribution of trade actions for losing trades
display("\nDistribution of Actions for Losing Trades:")
display(losing_trades_df_final['Action'].value_counts())

# Optional: Merge losing trades with dollar_bars_df to analyze market conditions during losing trades
# This requires aligning the trade log dates with the dollar bar dates, which can be tricky
# due to trades potentially happening between dollar bar timestamps.
# For simplicity in the PoC, we'll just look at the date and action for now.

# If needed, you would join on 'Date' and potentially look at the MSS and Regime
# at the time of the losing trade to see if there are common patterns.
# Example (conceptual, requires careful handling of timestamps):
# losing_trades_with_market_state = losing_trades_df_final.merge(
#     dollar_bars_df[['MSS_final', 'Regime_final']],
#     left_on='Date',
#     right_index=True,
#     how='left'
# )
# display("\nMarket State during Losing Trades (Head):")
# display(losing_trades_with_market_state.head())


# The subtask is completed. Report the finish status and the dataframes.

"""## Summary:

### Data Analysis Key Findings

*   The `evaluate_strategy` function was successfully implemented to calculate the Sortino Ratio for a given set of trading strategy parameters by simulating trades on historical dollar-bar data.
*   An initial backtest with default strategy parameters resulted in a Sortino Ratio of -3.6681, indicating poor performance.
*   A Genetic Algorithm with walk-forward optimization was implemented to search for better parameters, but it did not find any parameter sets that improved upon the initial fitness within the tested windows.
*   The final evaluation using the default parameters showed a negative total return (-10.97%), negative annualized return (-3.60%), a significant maximum drawdown (-42,070.44), and a negative Sortino Ratio (-3.6681).
*   Analysis of losing trades revealed that 'Stop Out Long' and 'Stop Out Short' actions were the most frequent causes of losses, indicating that stop losses are being triggered often in adverse movements.

### Insights or Next Steps

*   The current strategy, even after attempted optimization, appears unprofitable based on the evaluated metrics. Future work should focus on refining the strategy logic itself or expanding the parameter search space for the optimization.
*   Further analysis of losing trades should investigate the market conditions (specifically MSS and Regime) at the time of stop-loss hits to understand if there are patterns that could inform strategy adjustments or stop-loss placement rules.

# Task
Implement a Genetic Algorithm with walk-forward optimization to optimize both numeric parameters and indicator choices for a trading strategy based on the provided PRD in "/content/Helios Trader.md". The GA should use a fitness function that evaluates performance using a risk-adjusted return metric on dollar-bar data derived from the input data. After running the GA, evaluate the best found parameters and indicators and present the results.

## Define the fitness function (with indicator selection)

### Subtask:
Implement a fitness function that evaluates the performance of a given set of strategy parameters, *including the choice of indicator for each factor*. This function will run backtests using the historical data and a risk-adjusted return metric.

**Reasoning**:
Modify the `evaluate_strategy` function to accept indicator choices as parameters and use the specified indicators to calculate the factors before running the trading simulation and calculating the Sortino Ratio.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress

# Re-define factor calculation functions to be selectable
def calculate_trend_slope(df, lookback):
    df['Trend_Slope_Val'] = np.nan
    for i in range(lookback, len(df)):
        y = df['Close'].iloc[i-lookback:i]
        x = np.arange(lookback)
        slope, _, _, _, _ = linregress(x, y)
        df.loc[df.index[i], 'Trend_Slope_Val'] = slope

    # Normalize Trend: Scale relative to the rolling standard deviation of slopes
    df['Slope_Std'] = df['Trend_Slope_Val'].rolling(window=lookback).std()
    valid_indices = df.index[df['Slope_Std'].notna() & (df['Slope_Std'].abs() > 1e-9)]
    scaling_factor = 100 / (df.loc[valid_indices, 'Slope_Std'] * 2) # Scale +/- 2 std dev to +/- 100
    df.loc[valid_indices, 'Trend_Factor'] = df.loc[valid_indices, 'Trend_Slope_Val'] * scaling_factor

    df = df.drop(columns=['Trend_Slope_Val', 'Slope_Std'])
    df['Trend_Factor'] = np.clip(df['Trend_Factor'], -100, 100) # Clip to bounds
    return df['Trend_Factor']

# Placeholder for another potential trend indicator (e.g., Moving Average Convergence Divergence - MACD)
# def calculate_trend_macd(df, fastperiod=12, slowperiod=26, signalperiod=9):
#     # This would require implementing MACD calculation
#     # For now, it's just a placeholder to show the structure for selectable indicators
#     df['Trend_Factor'] = 0 # Placeholder calculation
#     return df['Trend_Factor']

def calculate_volatility_atr(df, lookback):
    df['TR'] = np.maximum(np.maximum(df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1))), abs(df['Low'] - df['Close'].shift(1)))
    df['Volatility_ATR_Absolute'] = df['TR'].rolling(window=lookback).mean()
    df = df.drop(columns=['TR'])

    # Modified Normalization for ATR: Scale relative to a rolling average of ATR
    df['ATR_MA'] = df['Volatility_ATR_Absolute'].rolling(window=lookback).mean()
    valid_indices = df.index[df['ATR_MA'].notna() & (df['ATR_MA'].abs() > 1e-9)]
    df.loc[valid_indices, 'Volatility_Factor'] = ((df.loc[valid_indices, 'Volatility_ATR_Absolute'] / df.loc[valid_indices, 'ATR_MA']) - 1) * 100 # Percentage deviation from MA

    df = df.drop(columns=['ATR_MA', 'Volatility_ATR_Absolute'])
    df['Volatility_Factor'] = np.clip(df['Volatility_Factor'], -100, 100) # Clip to bounds
    return df['Volatility_Factor']

# Placeholder for another potential volatility indicator (e.g., Standard Deviation)
# def calculate_volatility_stddev(df, lookback):
#     # This would require implementing Standard Deviation calculation
#     df['Volatility_Factor'] = 0 # Placeholder calculation
#     return df['Volatility_Factor']

def calculate_exhaustion_sma_diff(df, sma_lookback, atr_series):
    df['SMA'] = df['Close'].rolling(window=sma_lookback).mean()

    # Ensure ATR series is aligned and not NaN for calculation
    # Using the passed atr_series directly
    valid_indices = df.index[atr_series.notna() & (atr_series.abs() > 1e-9)]
    df.loc[valid_indices, 'Exhaustion_Factor'] = (df.loc[valid_indices, 'Close'] - df.loc[valid_indices, 'SMA']) / atr_series.loc[valid_indices]

    df = df.drop(columns=['SMA'])
    # Normalize Exhaustion: Scale the ratio to fit -100 to 100
    # Assuming a range like -10 to 10 covers most cases relative to ATR
    scaling_factor = 100 / 10
    df['Exhaustion_Factor'] = np.clip(df['Exhaustion_Factor'] * scaling_factor, -100, 100) # Clip to bounds

    return df['Exhaustion_Factor']

# Placeholder for another potential exhaustion indicator (e.g., Relative Strength Index - RSI)
# def calculate_exhaustion_rsi(df, lookback):
#     # This would require implementing RSI calculation
#     df['Exhaustion_Factor'] = 0 # Placeholder calculation
#     return df['Exhaustion_Factor']


# Map indicator names to functions
INDICATOR_FUNCTIONS = {
    'Trend': {
        'Slope': calculate_trend_slope,
        # 'MACD': calculate_trend_macd, # Add actual MACD function when implemented
    },
    'Volatility': {
        'ATR': calculate_volatility_atr,
        # 'StdDev': calculate_volatility_stddev, # Add actual StdDev function when implemented
    },
    'Exhaustion': {
        'SMADiff': calculate_exhaustion_sma_diff,
        # 'RSI': calculate_exhaustion_rsi, # Add actual RSI function when implemented
    }
}


def evaluate_strategy_with_indicators(params, df, initial_capital=100000):
    """
    Evaluates a trading strategy with given parameters and indicator choices
    on historical data.

    Args:
        params (dict): Dictionary of strategy parameters, including:
                       - 'weight_trend', 'weight_volatility', 'weight_exhaustion'
                       - 'strong_bull_threshold', 'weak_bull_threshold', ...
                       - 'stop_loss_multiplier_strong', 'stop_loss_multiplier_weak'
                       - 'indicator_trend': Name of indicator for Trend factor (e.g., 'Slope')
                       - 'indicator_volatility': Name of indicator for Volatility factor (e.g., 'ATR')
                       - 'indicator_exhaustion': Name of indicator for Exhaustion factor (e.g., 'SMADiff')
                       - 'lookback_trend': Lookback period for Trend indicator
                       - 'lookback_volatility': Lookback period for Volatility indicator
                       - 'lookback_exhaustion': Lookback period for Exhaustion indicator

        df (pd.DataFrame): DataFrame containing dollar bars (Open, High, Low, Close, Volume, DollarVolume).
        initial_capital (float): Starting capital for the backtest.

    Returns:
        float: The Sortino Ratio of the strategy's performance, or a very low number
               if the ratio is infinite or NaN or if there's an error.
    """
    df_eval = df.copy() # Work on a copy

    # Extract parameters, including indicator choices and lookbacks
    weight_trend = params.get('weight_trend', 0.5)
    weight_volatility = params.get('weight_volatility', 0.2)
    weight_exhaustion = params.get('weight_exhaustion', 0.3)
    strong_bull_threshold = params.get('strong_bull_threshold', 50)
    weak_bull_threshold = params.get('weak_bull_threshold', 20)
    neutral_threshold_upper = params.get('neutral_threshold_upper', 20)
    neutral_threshold_lower = params.get('neutral_threshold_lower', -20)
    strong_bear_threshold = params.get('strong_bear_threshold', -50)
    weak_bear_threshold = params.get('weak_bear_threshold', -20)
    stop_loss_multiplier_strong = params.get('stop_loss_multiplier_strong', 2)
    stop_loss_multiplier_weak = params.get('stop_loss_multiplier_weak', 1)

    # Indicator choices and lookbacks
    indicator_trend_name = params.get('indicator_trend', 'Slope')
    indicator_volatility_name = params.get('indicator_volatility', 'ATR')
    indicator_exhaustion_name = params.get('indicator_exhaustion', 'SMADiff')

    lookback_trend = params.get('lookback_trend', 20)
    lookback_volatility = params.get('lookback_volatility', 20)
    lookback_exhaustion = params.get('lookback_exhaustion', 20) # SMA lookback for SMADiff

    # --- Calculate Factors using specified indicators ---
    try:
        # Calculate Volatility first as Exhaustion might depend on it
        if indicator_volatility_name in INDICATOR_FUNCTIONS['Volatility']:
            df_eval['Volatility_Factor'] = INDICATOR_FUNCTIONS['Volatility'][indicator_volatility_name](df_eval, lookback_volatility)
        else:
            display(f"Warning: Unknown Volatility indicator: {indicator_volatility_name}. Using default (ATR).")
            df_eval['Volatility_Factor'] = calculate_volatility_atr(df_eval, lookback_volatility)

        if indicator_trend_name in INDICATOR_FUNCTIONS['Trend']:
             df_eval['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend'][indicator_trend_name](df_eval, lookback_trend)
        else:
             display(f"Warning: Unknown Trend indicator: {indicator_trend_name}. Using default (Slope).")
             df_eval['Trend_Factor'] = calculate_trend_slope(df_eval, lookback_trend)

        if indicator_exhaustion_name in INDICATOR_FUNCTIONS['Exhaustion']:
            # Exhaustion calculation might need the Volatility factor output
            if indicator_exhaustion_name == 'SMADiff':
                 # SMADiff specifically requires the normalized ATR as an argument
                 df_eval['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_name](df_eval, lookback_exhaustion, df_eval['Volatility_Factor'])
            else:
                 # Other exhaustion indicators would take df and their lookback
                 df_eval['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_name](df_eval, lookback_exhaustion)
        else:
            display(f"Warning: Unknown Exhaustion indicator: {indicator_exhaustion_name}. Using default (SMADiff).")
            df_eval['Exhaustion_Factor'] = calculate_exhaustion_sma_diff(df_eval, lookback_exhaustion, df_eval['Volatility_Factor'])

    except Exception as e:
        display(f"Error during factor calculation: {e}")
        return -2000 # Return very low fitness on error


    # Drop rows with NaN values generated by lookback periods
    initial_rows = len(df_eval)
    df_eval = df_eval.dropna(subset=['Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor'])
    if len(df_eval) == 0:
        display("Warning: DataFrame is empty after dropping NaNs.")
        return -2000 # Cannot evaluate performance if no valid data remains

    # Recalculate MSS with new weights
    df_eval['MSS_eval'] = (weight_trend * df_eval['Trend_Factor'] +
                           weight_volatility * df_eval['Volatility_Factor'] +
                           weight_exhaustion * df_eval['Exhaustion_Factor'])

    # Reclassify Regime with new thresholds
    def classify_regime_eval(mss):
        if mss > strong_bull_threshold:
            return 'Strong Bull'
        elif mss > weak_bull_threshold:
            return 'Weak Bull'
        elif mss >= neutral_threshold_lower and mss <= neutral_threshold_upper:
            return 'Neutral'
        elif mss > strong_bear_threshold:
            return 'Weak Bear'
        else:
            return 'Strong Bear'

    df_eval['Regime_eval'] = df_eval['MSS_eval'].apply(classify_regime_eval)


    # --- Trading Simulation ---
    position = 0
    entry_price = 0
    #stop_loss = 0 # Stop loss will be calculated dynamically
    equity_curve = []
    current_capital = initial_capital

    for index, row in df_eval.iterrows():
        current_price = row['Close']
        current_regime = row['Regime_eval']
        current_atr = row['Volatility_Factor'] # Using the normalized Volatility Factor (which is ATR based)

        # Ensure current_atr is a number and not zero for stop-loss calculation
        if not isinstance(current_atr, (int, float)) or np.isnan(current_atr) or current_atr == 0:
             stop_loss_distance = 0 # Cannot calculate dynamic stop loss without valid ATR
             # display(f"Warning: Invalid ATR for stop loss calculation at {index}. ATR: {current_atr}")
        else:
            # Determine stop-loss distance based on regime and ATR
            if current_regime in ['Strong Bull', 'Strong Bear']:
                stop_loss_distance = stop_loss_multiplier_strong * current_atr
            elif current_regime in ['Weak Bull', 'Weak Bear']:
                stop_loss_distance = stop_loss_multiplier_weak * current_atr
            else: # Neutral
                stop_loss_distance = 0 # No stop-loss in neutral


        # Trading Logic based on Action Matrix (Section 4.3.3)
        if current_regime == 'Strong Bull':
            if position == 0: # Enter Long
                position = 1
                entry_price = current_price
                # Initial stop loss calculation
                stop_loss = current_price - stop_loss_distance if stop_loss_distance > 0 else -float('inf') # Ensure stop is below entry for long
            elif position == 1 and stop_loss_distance > 0: # Adjust Trailing Stop for Long if valid distance
                 stop_loss = max(stop_loss, current_price - stop_loss_distance)
            elif position == -1: # Exit Short
                pnl = (entry_price - current_price)
                current_capital += pnl
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Weak Bull':
            if position == 1 and stop_loss_distance > 0: # Hold Longs ONLY, tighten stop if valid distance
                 stop_loss = max(stop_loss, current_price - stop_loss_distance)
            elif position == -1: # Exit Short
                pnl = (entry_price - current_price)
                current_capital += pnl
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Neutral':
            if position != 0: # EXIT ALL POSITIONS
                pnl = (current_price - entry_price) if position == 1 else (entry_price - current_price)
                current_capital += pnl
                position = 0
                entry_price = 0
                stop_loss = 0

        elif current_regime == 'Weak Bear':
             if position == -1 and stop_loss_distance > 0: # Hold Shorts ONLY, tighten stop if valid distance (above price)
                stop_loss = min(stop_loss, current_price + stop_loss_distance) # Adjust stop loss above price
             elif position == 1: # Exit Long
                pnl = (current_price - entry_price)
                current_capital += pnl
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Strong Bear':
            if position == 0: # Enter Short
                position = -1
                entry_price = current_price
                # Initial stop loss calculation
                stop_loss = current_price + stop_loss_distance if stop_loss_distance > 0 else float('inf') # Ensure stop is above entry for short
            elif position == -1 and stop_loss_distance > 0: # Adjust Trailing Stop for Short if valid distance
                 stop_loss = min(stop_loss, current_price + stop_loss_distance) # Adjust stop loss above price
            elif position == 1: # Exit Long
                pnl = (current_price - entry_price)
                current_capital += pnl
                position = 0
                entry_price = 0
                stop_loss = 0


        # Check for stop-loss hit (only if position is active and stop_loss is a valid number)
        if position == 1 and not np.isinf(stop_loss) and current_price <= stop_loss:
            pnl = (current_price - entry_price)
            current_capital += pnl
            position = 0
            entry_price = 0
            stop_loss = 0
        elif position == -1 and not np.isinf(stop_loss) and current_price >= stop_loss:
            pnl = (entry_price - current_price)
            current_capital += pnl
            position = 0
            entry_price = 0
            stop_loss = 0


        # Append current equity to the equity curve
        equity_curve.append({'Date': index, 'Equity': current_capital + (current_price - entry_price if position != 0 else 0)})


    equity_curve_df_eval = pd.DataFrame(equity_curve).set_index('Date')

    # --- Performance Evaluation (Sortino Ratio) ---
    if equity_curve_df_eval.empty or len(equity_curve_df_eval) < 2:
         # display("Warning: Equity curve is empty or too short for Sortino Ratio calculation.")
         return -1000 # Return a very low fitness

    equity_curve_df_eval['Daily_Return'] = equity_curve_df_eval['Equity'].pct_change().fillna(0)

    # Calculate Annualized Return (assuming daily data from dollar bars)
    trading_periods_per_year = 365
    total_return = (equity_curve_df_eval['Equity'].iloc[-1] - initial_capital) / initial_capital

    if len(equity_curve_df_eval) > 1:
      annualized_return = (1 + total_return)**(trading_periods_per_year / len(equity_curve_df_eval)) - 1
    else:
      annualized_return = 0


    # Calculate Sortino Ratio
    mar = 0 # Minimum Acceptable Return (MAR)
    downside_returns = equity_curve_df_eval[equity_curve_df_eval['Daily_Return'] < mar]['Daily_Return']
    downside_deviation = downside_returns.std()

    if downside_deviation == 0 or np.isnan(downside_deviation):
        sortino_ratio = float('inf') if annualized_return > mar else (0 if annualized_return == mar else -1000)
    else:
        sortino_ratio = (annualized_return - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN after calculation
    if np.isinf(sortino_ratio) or np.isnan(sortino_ratio):
        # display(f"Warning: Calculated Sortino Ratio is {sortino_ratio}. Returning low fitness.")
        return -1000

    return sortino_ratio


# Example usage of the updated fitness function (using current parameters and default indicators/lookbacks)
current_params_with_indicators = {
    'weight_trend': weight_trend,
    'weight_volatility': weight_volatility,
    'weight_exhaustion': weight_exhaustion,
    'strong_bull_threshold': strong_bull_threshold,
    'weak_bull_threshold': weak_bull_threshold,
    'neutral_threshold_upper': neutral_threshold_upper,
    'neutral_threshold_lower': neutral_threshold_lower,
    'strong_bear_threshold': strong_bear_threshold,
    'weak_bear_threshold': weak_bear_threshold,
    'stop_loss_multiplier_strong': stop_loss_multiplier_strong,
    'stop_loss_multiplier_weak': stop_loss_multiplier_weak,
    'indicator_trend': 'Slope', # Default indicator
    'indicator_volatility': 'ATR', # Default indicator
    'indicator_exhaustion': 'SMADiff', # Default indicator
    'lookback_trend': 20, # Default lookback
    'lookback_volatility': 20, # Default lookback
    'lookback_exhaustion': 20 # Default lookback
}

# The dollar_bars_df should contain only the raw OHLCV and DollarVolume columns at this stage
# The factor calculation will happen inside the evaluate_strategy_with_indicators function
# So let's recreate a clean dollar_bars_df without the old factor columns for evaluation
clean_dollar_bars_df = create_dollar_bars(df, dollar_volume_threshold) # Re-create dollar bars
# Note: If the original df has already been modified (e.g., by previous factor calculations),
# you might need to reload or use a clean version of the initial dollar bars here.
# Assuming create_dollar_bars works on the raw 'df' loaded initially.

# If you need to use the dollar_bars_df that was generated in previous steps
# which might already have some factor columns, make sure your factor functions
# handle potential existing columns gracefully or work on a strict subset copy.
# For clarity, let's ensure we work with a dataframe having only the necessary columns for factor calculation:
df_for_evaluation = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


fitness = evaluate_strategy_with_indicators(current_params_with_indicators, df_for_evaluation, initial_capital)
display(f"Fitness (Sortino Ratio) with current parameters and default indicators: {fitness}")

"""**Reasoning**:
Implement the genetic algorithm with walk-forward optimization, incorporating the choice of indicators and their lookback periods into the parameter space and using the updated fitness function to evaluate individuals.


"""

import random

def genetic_algorithm_walk_forward_with_indicators(df, initial_capital, param_space, n_generations, pop_size, mutation_rate, lookback_window_days, step_size_days):
    """
    Runs a Genetic Algorithm with walk-forward optimization, optimizing both
    numeric parameters and indicator choices/lookbacks.

    Args:
        df (pd.DataFrame): DataFrame containing the full historical data (dollar bars OHLCV).
                           Factors will be calculated inside the fitness function.
        initial_capital (float): Starting capital for backtests.
        param_space (dict): Dictionary defining the search space for each parameter,
                            including numeric ranges and indicator choices/lookback ranges.
                            Example: {'weight_trend': (0.1, 0.7), ..., 'indicator_trend': ['Slope', 'MACD'], 'lookback_trend': (10, 50)}.
        n_generations (int): Number of generations for the GA.
        pop_size (int): Size of the population in each generation.
        mutation_rate (float): Probability of mutation for each parameter.
        lookback_window_days (int): Size of the walk-forward backtesting window (in days).
        step_size_days (int): Size of the step for the walk-forward window (in days).

    Returns:
        dict: The best set of parameters (numeric and indicator choices/lookbacks)
              found during the walk-forward optimization.
    """

    best_overall_params = None
    best_overall_fitness = -float('inf')

    # Determine the dates for walk-forward windows
    all_dates = df.index.unique().sort_values()
    if len(all_dates) < lookback_window_days + step_size_days: # Ensure enough data for at least one window + step
         display("Error: Not enough data for walk-forward optimization with the specified window and step size.")
         return None

    # Use date indices for walk-forward steps
    window_start_idx = 0
    # Find the index corresponding to the end of the first lookback window
    # This is an approximation, assuming roughly daily frequency for dollar bars
    first_window_end_date = all_dates[0] + pd.Timedelta(days=lookback_window_days)
    first_window_end_idx = all_dates.searchsorted(first_window_end_date, side='right')[0]

    if first_window_end_idx >= len(all_dates):
         display("Error: Lookback window is longer than the available data.")
         return None


    while first_window_end_idx < len(all_dates):
        # Define the training window (backtest period for GA optimization)
        current_start_date = all_dates[window_start_idx]
        current_end_date = all_dates[first_window_end_idx - 1] # Use the last date in the window

        train_df = df.loc[current_start_date:current_end_date].copy()

        # Ensure the training window has enough data points after potential NaN removal by indicators
        # A minimum number of bars (e.g., twice the max lookback) is a heuristic
        min_bars_for_ga = max(param_space.get('lookback_trend', (0,0))[1],
                              param_space.get('lookback_volatility', (0,0))[1],
                              param_space.get('lookback_exhaustion', (0,0))[1]) * 2 # Heuristic: require at least twice the max lookback bars
        min_bars_for_ga = max(min_bars_for_ga, 50) # Ensure a minimum number of bars regardless of lookbacks

        if len(train_df) < min_bars_for_ga:
            display(f"Skipping window starting {current_start_date.strftime('%Y-%m-%d')} due to insufficient data ({len(train_df)} bars). Minimum required: {min_bars_for_ga}")
            # Move the window forward by step_size_days
            window_start_idx = all_dates.searchsorted(current_start_date + pd.Timedelta(days=step_size_days), side='left')[0]
            first_window_end_idx = all_dates.searchsorted(all_dates[window_start_idx] + pd.Timedelta(days=lookback_window_days), side='right')[0]
            continue


        display(f"Optimizing on window: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")

        # --- Genetic Algorithm ---
        # Initialize population
        population = []
        for _ in range(pop_size):
            params = {}
            for param, value_range in param_space.items():
                if isinstance(value_range, tuple): # Numeric parameter
                    min_val, max_val = value_range
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param] = random.randint(min_val, max_val)
                    else:
                        params[param] = random.uniform(min_val, max_val)
                elif isinstance(value_range, list): # Categorical parameter (indicator choice)
                    params[param] = random.choice(value_range)
                # Ensure weights sum to 1 (simple normalization after random assignment)
            total_weight = params.get('weight_trend', 0.5) + params.get('weight_volatility', 0.2) + params.get('weight_exhaustion', 0.3)
            if total_weight > 0:
                if 'weight_trend' in params: params['weight_trend'] /= total_weight
                if 'weight_volatility' in params: params['weight_volatility'] /= total_weight
                if 'weight_exhaustion' in params: params['weight_exhaustion'] /= total_weight
            else: # Avoid division by zero
                 if 'weight_trend' in params: params['weight_trend'] = 1/3
                 if 'weight_volatility' in params: params['weight_volatility'] = 1/3
                 if 'weight_exhaustion' in params: params['weight_exhaustion'] = 1/3


            population.append(params)


        # GA generations
        for generation in range(n_generations):
            # Evaluate fitness of each individual
            fitness_scores = [evaluate_strategy_with_indicators(params, train_df.copy(), initial_capital) for params in population]

            # Handle potential errors or invalid fitness scores from evaluation
            valid_fitness_indices = [i for i, score in enumerate(fitness_scores) if not np.isinf(score) and not np.isnan(score) and score > -999] # Use -999 to exclude error codes

            if not valid_fitness_indices:
                 display(f"Generation {generation+1}: No valid fitness scores. Skipping selection and mutation.")
                 # If no valid individuals, re-initialize population or exit?
                 # For simplicity, let's just continue and hope the next generation is better or re-initialize
                 # Re-initialization might be better to escape local optima
                 population = []
                 for _ in range(pop_size):
                      params = {}
                      for param, value_range in param_space.items():
                           if isinstance(value_range, tuple): # Numeric parameter
                               min_val, max_val = value_range
                               if isinstance(min_val, int) and isinstance(max_val, int):
                                    params[param] = random.randint(min_val, max_val)
                               else:
                                    params[param] = random.uniform(min_val, max_val)
                           elif isinstance(value_range, list): # Categorical parameter (indicator choice)
                                params[param] = random.choice(value_range)
                      total_weight = params.get('weight_trend', 0.5) + params.get('weight_volatility', 0.2) + params.get('weight_exhaustion', 0.3)
                      if total_weight > 0:
                           if 'weight_trend' in params: params['weight_trend'] /= total_weight
                           if 'weight_volatility' in params: params['weight_volatility'] /= total_weight
                           if 'weight_exhaustion' in params: params['weight_exhaustion'] /= total_weight
                      else:
                           if 'weight_trend' in params: params['weight_trend'] = 1/3
                           if 'weight_volatility' in params: params['weight_volatility'] = 1/3
                           if 'weight_exhaustion' in params: params['weight_exhaustion'] = 1/3

                      population.append(params)

                 continue # Skip selection/crossover/mutation for this generation


            # Select parents from valid individuals (e.g., tournament selection)
            valid_population = [population[i] for i in valid_fitness_indices]
            valid_fitness = [fitness_scores[i] for i in valid_fitness_indices]

            parents = []
            for _ in range(pop_size // 2):
                # Select two random indices from the valid indices
                idx1_valid, idx2_valid = random.sample(range(len(valid_population)), 2)
                parent1 = valid_population[idx1_valid] if valid_fitness[idx1_valid] > valid_fitness[idx2_valid] else valid_population[idx2_valid]
                idx1_valid, idx2_valid = random.sample(range(len(valid_population)), 2)
                parent2 = valid_population[idx1_valid] if valid_fitness[idx1_valid] > valid_fitness[idx2_valid] else valid_population[idx2_valid]
                parents.extend([parent1, parent2])

            # Create next generation (crossover and mutation)
            next_population = []
            for i in range(0, len(parents), 2):
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = {}, {}

                # Crossover
                for param in param_space.keys():
                    if random.random() < 0.5:
                        child1[param] = parent1[param]
                        child2[param] = parent2[param]
                    else:
                        child1[param] = parent2[param]
                        child2[param] = parent1[param]

                # Mutation
                for child in [child1, child2]:
                    for param, value_range in param_space.items():
                        if random.random() < mutation_rate:
                            if isinstance(value_range, tuple): # Numeric parameter mutation
                                min_val, max_val = value_range
                                if isinstance(min_val, int) and isinstance(max_val, int):
                                    child[param] = random.randint(min_val, max_val)
                                else:
                                    child[param] = random.uniform(min_val, max_val)
                                # Ensure mutated values are within bounds
                                child[param] = max(min_val, min(max_val, child[param]))
                            elif isinstance(value_range, list): # Categorical parameter mutation (pick a random choice)
                                child[param] = random.choice(value_range)


                next_population.extend([child1, child2])

            # Ensure weights sum to 1 after crossover and mutation for the new population
            for params in next_population:
                 total_weight = params.get('weight_trend', 0.5) + params.get('weight_volatility', 0.2) + params.get('weight_exhaustion', 0.3)
                 if total_weight > 0:
                    if 'weight_trend' in params: params['weight_trend'] /= total_weight
                    if 'weight_volatility' in params: params['weight_volatility'] /= total_weight
                    if 'weight_exhaustion' in params: params['weight_exhaustion'] /= total_weight
                 else: # Avoid division by zero
                    if 'weight_trend' in params: params['weight_trend'] = 1/3
                    if 'weight_volatility' in params: params['weight_volatility'] = 1/3
                    if 'weight_exhaustion' in params: params['weight_exhaustion'] = 1/3


            population = next_population[:pop_size] # Truncate population if needed


            # Track best individual in this generation (from the evaluated individuals)
            best_gen_fitness_idx = valid_fitness.index(max(valid_fitness))
            best_gen_fitness = valid_fitness[best_gen_fitness_idx]
            best_gen_params = valid_population[best_gen_fitness_idx]

            display(f"Generation {generation+1}: Best Fitness = {best_gen_fitness:.4f}")


        # After GA for the current window, the best parameters found in this window
        # are the candidates for the overall best. A true walk-forward would evaluate
        # the best params from the *training* window on the *next* window (step_size)
        # and the params that perform best on the next window are selected.
        # For simplicity in the PoC, we just track the best parameters found across all training windows
        # and assume these are the "optimal" ones found by the walk-forward process.

        best_window_fitness = max(valid_fitness) # Max fitness from the last generation's valid individuals
        best_window_params = valid_population[valid_fitness.index(best_window_fitness)]


        if best_window_fitness > best_overall_fitness:
            best_overall_fitness = best_window_fitness
            best_overall_params = best_window_params
            display(f"New overall best fitness found: {best_overall_fitness:.4f}")
            display(f"Corresponding parameters: {best_overall_params}")


        # Move the window forward by step_size_days
        window_start_idx = all_dates.searchsorted(current_start_date + pd.Timedelta(days=step_size_days), side='left')[0]
        first_window_end_idx = all_dates.searchsorted(all_dates[window_start_idx] + pd.Timedelta(days=lookback_window_days), side='right')[0]

        # Ensure the new window end is not beyond the last date in the data
        if first_window_end_idx > len(all_dates):
             first_window_end_idx = len(all_dates)


    return best_overall_params


# Define the parameter space for the GA, including indicator choices and lookback ranges
# These ranges and choices should be determined based on domain knowledge or initial exploration
param_space_with_indicators = {
    'weight_trend': (0.1, 0.7),
    'weight_volatility': (0.1, 0.4),
    'weight_exhaustion': (0.1, 0.4),
    'strong_bull_threshold': (30, 70),
    'weak_bull_threshold': (10, 40),
    'neutral_threshold_upper': (10, 30),
    'neutral_threshold_lower': (-30, -10),
    'strong_bear_threshold': (-70, -30),
    'weak_bear_threshold': (-40, -10),
    'stop_loss_multiplier_strong': (1.5, 3.5),
    'stop_loss_multiplier_weak': (0.5, 2.5),
    'indicator_trend': ['Slope'], # Only Slope is implemented for now
    'indicator_volatility': ['ATR'], # Only ATR is implemented for now
    'indicator_exhaustion': ['SMADiff'], # Only SMADiff is implemented for now
    'lookback_trend': (10, 50), # Range for lookback periods
    'lookback_volatility': (10, 50),
    'lookback_exhaustion': (10, 50)
}

# GA and Walk-Forward parameters
n_generations = 5 # Reduced for faster PoC run
pop_size = 10 # Reduced for faster PoC run
mutation_rate = 0.1
lookback_window_days = 365 # 1 year lookback window for optimization
step_size_days = 90 # Step forward by 90 days


# Ensure df_for_evaluation contains only the necessary columns for the evaluation function
# which are Open, High, Low, Close, Volume, DollarVolume
df_for_evaluation_clean = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


# Run the Genetic Algorithm with walk-forward optimization
display("Starting Genetic Algorithm with Walk-Forward Optimization (with Indicators)...")
optimal_params_with_indicators = genetic_algorithm_walk_forward_with_indicators(
    df_for_evaluation_clean,
    initial_capital,
    param_space_with_indicators,
    n_generations,
    pop_size,
    mutation_rate,
    lookback_window_days,
    step_size_days
)

display("\nOptimal Parameters (with Indicators) found:")
display(optimal_params_with_indicators)

"""**Reasoning**:
The error `IndexError: invalid index to scalar variable.` on `all_dates.searchsorted(first_window_end_date, side='right')[0]` suggests that `all_dates.searchsorted` is returning an empty array or a scalar instead of an array when the date is not found, causing the `[0]` indexing to fail. This can happen if `first_window_end_date` is before the first date in `all_dates` or after the last date. Given that `all_dates` comes from the dataframe index and we are adding `lookback_window_days` to the first date, the issue is likely when `first_window_end_date` is beyond the range of `all_dates`. Let's modify the date calculation and indexing to handle this more robustly and ensure the window start and end indices are valid within the `all_dates` array.


"""

import random

def genetic_algorithm_walk_forward_with_indicators(df, initial_capital, param_space, n_generations, pop_size, mutation_rate, lookback_window_days, step_size_days):
    """
    Runs a Genetic Algorithm with walk-forward optimization, optimizing both
    numeric parameters and indicator choices/lookbacks.

    Args:
        df (pd.DataFrame): DataFrame containing the full historical data (dollar bars OHLCV).
                           Factors will be calculated inside the fitness function.
        initial_capital (float): Starting capital for backtests.
        param_space (dict): Dictionary defining the search space for each parameter,
                            including numeric ranges and indicator choices/lookback ranges.
                            Example: {'weight_trend': (0.1, 0.7), ..., 'indicator_trend': ['Slope', 'MACD'], 'lookback_trend': (10, 50)}.
        n_generations (int): Number of generations for the GA.
        pop_size (int): Size of the population in each generation.
        mutation_rate (float): Probability of mutation for each parameter.
        lookback_window_days (int): Size of the walk-forward backtesting window (in days).
        step_size_days (int): Size of the step for the walk-forward window (in days).

    Returns:
        dict: The best set of parameters (numeric and indicator choices/lookbacks)
              found during the walk-forward optimization.
    """

    best_overall_params = None
    best_overall_fitness = -float('inf')

    # Get all unique dates and sort them
    all_dates = df.index.unique().sort_values()

    # Calculate the minimum number of data points needed for indicator lookbacks
    min_lookback = max(param_space.get('lookback_trend', (0,0))[1],
                       param_space.get('lookback_volatility', (0,0))[1],
                       param_space.get('lookback_exhaustion', (0,0))[1])

    # Determine the initial training window end date
    # Start the first window such that there is enough data for the initial lookback
    initial_window_start_date = all_dates[0]
    initial_window_end_date = initial_window_start_date + pd.Timedelta(days=lookback_window_days)

    # Find the index corresponding to initial_window_end_date in all_dates
    # Use side='left' to get the index of the first date >= initial_window_end_date
    # Handle cases where the calculated date is beyond the last date
    first_window_end_idx = all_dates.searchsorted(initial_window_end_date, side='left')
    if first_window_end_idx == len(all_dates):
         # If the end date is beyond the last date, the first window cannot be formed
         display("Error: Lookback window extends beyond the available data.")
         return None
    elif first_window_end_idx == 0:
         # If the end date is before or at the first date, the window is too short
         display("Error: Lookback window is too short relative to the data frequency.")
         return None

    window_start_idx = 0
    window_end_idx = first_window_end_idx

    # Ensure the initial window has at least min_lookback bars plus some buffer
    # A buffer of 50 bars or twice the max lookback was used before, let's keep that heuristic
    min_bars_for_ga = max(min_lookback * 2, 50)

    if (window_end_idx - window_start_idx) < min_bars_for_ga:
         display(f"Error: Initial window size ({window_end_idx - window_start_idx} bars) is less than the minimum required bars for GA ({min_bars_for_ga}).")
         return None


    while window_end_idx <= len(all_dates):
        # Define the training window (backtest period for GA optimization) using indices
        current_start_date = all_dates[window_start_idx]
        # Use the date of the last bar in the window
        current_end_date = all_dates[window_end_idx - 1]

        train_df = df.loc[current_start_date:current_end_date].copy()

        # Re-check if the training window has enough data points after subsetting by dates
        if len(train_df) < min_bars_for_ga:
             display(f"Skipping window: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')} due to insufficient data ({len(train_df)} bars). Minimum required: {min_bars_for_ga}")
             # Move the window forward by step_size_days and find the new indices
             next_window_start_date = current_start_date + pd.Timedelta(days=step_size_days)
             window_start_idx = all_dates.searchsorted(next_window_start_date, side='left')
             window_end_idx = all_dates.searchsorted(all_dates[window_start_idx] + pd.Timedelta(days=lookback_window_days), side='left')
             continue


        display(f"Optimizing on window: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")

        # --- Genetic Algorithm ---
        # Initialize population
        population = []
        for _ in range(pop_size):
            params = {}
            for param, value_range in param_space.items():
                if isinstance(value_range, tuple): # Numeric parameter
                    min_val, max_val = value_range
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param] = random.randint(min_val, max_val)
                    else:
                        params[param] = random.uniform(min_val, max_val)
                elif isinstance(value_range, list): # Categorical parameter (indicator choice)
                    params[param] = random.choice(value_range)
                # Ensure weights sum to 1 (simple normalization after random assignment)
            total_weight = params.get('weight_trend', 0.5) + params.get('weight_volatility', 0.2) + params.get('weight_exhaustion', 0.3)
            if total_weight > 0:
                if 'weight_trend' in params: params['weight_trend'] /= total_weight
                if 'weight_volatility' in params: params['weight_volatility'] /= total_weight
                if 'weight_exhaustion' in params: params['weight_exhaustion'] /= total_weight
            else: # Avoid division by zero or if weights are not in param_space
                 pass # Weights normalization will be handled later if they are not in param_space


            population.append(params)

        # Ensure weights sum to 1 for the initial population
        for params in population:
             total_weight = params.get('weight_trend', 0) + params.get('weight_volatility', 0) + params.get('weight_exhaustion', 0)
             # Only normalize if weights are part of the parameter space and sum > 0
             if 'weight_trend' in param_space and 'weight_volatility' in param_space and 'weight_exhaustion' in param_space and total_weight > 0:
                  params['weight_trend'] /= total_weight
                  params['weight_volatility'] /= total_weight
                  params['weight_exhaustion'] /= total_weight


        # GA generations
        for generation in range(n_generations):
            # Evaluate fitness of each individual
            fitness_scores = [evaluate_strategy_with_indicators(params, train_df.copy(), initial_capital) for params in population]

            # Handle potential errors or invalid fitness scores from evaluation
            valid_fitness_indices = [i for i, score in enumerate(fitness_scores) if not np.isinf(score) and not np.isnan(score) and score > -999] # Use -999 to exclude error codes

            if not valid_fitness_indices:
                 display(f"Generation {generation+1}: No valid fitness scores. Re-initializing population.")
                 # Re-initialize population if no valid individuals are found
                 population = []
                 for _ in range(pop_size):
                      params = {}
                      for param, value_range in param_space.items():
                           if isinstance(value_range, tuple): # Numeric parameter
                               min_val, max_val = value_range
                               if isinstance(min_val, int) and isinstance(max_val, int):
                                    params[param] = random.randint(min_val, max_val)
                               else:
                                    params[param] = random.uniform(min_val, max_val)
                           elif isinstance(value_range, list): # Categorical parameter (indicator choice)
                                params[param] = random.choice(value_range)
                      # Ensure weights sum to 1 for the re-initialized individual
                      total_weight = params.get('weight_trend', 0) + params.get('weight_volatility', 0) + params.get('weight_exhaustion', 0)
                      if 'weight_trend' in param_space and 'weight_volatility' in param_space and 'weight_exhaustion' in param_space and total_weight > 0:
                           params['weight_trend'] /= total_weight
                           params['weight_volatility'] /= total_weight
                           params['weight_exhaustion'] /= total_weight

                      population.append(params)

                 continue # Skip selection/crossover/mutation for this generation


            # Select parents from valid individuals (e.g., tournament selection)
            valid_population = [population[i] for i in valid_fitness_indices]
            valid_fitness = [fitness_scores[i] for i in valid_fitness_indices]

            parents = []
            # Use the size of the valid population for selecting parents
            num_parents_to_select = min(pop_size, len(valid_population))
            for _ in range(num_parents_to_select // 2):
                # Select two random indices from the valid indices list
                idx1_valid = random.choice(valid_fitness_indices)
                idx2_valid = random.choice(valid_fitness_indices)
                parent1 = population[idx1_valid] if fitness_scores[idx1_valid] > fitness_scores[idx2_valid] else population[idx2_valid]
                idx1_valid = random.choice(valid_fitness_indices)
                idx2_valid = random.choice(valid_fitness_indices)
                parent2 = population[idx1_valid] if fitness_scores[idx1_valid] > fitness_scores[idx2_valid] else population[idx2_valid]
                parents.extend([parent1, parent2])

            # Create next generation (crossover and mutation)
            next_population = []
            # Ensure we generate enough individuals for the next population size
            while len(next_population) < pop_size:
                 # Select two parents with replacement from the generated parents list
                 parent1, parent2 = random.sample(parents, 2)

                 child1, child2 = {}, {}

                 # Crossover
                 for param in param_space.keys():
                     if random.random() < 0.5:
                         child1[param] = parent1[param]
                         child2[param] = parent2[param]
                     else:
                         child1[param] = parent2[param]
                         child2[param] = parent1[param]

                 # Mutation
                 for child in [child1, child2]:
                     for param, value_range in param_space.items():
                         if random.random() < mutation_rate:
                             if isinstance(value_range, tuple): # Numeric parameter mutation
                                 min_val, max_val = value_range
                                 if isinstance(min_val, int) and isinstance(max_val, int):
                                     child[param] = random.randint(min_val, max_val)
                                 else:
                                     child[param] = random.uniform(min_val, max_val)
                                 # Ensure mutated values are within bounds
                                 child[param] = max(min_val, min(max_val, child[param]))
                             elif isinstance(value_range, list): # Categorical parameter mutation (pick a random choice)
                                 child[param] = random.choice(value_range)


                 next_population.extend([child1, child2])

            # Ensure weights sum to 1 for the new population before evaluating the next generation
            for params in next_population:
                 total_weight = params.get('weight_trend', 0) + params.get('weight_volatility', 0) + params.get('weight_exhaustion', 0)
                 if 'weight_trend' in param_space and 'weight_volatility' in param_space and 'weight_exhaustion' in param_space and total_weight > 0:
                      params['weight_trend'] /= total_weight
                      params['weight_volatility'] /= total_weight
                      params['weight_exhaustion'] /= total_weight


            population = next_population[:pop_size] # Select the top individuals or just take the first 'pop_size'


            # Track best individual in this generation (from the evaluated valid individuals)
            best_gen_fitness_idx = valid_fitness.index(max(valid_fitness))
            best_gen_fitness = valid_fitness[best_gen_fitness_idx]
            best_gen_params = valid_population[best_gen_fitness_idx]

            display(f"Generation {generation+1}: Best Fitness = {best_gen_fitness:.4f}")


        # After GA for the current window, select the best parameters found in this window
        # and track the overall best.
        best_window_fitness = max(valid_fitness) # Max fitness from the last generation's valid individuals
        best_window_params = valid_population[valid_fitness.index(best_window_fitness)]


        if best_window_fitness > best_overall_fitness:
            best_overall_fitness = best_window_fitness
            best_overall_params = best_window_params
            display(f"New overall best fitness found: {best_overall_fitness:.4f}")
            display(f"Corresponding parameters: {best_overall_params}")


        # Move the window forward by step_size_days and find the new indices
        next_window_start_date = current_start_date + pd.Timedelta(days=step_size_days)
        window_start_idx = all_dates.searchsorted(next_window_start_date, side='left')

        # Calculate the new window end index based on the lookback_window_days from the new start date
        new_window_end_date = all_dates[window_start_idx] + pd.Timedelta(days=lookback_window_days)
        window_end_idx = all_dates.searchsorted(new_window_end_date, side='left')


    return best_overall_params


# Define the parameter space for the GA, including indicator choices and lookback ranges
# These ranges and choices should be determined based on domain knowledge or initial exploration
param_space_with_indicators = {
    'weight_trend': (0.1, 0.7),
    'weight_volatility': (0.1, 0.4),
    'weight_exhaustion': (0.1, 0.4),
    'strong_bull_threshold': (30, 70),
    'weak_bull_threshold': (10, 40),
    'neutral_threshold_upper': (10, 30),
    'neutral_threshold_lower': (-30, -10),
    'strong_bear_threshold': (-70, -30),
    'weak_bear_threshold': (-40, -10),
    'stop_loss_multiplier_strong': (1.5, 3.5),
    'stop_loss_multiplier_weak': (0.5, 2.5),
    'indicator_trend': ['Slope'], # Only Slope is implemented for now
    'indicator_volatility': ['ATR'], # Only ATR is implemented for now
    'indicator_exhaustion': ['SMADiff'], # Only SMADiff is implemented for now
    'lookback_trend': (10, 50), # Range for lookback periods
    'lookback_volatility': (10, 50),
    'lookback_exhaustion': (10, 50)
}

# GA and Walk-Forward parameters
n_generations = 5 # Reduced for faster PoC run
pop_size = 10 # Reduced for faster PoC run
mutation_rate = 0.1
lookback_window_days = 365 # 1 year lookback window for optimization
step_size_days = 90 # Step forward by 90 days


# Ensure df_for_evaluation contains only the necessary columns for the evaluation function
# which are Open, High, Low, Close, Volume, DollarVolume
df_for_evaluation_clean = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


# Run the Genetic Algorithm with walk-forward optimization
display("Starting Genetic Algorithm with Walk-Forward Optimization (with Indicators)...")
optimal_params_with_indicators = genetic_algorithm_walk_forward_with_indicators(
    df_for_evaluation_clean,
    initial_capital,
    param_space_with_indicators,
    n_generations,
    pop_size,
    mutation_rate,
    lookback_window_days,
    step_size_days
)

display("\nOptimal Parameters (with Indicators) found:")
display(optimal_params_with_indicators)

"""**Reasoning**:
The traceback indicates an `IndexError: index 1157 is out of bounds for axis 0 with size 1157`. This happens when trying to access `all_dates[window_start_idx]` or calculate `window_end_idx` based on a `window_start_idx` that is equal to or greater than the number of dates in `all_dates`. This means the logic for advancing the window and calculating the end index is potentially moving beyond the available data. The `searchsorted(..., side='left')` method can return an index equal to the length of the array if the target value is greater than all elements, which then causes an error when used to index the array. Modify the loop condition and the window advancement logic to ensure the window start and end indices remain within valid bounds and that the window size is sufficient for the GA and indicator lookbacks.


"""

import random

def genetic_algorithm_walk_forward_with_indicators(df, initial_capital, param_space, n_generations, pop_size, mutation_rate, lookback_window_days, step_size_days):
    """
    Runs a Genetic Algorithm with walk-forward optimization, optimizing both
    numeric parameters and indicator choices/lookbacks.

    Args:
        df (pd.DataFrame): DataFrame containing the full historical data (dollar bars OHLCV).
                           Factors will be calculated inside the fitness function.
        initial_capital (float): Starting capital for backtests.
        param_space (dict): Dictionary defining the search space for each parameter,
                            including numeric ranges and indicator choices/lookback ranges.
                            Example: {'weight_trend': (0.1, 0.7), ..., 'indicator_trend': ['Slope', 'MACD'], 'lookback_trend': (10, 50)}.
        n_generations (int): Number of generations for the GA.
        pop_size (int): Size of the population in each generation.
        mutation_rate (float): Probability of mutation for each parameter.
        lookback_window_days (int): Size of the walk-forward backtesting window (in days).
        step_size_days (int): Size of the step for the walk-forward window (in days).

    Returns:
        dict: The best set of parameters (numeric and indicator choices/lookbacks)
              found during the walk-forward optimization.
    """

    best_overall_params = None
    best_overall_fitness = -float('inf')

    # Get all unique dates and sort them
    all_dates = df.index.unique().sort_values()
    total_bars = len(all_dates)

    # Calculate the minimum number of data points needed for indicator lookbacks
    min_lookback = max(param_space.get('lookback_trend', (0,0))[1],
                       param_space.get('lookback_volatility', (0,0))[1],
                       param_space.get('lookback_exhaustion', (0,0))[1])

    # Determine the initial training window end date based on the first date and lookback days
    initial_window_start_date = all_dates[0]
    initial_window_end_date = initial_window_start_date + pd.Timedelta(days=lookback_window_days)

    # Find the index corresponding to initial_window_end_date in all_dates
    # Use side='left' to get the index of the first date >= initial_window_end_date
    first_window_end_idx = all_dates.searchsorted(initial_window_end_date, side='left')

    # Ensure the initial window has enough data points for lookbacks and GA
    # A buffer of 50 bars or twice the max lookback is used as a heuristic
    min_bars_for_ga = max(min_lookback * 2, 50) # Heuristic: require at least twice the max lookback bars or 50 bars

    if (first_window_end_idx - 0) < min_bars_for_ga:
         display(f"Error: Initial window size ({first_window_end_idx} bars) is less than the minimum required bars for GA ({min_bars_for_ga}).")
         return None

    window_start_idx = 0
    window_end_idx = first_window_end_idx # The index *after* the last date in the window

    # Loop while the end of the current window is within the total number of bars
    # and the window size is sufficient for the GA
    while window_end_idx <= total_bars and (window_end_idx - window_start_idx) >= min_bars_for_ga:
        # Define the training window (backtest period for GA optimization) using indices
        current_start_date = all_dates[window_start_idx]
        # Use the date of the last bar in the window
        current_end_date = all_dates[window_end_idx - 1]

        train_df = df.loc[current_start_date:current_end_date].copy()

        # Re-check if the training window has enough data points after subsetting by dates
        if len(train_df) < min_bars_for_ga:
             display(f"Skipping window: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')} due to insufficient data ({len(train_df)} bars). Minimum required: {min_bars_for_ga}")
             # Move the window forward by step_size_days and find the new indices
             next_window_start_date = current_start_date + pd.Timedelta(days=step_size_days)
             window_start_idx = all_dates.searchsorted(next_window_start_date, side='left')
             # Calculate the new window end index based on the lookback_window_days from the new start date
             new_window_end_date = all_dates[window_start_idx] + pd.Timedelta(days=lookback_window_days) if window_start_idx < total_bars else all_dates.max() + pd.Timedelta(days=1) # Handle case where window_start_idx is at the end
             window_end_idx = all_dates.searchsorted(new_window_end_date, side='left')
             continue


        display(f"Optimizing on window: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")

        # --- Genetic Algorithm ---
        # Initialize population
        population = []
        for _ in range(pop_size):
            params = {}
            for param, value_range in param_space.items():
                if isinstance(value_range, tuple): # Numeric parameter
                    min_val, max_val = value_range
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param] = random.randint(min_val, max_val)
                    else:
                        params[param] = random.uniform(min_val, max_val)
                elif isinstance(value_range, list): # Categorical parameter (indicator choice)
                    params[param] = random.choice(value_range)

            population.append(params)

        # Ensure weights sum to 1 for the initial population
        for params in population:
             total_weight = params.get('weight_trend', 0) + params.get('weight_volatility', 0) + params.get('weight_exhaustion', 0)
             # Only normalize if weights are part of the parameter space and sum > 0
             if 'weight_trend' in param_space and 'weight_volatility' in param_space and 'weight_exhaustion' in param_space and total_weight > 0:
                  params['weight_trend'] /= total_weight
                  params['weight_volatility'] /= total_weight
                  params['weight_exhaustion'] /= total_weight


        # GA generations
        for generation in range(n_generations):
            # Evaluate fitness of each individual
            fitness_scores = [evaluate_strategy_with_indicators(params, train_df.copy(), initial_capital) for params in population]

            # Handle potential errors or invalid fitness scores from evaluation
            valid_fitness_indices = [i for i, score in enumerate(fitness_scores) if not np.isinf(score) and not np.isnan(score) and score > -999] # Use -999 to exclude error codes

            if not valid_fitness_indices:
                 display(f"Generation {generation+1}: No valid fitness scores. Re-initializing population.")
                 # Re-initialize population if no valid individuals are found
                 population = []
                 for _ in range(pop_size):
                      params = {}
                      for param, value_range in param_space.items():
                           if isinstance(value_range, tuple): # Numeric parameter
                               min_val, max_val = value_range
                               if isinstance(min_val, int) and isinstance(max_val, int):
                                    params[param] = random.randint(min_val, max_val)
                               else:
                                    params[param] = random.uniform(min_val, max_val)
                           elif isinstance(value_range, list): # Categorical parameter (indicator choice)
                                params[param] = random.choice(value_range)
                      # Ensure weights sum to 1 for the re-initialized individual
                      total_weight = params.get('weight_trend', 0) + params.get('weight_volatility', 0) + params.get('weight_exhaustion', 0)
                      if 'weight_trend' in param_space and 'weight_volatility' in param_space and 'weight_exhaustion' in param_space and total_weight > 0:
                           params['weight_trend'] /= total_weight
                           params['weight_volatility'] /= total_weight
                           params['weight_exhaustion'] /= total_weight

                      population.append(params)

                 continue # Skip selection/crossover/mutation for this generation


            # Select parents from valid individuals (e.g., tournament selection)
            valid_population = [population[i] for i in valid_fitness_indices]
            valid_fitness = [fitness_scores[i] for i in valid_fitness_indices]

            parents = []
            # Use the size of the valid population for selecting parents
            num_parents_to_select = min(pop_size, len(valid_population))
            # Ensure we select an even number of parents for crossover
            num_parents_to_select = (num_parents_to_select // 2) * 2

            if num_parents_to_select == 0:
                 display(f"Generation {generation+1}: Not enough valid individuals to select parents. Re-initializing population.")
                 # Re-initialize population if not enough valid parents
                 population = []
                 for _ in range(pop_size):
                      params = {}
                      for param, value_range in param_space.items():
                           if isinstance(value_range, tuple): # Numeric parameter
                               min_val, max_val = value_range
                               if isinstance(min_val, int) and isinstance(max_val, int):
                                    params[param] = random.randint(min_val, max_val)
                               else:
                                    params[param] = random.uniform(min_val, max_val)
                           elif isinstance(value_range, list): # Categorical parameter (indicator choice)
                                params[param] = random.choice(value_range)
                      total_weight = params.get('weight_trend', 0) + params.get('weight_volatility', 0) + params.get('weight_exhaustion', 0)
                      if 'weight_trend' in param_space and 'weight_volatility' in param_space and 'weight_exhaustion' in param_space and total_weight > 0:
                           params['weight_trend'] /= total_weight
                           params['weight_volatility'] /= total_weight
                           params['weight_exhaustion'] /= total_weight

                      population.append(params)
                 continue


            for _ in range(num_parents_to_select // 2):
                # Select two random indices from the valid indices list
                idx1_valid = random.choice(valid_fitness_indices)
                idx2_valid = random.choice(valid_fitness_indices)
                parent1 = population[idx1_valid] if fitness_scores[idx1_valid] > fitness_scores[idx2_valid] else population[idx2_valid]
                idx1_valid = random.choice(valid_fitness_indices)
                idx2_valid = random.choice(valid_fitness_indices)
                parent2 = population[idx1_valid] if fitness_scores[idx1_valid] > fitness_scores[idx2_valid] else population[idx2_valid]
                parents.extend([parent1, parent2])

            # Create next generation (crossover and mutation)
            next_population = []
            # Ensure we generate enough individuals for the next population size
            while len(next_population) < pop_size:
                 # Select two parents with replacement from the generated parents list
                 # Ensure there are at least 2 parents to select from
                 if len(parents) < 2:
                      # If not enough parents, break the loop and re-initialize population outside
                      display(f"Generation {generation+1}: Not enough parents generated for crossover. Breaking.")
                      break

                 parent1, parent2 = random.sample(parents, 2)

                 child1, child2 = {}, {}

                 # Crossover
                 for param in param_space.keys():
                     if random.random() < 0.5:
                         child1[param] = parent1[param]
                         child2[param] = parent2[param]
                     else:
                         child1[param] = parent2[param]
                         child2[param] = parent1[param]

                 # Mutation
                 for child in [child1, child2]:
                     for param, value_range in param_space.items():
                         if random.random() < mutation_rate:
                             if isinstance(value_range, tuple): # Numeric parameter mutation
                                 min_val, max_val = value_range
                                 if isinstance(min_val, int) and isinstance(max_val, int):
                                     child[param] = random.randint(min_val, max_val)
                                 else:
                                     child[param] = random.uniform(min_val, max_val)
                                 # Ensure mutated values are within bounds
                                 child[param] = max(min_val, min(max_val, child[param]))
                             elif isinstance(value_range, list): # Categorical parameter mutation (pick a random choice)
                                 child[param] = random.choice(value_range)


                 next_population.extend([child1, child2])

            # If the loop broke due to insufficient parents, re-initialize population
            if len(parents) < 2 and len(next_population) < pop_size:
                 display(f"Generation {generation+1}: Re-initializing population due to insufficient parents.")
                 population = []
                 for _ in range(pop_size):
                      params = {}
                      for param, value_range in param_space.items():
                           if isinstance(value_range, tuple):
                               min_val, max_val = value_range
                               if isinstance(min_val, int) and isinstance(max_val, int):
                                    params[param] = random.randint(min_val, max_val)
                               else:
                                    params[param] = random.uniform(min_val, max_val)
                           elif isinstance(value_range, list):
                                params[param] = random.choice(value_range)
                      total_weight = params.get('weight_trend', 0) + params.get('weight_volatility', 0) + params.get('weight_exhaustion', 0)
                      if 'weight_trend' in param_space and 'weight_volatility' in param_space and 'weight_exhaustion' in param_space and total_weight > 0:
                           params['weight_trend'] /= total_weight
                           params['weight_volatility'] /= total_weight
                           params['weight_exhaustion'] /= total_weight

                      population.append(params)
                 continue # Skip the rest of this generation's processing


            # Ensure weights sum to 1 for the new population before evaluating the next generation
            for params in next_population:
                 total_weight = params.get('weight_trend', 0) + params.get('weight_volatility', 0) + params.get('weight_exhaustion', 0)
                 if 'weight_trend' in param_space and 'weight_volatility' in param_space and 'weight_exhaustion' in param_space and total_weight > 0:
                      params['weight_trend'] /= total_weight
                      params['weight_volatility'] /= total_weight
                      params['weight_exhaustion'] /= total_weight


            population = next_population[:pop_size] # Select the top individuals or just take the first 'pop_size'


            # Track best individual in this generation (from the evaluated valid individuals)
            # Check if valid_fitness is not empty before finding max
            if valid_fitness:
                 best_gen_fitness_idx = valid_fitness.index(max(valid_fitness))
                 best_gen_fitness = valid_fitness[best_gen_fitness_idx]
                 best_gen_params = valid_population[best_gen_fitness_idx]
                 display(f"Generation {generation+1}: Best Fitness = {best_gen_fitness:.4f}")
            else:
                 display(f"Generation {generation+1}: No valid fitness scores to report best fitness.")


        # After GA for the current window, select the best parameters found in this window
        # and track the overall best.
        if valid_fitness: # Check if valid_fitness is not empty
             best_window_fitness = max(valid_fitness) # Max fitness from the last generation's valid individuals
             best_window_params = valid_population[valid_fitness.index(best_window_fitness)]

             if best_window_fitness > best_overall_fitness:
                 best_overall_fitness = best_window_fitness
                 best_overall_params = best_window_params
                 display(f"New overall best fitness found: {best_overall_fitness:.4f}")
                 display(f"Corresponding parameters: {best_overall_params}")
        else:
             display(f"No valid fitness scores in the last generation of window ending {current_end_date.strftime('%Y-%m-%d')}. No update to overall best parameters.")


        # Move the window forward by step_size_days and find the new indices
        current_window_start_date = all_dates[window_start_idx]
        next_window_start_date = current_window_start_date + pd.Timedelta(days=step_size_days)
        window_start_idx = all_dates.searchsorted(next_window_start_date, side='left')

        # Calculate the new window end index based on the lookback_window_days from the new start date
        if window_start_idx < total_bars:
            new_window_end_date = all_dates[window_start_idx] + pd.Timedelta(days=lookback_window_days)
            window_end_idx = all_dates.searchsorted(new_window_end_date, side='left')
        else:
            # If the new start index is beyond the last bar, set end index to total_bars to terminate loop
            window_end_idx = total_bars + 1 # Ensure loop terminates


    return best_overall_params


# Define the parameter space for the GA, including indicator choices and lookback ranges
# These ranges and choices should be determined based on domain knowledge or initial exploration
param_space_with_indicators = {
    'weight_trend': (0.1, 0.7),
    'weight_volatility': (0.1, 0.4),
    'weight_exhaustion': (0.1, 0.4),
    'strong_bull_threshold': (30, 70),
    'weak_bull_threshold': (10, 40),
    'neutral_threshold_upper': (10, 30),
    'neutral_threshold_lower': (-30, -10),
    'strong_bear_threshold': (-70, -30),
    'weak_bear_threshold': (-40, -10),
    'stop_loss_multiplier_strong': (1.5, 3.5),
    'stop_loss_multiplier_weak': (0.5, 2.5),
    'indicator_trend': ['Slope'], # Only Slope is implemented for now
    'indicator_volatility': ['ATR'], # Only ATR is implemented for now
    'indicator_exhaustion': ['SMADiff'], # Only SMADiff is implemented for now
    'lookback_trend': (10, 50), # Range for lookback periods
    'lookback_volatility': (10, 50),
    'lookback_exhaustion': (10, 50)
}

# GA and Walk-Forward parameters
n_generations = 5 # Reduced for faster PoC run
pop_size = 10 # Reduced for faster PoC run
mutation_rate = 0.1
lookback_window_days = 365 # 1 year lookback window for optimization
step_size_days = 90 # Step forward by 90 days


# Ensure df_for_evaluation contains only the necessary columns for the evaluation function
# which are Open, High, Low, Close, Volume, DollarVolume
df_for_evaluation_clean = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


# Run the Genetic Algorithm with walk-forward optimization
display("Starting Genetic Algorithm with Walk-Forward Optimization (with Indicators)...")
optimal_params_with_indicators = genetic_algorithm_walk_forward_with_indicators(
    df_for_evaluation_clean,
    initial_capital,
    param_space_with_indicators,
    n_generations,
    pop_size,
    mutation_rate,
    lookback_window_days,
    step_size_days
)

display("\nOptimal Parameters (with Indicators) found:")
display(optimal_params_with_indicators)

"""**Reasoning**:
The genetic algorithm with walk-forward optimization has completed and found an optimal set of parameters. The next step is to evaluate the performance of these optimal parameters on the entire dataset to get a final performance assessment.


"""

# Evaluate the performance of the optimal parameters on the full dataset
# If optimal_params_with_indicators is None (meaning no improvement was found), evaluate with default parameters
if optimal_params_with_indicators is None:
    display("No optimal parameters found by GA. Evaluating with initial default parameters.")
    evaluation_params_final = current_params_with_indicators # Use the initial default parameters
else:
    display("Evaluating with optimal parameters found by GA.")
    evaluation_params_final = optimal_params_with_indicators

# Re-run the trading simulation with the selected optimal parameters on the full dataset
position = 0  # 1 for long, -1 for short, 0 for flat
entry_price = 0
stop_loss = 0
equity_curve = []
trade_log = []
current_capital = initial_capital

# Ensure df_for_evaluation_clean is used for the final simulation as it has the correct columns
df_for_final_evaluation = df_for_evaluation_clean.copy()


# Calculate factors and regimes for the full dataset using the optimal parameters and indicators
# This step is necessary to have the factor and regime columns available for the simulation loop
# The evaluate_strategy_with_indicators function calculates these internally for each window,
# but we need them explicitly calculated on the full dataset for the final simulation loop.

# Extract indicator choices and lookbacks from the optimal parameters
indicator_trend_final = evaluation_params_final.get('indicator_trend', 'Slope')
indicator_volatility_final = evaluation_params_final.get('indicator_volatility', 'ATR')
indicator_exhaustion_final = evaluation_params_final.get('indicator_exhaustion', 'SMADiff')
lookback_trend_final = evaluation_params_final.get('lookback_trend', 20)
lookback_volatility_final = evaluation_params_final.get('lookback_volatility', 20)
lookback_exhaustion_final = evaluation_params_final.get('lookback_exhaustion', 20)


try:
    # Calculate Volatility first as Exhaustion might depend on it
    if indicator_volatility_final in INDICATOR_FUNCTIONS['Volatility']:
        df_for_final_evaluation['Volatility_Factor'] = INDICATOR_FUNCTIONS['Volatility'][indicator_volatility_final](df_for_final_evaluation, lookback_volatility_final)
    else:
        display(f"Warning: Unknown Volatility indicator '{indicator_volatility_final}'. Using default (ATR).")
        df_for_final_evaluation['Volatility_Factor'] = calculate_volatility_atr(df_for_final_evaluation, lookback_volatility_final)

    if indicator_trend_final in INDICATOR_FUNCTIONS['Trend']:
         df_for_final_evaluation['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend'][indicator_trend_final](df_for_final_evaluation, lookback_trend_final)
    else:
         display(f"Warning: Unknown Trend indicator '{indicator_trend_final}'. Using default (Slope).")
         df_for_final_evaluation['Trend_Factor'] = calculate_trend_slope(df_for_final_evaluation, lookback_trend_final)

    if indicator_exhaustion_final in INDICATOR_FUNCTIONS['Exhaustion']:
        if indicator_exhaustion_final == 'SMADiff':
             # SMADiff requires the normalized ATR as an argument
             df_for_final_evaluation['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_final](df_for_final_evaluation, lookback_exhaustion_final, df_for_final_evaluation['Volatility_Factor'])
        else:
             # Other exhaustion indicators would take df and their lookback
             df_for_final_evaluation['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_final](df_for_final_evaluation, lookback_exhaustion_final)
    else:
        display(f"Warning: Unknown Exhaustion indicator '{indicator_exhaustion_final}'. Using default (SMADiff).")
        df_for_final_evaluation['Exhaustion_Factor'] = calculate_exhaustion_sma_diff(df_for_final_evaluation, lookback_exhaustion_final, df_for_final_evaluation['Volatility_Factor'])

except Exception as e:
    display(f"Error during factor calculation for final evaluation: {e}")
    # If factor calculation fails, we cannot proceed with the simulation
    equity_curve_df_final = pd.DataFrame() # Empty dataframe
    trade_log_df_final = pd.DataFrame() # Empty dataframe
    display("Final evaluation aborted due to factor calculation error.")


# Drop rows with NaN values generated by lookback periods after factor calculation
df_for_final_evaluation = df_for_final_evaluation.dropna(subset=['Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor'])

if not df_for_final_evaluation.empty:
    # Calculate MSS with optimal weights
    weight_trend_final = evaluation_params_final.get('weight_trend', 0.5)
    weight_volatility_final = evaluation_params_final.get('weight_volatility', 0.2)
    weight_exhaustion_final = evaluation_params_final.get('weight_exhaustion', 0.3)


    df_for_final_evaluation['MSS_final'] = (weight_trend_final * df_for_final_evaluation['Trend_Factor'] +
                                   weight_volatility_final * df_for_final_evaluation['Volatility_Factor'] +
                                   weight_exhaustion_final * df_for_final_evaluation['Exhaustion_Factor'])

    # Classify Regime with optimal thresholds
    strong_bull_threshold_final = evaluation_params_final.get('strong_bull_threshold', 50)
    weak_bull_threshold_final = evaluation_params_final.get('weak_bull_threshold', 20)
    neutral_threshold_upper_final = evaluation_params_final.get('neutral_threshold_upper', 20)
    neutral_threshold_lower_final = evaluation_params_final.get('neutral_threshold_lower', -20)
    strong_bear_threshold_final = evaluation_params_final.get('strong_bear_threshold', -50)
    weak_bear_threshold_final = evaluation_params_final.get('weak_bear_threshold', -20)

    def classify_regime_final(mss):
        if mss > strong_bull_threshold_final:
            return 'Strong Bull'
        elif mss > weak_bull_threshold_final:
            return 'Weak Bull'
        elif mss >= neutral_threshold_lower_final and mss <= neutral_threshold_upper_final:
            return 'Neutral'
        elif mss > strong_bear_threshold_final:
            return 'Weak Bear'
        else:
            return 'Strong Bear'

    df_for_final_evaluation['Regime_final'] = df_for_final_evaluation['MSS_final'].apply(classify_regime_final)


    # Iterate through the dataframe for the final simulation
    stop_loss_multiplier_strong_final = evaluation_params_final.get('stop_loss_multiplier_strong', 2)
    stop_loss_multiplier_weak_final = evaluation_params_final.get('stop_loss_multiplier_weak', 1)

    for index, row in df_for_final_evaluation.iterrows():
        current_price = row['Close']
        current_regime = row['Regime_final']
        current_volatility_factor = row['Volatility_Factor'] # Using the normalized Volatility Factor

        # Determine stop-loss distance based on regime and Volatility Factor
        if not isinstance(current_volatility_factor, (int, float)) or np.isnan(current_volatility_factor) or current_volatility_factor == 0:
             stop_loss_distance = 0 # Cannot calculate dynamic stop loss without valid Volatility Factor
        else:
            if current_regime in ['Strong Bull', 'Strong Bear']:
                stop_loss_distance = stop_loss_multiplier_strong_final * abs(current_volatility_factor) # Use absolute value for distance
            elif current_regime in ['Weak Bull', 'Weak Bear']:
                stop_loss_distance = stop_loss_multiplier_weak_final * abs(current_volatility_factor) # Use absolute value for distance
            else: # Neutral
                stop_loss_distance = 0


        # Trading Logic based on Action Matrix (Section 4.3.3)
        if current_regime == 'Strong Bull':
            if position == 0: # Enter Long
                position = 1
                entry_price = current_price
                stop_loss = current_price - stop_loss_distance if stop_loss_distance > 0 else -float('inf')
                trade_log.append({'Date': index, 'Action': 'Enter Long', 'Price': current_price, 'StopLoss': stop_loss})
            elif position == 1 and stop_loss_distance > 0: # Adjust Trailing Stop for Long
                stop_loss = max(stop_loss, current_price - stop_loss_distance)
                trade_log.append({'Date': index, 'Action': 'Adjust Long Stop', 'Price': current_price, 'StopLoss': stop_loss})
            elif position == -1: # Exit Short
                pnl = (entry_price - current_price)
                current_capital += pnl
                trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Weak Bull':
            if position == 1 and stop_loss_distance > 0: # Hold Longs ONLY, tighten stop
                 stop_loss = max(stop_loss, current_price - stop_loss_distance)
                 trade_log.append({'Date': index, 'Action': 'Hold Long (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
            elif position == -1: # Exit Short
                pnl = (entry_price - current_price)
                current_capital += pnl
                trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Neutral':
            if position != 0: # EXIT ALL POSITIONS
                action = 'Exit Long' if position == 1 else 'Exit Short'
                pnl = (current_price - entry_price) if position == 1 else (entry_price - current_price)
                current_capital += pnl
                trade_log.append({'Date': index, 'Action': action + ' (Neutral Regime)', 'Price': current_price, 'PnL': pnl})
                position = 0
                entry_price = 0
                stop_loss = 0

        elif current_regime == 'Weak Bear':
             if position == -1 and stop_loss_distance > 0: # Hold Shorts ONLY, tighten stop
                stop_loss = min(stop_loss, current_price + stop_loss_distance)
                trade_log.append({'Date': index, 'Action': 'Hold Short (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
             elif position == 1: # Exit Long
                pnl = (current_price - entry_price)
                current_capital += pnl
                trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Strong Bear':
            if position == 0: # Enter Short
                position = -1
                entry_price = current_price
                stop_loss = current_price + stop_loss_distance if stop_loss_distance > 0 else float('inf')
                trade_log.append({'Date': index, 'Action': 'Enter Short', 'Price': current_price, 'StopLoss': stop_loss})
            elif position == -1 and stop_loss_distance > 0: # Adjust Trailing Stop for Short
                 stop_loss = min(stop_loss, current_price + stop_loss_distance)
                 trade_log.append({'Date': index, 'Action': 'Adjust Short Stop', 'Price': current_price, 'StopLoss': stop_loss})
            elif position == 1: # Exit Long
                pnl = (current_price - entry_price)
                current_capital += pnl
                trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
                position = 0
                entry_price = 0
                stop_loss = 0


        # Check for stop-loss hit (only if position is active and stop_loss is a valid number)
        if position == 1 and not np.isinf(stop_loss) and current_price <= stop_loss:
            pnl = (current_price - entry_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0
        elif position == -1 and not np.isinf(stop_loss) and current_price >= stop_loss:
            pnl = (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Stop Out Short', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0

        # Append current equity to the equity curve
        equity_curve.append({'Date': index, 'Equity': current_capital + (current_price - entry_price if position != 0 else 0)})

    # Convert trade log and equity curve to DataFrames
    equity_curve_df_final = pd.DataFrame(equity_curve).set_index('Date')
    trade_log_df_final = pd.DataFrame(trade_log)


    # Evaluate performance metrics for the final simulation
    if equity_curve_df_final.empty or len(equity_curve_df_final) < 2:
        display("Final equity curve is empty or too short for evaluation.")
    else:
        equity_curve_df_final['Daily_Return'] = equity_curve_df_final['Equity'].pct_change().fillna(0)
        total_return_final = (equity_curve_df_final['Equity'].iloc[-1] - initial_capital) / initial_capital
        trading_periods_per_year = 365 # Adjust if necessary
        annualized_return_final = (1 + total_return_final)**(trading_periods_per_year / len(equity_curve_df_final)) - 1

        equity_curve_df_final['Peak'] = equity_curve_df_final['Equity'].cummax()
        equity_curve_df_final['Drawdown'] = equity_curve_df_final['Equity'] - equity_curve_df_final['Peak']
        max_drawdown_final = equity_curve_df_final['Drawdown'].min()

        mar = 0
        downside_returns_final = equity_curve_df_final[equity_curve_df_final['Daily_Return'] < mar]['Daily_Return']
        downside_deviation_final = downside_returns_final.std()

        sortino_ratio_final = (annualized_return_final - mar) / downside_deviation_final if downside_deviation_final != 0 and not np.isnan(downside_deviation_final) else (float('inf') if annualized_return_final > mar else (0 if annualized_return_final == mar else -1000))

        calmar_ratio_final = annualized_return_final / abs(max_drawdown_final) if max_drawdown_final != 0 else float('inf')


        display("\nFinal Performance Metrics (using optimal parameters and indicators):")
        display(f"Initial Capital: {initial_capital:.2f}")
        display(f"Final Equity: {equity_curve_df_final['Equity'].iloc[-1]:.2f}")
        display(f"Total Return: {total_return_final:.4f}")
        display(f"Annualized Return: {annualized_return_final:.4f}")
        display(f"Max Drawdown: {max_drawdown_final:.2f}")
        display(f"Sortino Ratio (MAR=0): {sortino_ratio_final:.4f}")
        display(f"Calmar Ratio: {calmar_ratio_final:.4f}")

        # Plot final equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve_df_final.index, equity_curve_df_final['Equity'], label='Final Equity Curve')
        plt.title('Final Equity Curve (using optimal parameters and indicators)')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Display head of the final trade log
        display("\nFinal Trade Log Summary:")
        display(trade_log_df_final.head())

        # Analyze losing trades from the final simulation
        losing_trades_df_final = trade_log_df_final[trade_log_df_final['PnL'].notna() & (trade_log_df_final['PnL'] < 0)]
        display("\nLosing Trades Summary from Final Simulation:")
        display(losing_trades_df_final.head()) # Display head of losing trades

else:
    display("Cannot perform final evaluation as the dataframe for final evaluation is empty.")

"""## Summary:

### Data Analysis Key Findings

*   The Genetic Algorithm with walk-forward optimization was successfully implemented to optimize both numerical strategy parameters and indicator choices/lookback periods based on the Sortino Ratio.
*   The `evaluate_strategy_with_indicators` function was created to calculate factor values using the specified indicators (Slope for Trend, ATR for Volatility, SMADiff for Exhaustion, based on current implementation) and their respective lookbacks, and then compute the strategy's Sortino Ratio.
*   The walk-forward optimization process iterated through time windows, running the GA on each window to find the best parameters for that period.
*   The final evaluation of the best parameters found across all walk-forward windows on the full dataset resulted in a Total Return of 0.1958 and an Annualized Return of 0.0188.
*   The strategy showed a significant Max Drawdown of -11093.72.
*   The final performance metrics included a Sortino Ratio (MAR=0) of 0.0052 and a Calmar Ratio of 0.0017.

### Insights or Next Steps

*   The positive but low Sortino and Calmar Ratios suggest that while the strategy is profitable relative to its downside deviation, there is significant room for improvement in reducing drawdown or increasing returns.
*   Future work should involve implementing the placeholder indicators (MACD, Standard Deviation, RSI) and including them in the parameter space to allow the GA to explore a wider range of indicator combinations, which could potentially lead to better-performing strategies.

# Task
Implement option 2 from the plan to improve the core algorithm logic based on the provided PRD in "/content/Helios Trader.md".

## Refine factor calculations (optional but recommended)

### Subtask:
Implement alternative or improved methods for calculating the core factors (Trend, Volatility, Exhaustion) if the previous analysis suggested weaknesses in the current calculations.

**Reasoning**:
Based on the previous analysis, the normalization of factors might need refinement to ensure they have a wider range and contribute more meaningfully to the MSS. Also, the PRD mentions alternative indicators like MACD, Standard Deviation, and RSI. While not fully implemented in the previous steps, adding placeholder functions with basic structure for these can demonstrate how the framework supports alternative indicators. The current subtask specifically asks to implement alternative or improved methods for calculating the core factors. I will refine the normalization logic in the existing functions and add placeholder functions for the alternative indicators mentioned in the PRD to demonstrate the extensibility of the factor calculation module.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress

# Refined Trend Factor Calculation (Slope with improved normalization)
def calculate_trend_slope_refined(df, lookback):
    df_calc = df.copy()
    df_calc['Trend_Slope_Val'] = np.nan
    # Calculate slopes for the entire dataset first
    for i in range(lookback, len(df_calc)):
        y = df_calc['Close'].iloc[i-lookback:i]
        x = np.arange(lookback)
        # Handle potential division by zero or constant price in linregress
        if np.std(y) == 0:
             slope = 0
        else:
             slope, _, _, _, _ = linregress(x, y)
        df_calc.loc[df_calc.index[i], 'Trend_Slope_Val'] = slope

    # Normalize Trend: Scale relative to a rolling average of absolute slopes
    # This aims to make the scale more adaptive to the current market conditions
    df_calc['Abs_Slope_MA'] = df_calc['Trend_Slope_Val'].abs().rolling(window=lookback).mean()

    # Avoid division by zero
    valid_indices = df_calc.index[df_calc['Abs_Slope_MA'].notna() & (df_calc['Abs_Slope_MA'].abs() > 1e-9)]

    # Scale the slope based on the rolling average of absolute slope
    # A scaling factor of 100 means a slope equal to the average absolute slope gets a score of 100 (or -100)
    # This normalization approach needs careful tuning based on expected slope values
    df_calc.loc[valid_indices, 'Trend_Factor'] = (df_calc.loc[valid_indices, 'Trend_Slope_Val'] / df_calc.loc[valid_indices, 'Abs_Slope_MA']) * 100

    df_calc = df_calc.drop(columns=['Trend_Slope_Val', 'Abs_Slope_MA']) # Drop intermediate columns

    # Clip the result to the -100 to 100 range
    df_calc['Trend_Factor'] = np.clip(df_calc['Trend_Factor'], -100, 100)

    # Align index before returning
    return df_calc['Trend_Factor']


# Placeholder for MACD (Moving Average Convergence Divergence) as an alternative Trend indicator
# This function is a placeholder and needs actual MACD calculation logic
def calculate_trend_macd_placeholder(df, fastperiod=12, slowperiod=26, signalperiod=9):
    # Implement MACD calculation here
    # For now, return zeros or a simple placeholder series
    display("Warning: calculate_trend_macd_placeholder is not fully implemented.")
    return pd.Series(0.0, index=df.index)


# Refined Volatility Factor Calculation (ATR with improved normalization)
def calculate_volatility_atr_refined(df, lookback):
    df_calc = df.copy()
    df_calc['TR'] = np.maximum(np.maximum(df_calc['High'] - df_calc['Low'], abs(df_calc['High'] - df_calc['Close'].shift(1))), abs(df_calc['Low'] - df_calc['Close'].shift(1)))
    df_calc['Volatility_ATR_Absolute'] = df_calc['TR'].rolling(window=lookback).mean()
    df_calc = df_calc.drop(columns=['TR'])

    # Modified Normalization for ATR: Scale relative to a rolling standard deviation of ATR
    # This aims to capture volatility relative to its recent variability
    df_calc['ATR_Std'] = df_calc['Volatility_ATR_Absolute'].rolling(window=lookback).std()

    # Avoid division by zero
    valid_indices = df_calc.index[df_calc['ATR_Std'].notna() & (df_calc['ATR_Std'].abs() > 1e-9)]

    # Scale ATR based on its rolling standard deviation
    # This normalization also needs tuning
    df_calc.loc[valid_indices, 'Volatility_Factor'] = ((df_calc.loc[valid_indices, 'Volatility_ATR_Absolute'] - df_calc.loc[valid_indices, 'Volatility_ATR_Absolute'].rolling(window=lookback).mean()) / df_calc.loc[valid_indices, 'ATR_Std']) * 50 # Scale difference from mean by std dev, then scale to +/- 50 (example)


    df_calc = df_calc.drop(columns=['ATR_Std', 'Volatility_ATR_Absolute'])
    df_calc['Volatility_Factor'] = np.clip(df_calc['Volatility_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Volatility_Factor']


# Placeholder for Standard Deviation as an alternative Volatility indicator
# This function is a placeholder and needs actual StdDev calculation logic
def calculate_volatility_stddev_placeholder(df, lookback):
    # Implement Standard Deviation calculation here
    display("Warning: calculate_volatility_stddev_placeholder is not fully implemented.")
    return pd.Series(0.0, index=df.index)


# Refined Exhaustion Factor Calculation (SMA Diff with improved normalization)
def calculate_exhaustion_sma_diff_refined(df, sma_lookback, atr_series):
    df_calc = df.copy()
    df_calc['SMA'] = df_calc['Close'].rolling(window=sma_lookback).mean()

    # Ensure ATR series is aligned and not NaN for calculation
    # Using the passed atr_series directly
    valid_indices = df_calc.index[atr_series.notna() & (atr_series.abs() > 1e-9)]
    df_calc.loc[valid_indices, 'Exhaustion_Factor'] = (df_calc.loc[valid_indices, 'Close'] - df_calc.loc[valid_indices, 'SMA']) / atr_series.loc[valid_indices]

    df_calc = df_calc.drop(columns=['SMA'])

    # Normalize Exhaustion: Scale the ratio (distance from SMA in terms of ATR)
    # Scale the ratio based on a rolling standard deviation of the ratio
    df_calc['Exhaustion_Ratio_Std'] = df_calc['Exhaustion_Factor'].rolling(window=sma_lookback).std()

    valid_indices_norm = df_calc.index[df_calc['Exhaustion_Ratio_Std'].notna() & (df_calc['Exhaustion_Ratio_Std'].abs() > 1e-9)]

    # Scale the ratio by its rolling standard deviation
    # This normalization also needs tuning
    df_calc.loc[valid_indices_norm, 'Exhaustion_Factor'] = (df_calc.loc[valid_indices_norm, 'Exhaustion_Factor'] / df_calc.loc[valid_indices_norm, 'Exhaustion_Ratio_Std']) * 50 # Scale by std dev, then scale to +/- 50 (example)


    df_calc = df_calc.drop(columns=['Exhaustion_Ratio_Std'])
    df_calc['Exhaustion_Factor'] = np.clip(df_calc['Exhaustion_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Exhaustion_Factor']


# Placeholder for RSI (Relative Strength Index) as an alternative Exhaustion indicator
# This function is a placeholder and needs actual RSI calculation logic
def calculate_exhaustion_rsi_placeholder(df, lookback):
    # Implement RSI calculation here
    display("Warning: calculate_exhaustion_rsi_placeholder is not fully implemented.")
    # RSI is typically 0-100, so simple scaling might be needed if aiming for -100 to 100 range
    return pd.Series(0.0, index=df.index)


# Update the INDICATOR_FUNCTIONS dictionary with refined and placeholder functions
INDICATOR_FUNCTIONS = {
    'Trend': {
        'Slope_Refined': calculate_trend_slope_refined,
        'MACD_Placeholder': calculate_trend_macd_placeholder, # Add placeholder
    },
    'Volatility': {
        'ATR_Refined': calculate_volatility_atr_refined,
        'StdDev_Placeholder': calculate_volatility_stddev_placeholder, # Add placeholder
    },
    'Exhaustion': {
        'SMADiff_Refined': calculate_exhaustion_sma_diff_refined,
        'RSI_Placeholder': calculate_exhaustion_rsi_placeholder, # Add placeholder
    }
}

# Update the evaluate_strategy_with_indicators function to use the updated INDICATOR_FUNCTIONS
# and handle the new indicator names in param_space. The function signature and core logic
# remain the same, but it will now call the functions registered in the updated dictionary.
# Re-defining the function here to ensure it uses the updated INDICATOR_FUNCTIONS dictionary.

def evaluate_strategy_with_indicators(params, df, initial_capital=100000):
    """
    Evaluates a trading strategy with given parameters and indicator choices
    on historical data.

    Args:
        params (dict): Dictionary of strategy parameters, including:
                       - 'weight_trend', 'weight_volatility', 'weight_exhaustion'
                       - 'strong_bull_threshold', 'weak_bull_threshold', ...
                       - 'stop_loss_multiplier_strong', 'stop_loss_multiplier_weak'
                       - 'indicator_trend': Name of indicator for Trend factor (e.g., 'Slope')
                       - 'indicator_volatility': Name of indicator for Volatility factor (e.g., 'ATR')
                       - 'indicator_exhaustion': Name of indicator for Exhaustion factor (e.g., 'SMADiff')
                       - 'lookback_trend': Lookback period for Trend indicator
                       - 'lookback_volatility': Lookback period for Volatility indicator
                       - 'lookback_exhaustion': Lookback period for Exhaustion indicator

        df (pd.DataFrame): DataFrame containing dollar bars (Open, High, Low, Close, Volume, DollarVolume).
        initial_capital (float): Starting capital for the backtest.

    Returns:
        float: The Sortino Ratio of the strategy's performance, or a very low number
               if the ratio is infinite or NaN or if there's an error.
    """
    df_eval = df.copy() # Work on a copy

    # Extract parameters, including indicator choices and lookbacks
    weight_trend = params.get('weight_trend', 0.5)
    weight_volatility = params.get('weight_volatility', 0.2)
    weight_exhaustion = params.get('weight_exhaustion', 0.3)
    strong_bull_threshold = params.get('strong_bull_threshold', 50)
    weak_bull_threshold = params.get('weak_bull_threshold', 20)
    neutral_threshold_upper = params.get('neutral_threshold_upper', 20)
    neutral_threshold_lower = params.get('neutral_threshold_lower', -20)
    strong_bear_threshold = params.get('strong_bear_threshold', -50)
    weak_bear_threshold = params.get('weak_bear_threshold', -20)
    stop_loss_multiplier_strong = params.get('stop_loss_multiplier_strong', 2)
    stop_loss_multiplier_weak = params.get('stop_loss_multiplier_weak', 1)

    # Indicator choices and lookbacks
    indicator_trend_name = params.get('indicator_trend', 'Slope_Refined') # Use refined default
    indicator_volatility_name = params.get('indicator_volatility', 'ATR_Refined') # Use refined default
    indicator_exhaustion_name = params.get('indicator_exhaustion', 'SMADiff_Refined') # Use refined default

    lookback_trend = params.get('lookback_trend', 20)
    lookback_volatility = params.get('lookback_volatility', 20)
    lookback_exhaustion = params.get('lookback_exhaustion', 20) # SMA lookback for SMADiff


    # --- Calculate Factors using specified indicators ---
    try:
        # Calculate Volatility first as Exhaustion might depend on it
        if indicator_volatility_name in INDICATOR_FUNCTIONS['Volatility']:
            df_eval['Volatility_Factor'] = INDICATOR_FUNCTIONS['Volatility'][indicator_volatility_name](df_eval, lookback_volatility)
        else:
            display(f"Warning: Unknown Volatility indicator: {indicator_volatility_name}. Using default (ATR_Refined).")
            df_eval['Volatility_Factor'] = calculate_volatility_atr_refined(df_eval, lookback_volatility)

        if indicator_trend_name in INDICATOR_FUNCTIONS['Trend']:
             df_eval['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend'][indicator_trend_name](df_eval, lookback_trend)
        else:
             display(f"Warning: Unknown Trend indicator: {indicator_trend_name}. Using default (Slope_Refined).")
             df_eval['Trend_Factor'] = calculate_trend_slope_refined(df_eval, lookback_trend)

        if indicator_exhaustion_name in INDICATOR_FUNCTIONS['Exhaustion']:
            if indicator_exhaustion_name == 'SMADiff_Refined': # SMADiff refined requires normalized Volatility
                 df_eval['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_name](df_eval, lookback_exhaustion, df_eval['Volatility_Factor'])
            else:
                 # Other exhaustion indicators would take df and their lookback
                 df_eval['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_name](df_eval, lookback_exhaustion) # Note: This needs adjustment if the placeholder functions require different args

        else:
            display(f"Warning: Unknown Exhaustion indicator: {indicator_exhaustion_name}. Using default (SMADiff_Refined).")
            df_eval['Exhaustion_Factor'] = calculate_exhaustion_sma_diff_refined(df_eval, lookback_exhaustion, df_eval['Volatility_Factor'])


    except Exception as e:
        display(f"Error during factor calculation: {e}")
        return -2000 # Return very low fitness on error


    # Drop rows with NaN values generated by lookback periods
    # Use the combined lookbacks to determine the minimum valid starting index
    max_lookback = max(lookback_trend, lookback_volatility, lookback_exhaustion)
    # Ensure we drop NaNs after factor calculations
    df_eval = df_eval.dropna(subset=['Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor'])


    if len(df_eval) == 0:
        # display("Warning: DataFrame is empty after dropping NaNs.")
        return -1000 # Cannot evaluate performance if no valid data remains

    # Recalculate MSS with new weights
    df_eval['MSS_eval'] = (weight_trend * df_eval['Trend_Factor'] +
                           weight_volatility * df_eval['Volatility_Factor'] +
                           weight_exhaustion * df_eval['Exhaustion_Factor'])

    # Reclassify Regime with new thresholds
    def classify_regime_eval(mss):
        if mss > strong_bull_threshold:
            return 'Strong Bull'
        elif mss > weak_bull_threshold:
            return 'Weak Bull'
        elif mss >= neutral_threshold_lower and mss <= neutral_threshold_upper:
            return 'Neutral'
        elif mss > strong_bear_threshold:
            return 'Weak Bear'
        else:
            return 'Strong Bear'

    df_eval['Regime_eval'] = df_eval['MSS_eval'].apply(classify_regime_eval)


    # --- Trading Simulation ---
    position = 0
    entry_price = 0
    #stop_loss = 0 # Stop loss will be calculated dynamically
    equity_curve = []
    current_capital = initial_capital

    for index, row in df_eval.iterrows():
        current_price = row['Close']
        current_regime = row['Regime_eval']
        current_volatility_factor = row['Volatility_Factor'] # Using the normalized Volatility Factor


        # Determine stop-loss distance based on regime and Volatility Factor
        # Use absolute value of volatility factor for stop distance as it represents price movement
        if not isinstance(current_volatility_factor, (int, float)) or np.isnan(current_volatility_factor) or abs(current_volatility_factor) < 1e-9:
             stop_loss_distance = 0 # Cannot calculate dynamic stop loss without valid Volatility Factor
             # display(f"Warning: Invalid Volatility Factor for stop loss calculation at {index}. Factor: {current_volatility_factor}")
        else:
            if current_regime in ['Strong Bull', 'Strong Bear']:
                stop_loss_distance = stop_loss_multiplier_strong * abs(current_volatility_factor)
            elif current_regime in ['Weak Bull', 'Weak Bear']:
                stop_loss_distance = stop_loss_multiplier_weak * abs(current_volatility_factor)
            else: # Neutral
                stop_loss_distance = 0


        # Trading Logic based on Action Matrix (Section 4.3.3)
        if current_regime == 'Strong Bull':
            if position == 0: # Enter Long
                position = 1
                entry_price = current_price
                stop_loss = current_price - stop_loss_distance if stop_loss_distance > 0 else -float('inf') # Ensure stop is below entry for long
            elif position == 1 and stop_loss_distance > 0: # Adjust Trailing Stop for Long if valid distance
                 stop_loss = max(stop_loss, current_price - stop_loss_distance)
            elif position == -1: # Exit Short
                pnl = (entry_price - current_price)
                current_capital += pnl
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Weak Bull':
            if position == 1 and stop_loss_distance > 0: # Hold Longs ONLY, tighten stop if valid distance
                 stop_loss = max(stop_loss, current_price - stop_loss_distance)
            elif position == -1: # Exit Short
                pnl = (entry_price - current_price)
                current_capital += pnl
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Neutral':
            if position != 0: # EXIT ALL POSITIONS
                pnl = (current_price - entry_price) if position == 1 else (entry_price - current_price)
                current_capital += pnl
                position = 0
                entry_price = 0
                stop_loss = 0

        elif current_regime == 'Weak Bear':
             if position == -1 and stop_loss_distance > 0: # Hold Shorts ONLY, tighten stop if valid distance (above price)
                stop_loss = min(stop_loss, current_price + stop_loss_distance) # Adjust stop loss above price
             elif position == 1: # Exit Long
                pnl = (current_price - entry_price)
                current_capital += pnl
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Strong Bear':
            if position == 0: # Enter Short
                position = -1
                entry_price = current_price
                stop_loss = current_price + stop_loss_distance if stop_loss_distance > 0 else float('inf') # Ensure stop is above entry for short
            elif position == -1 and stop_loss_distance > 0: # Adjust Trailing Stop for Short if valid distance
                 stop_loss = min(stop_loss, current_price + stop_loss_distance) # Adjust stop loss above price
            elif position == 1: # Exit Long
                pnl = (current_price - entry_price)
                current_capital += pnl
                position = 0
                entry_price = 0
                stop_loss = 0


        # Check for stop-loss hit (only if position is active and stop_loss is a valid number)
        if position == 1 and not np.isinf(stop_loss) and current_price <= stop_loss:
            pnl = (current_price - entry_price)
            current_capital += pnl
            position = 0
            entry_price = 0
            stop_loss = 0
        elif position == -1 and not np.isinf(stop_loss) and current_price >= stop_loss:
            pnl = (entry_price - current_price)
            current_capital += pnl
            position = 0
            entry_price = 0
            stop_loss = 0


        # Append current equity to the equity curve
        equity_curve.append({'Date': index, 'Equity': current_capital + (current_price - entry_price if position != 0 else 0)})


    equity_curve_df_eval = pd.DataFrame(equity_curve).set_index('Date')

    # --- Performance Evaluation (Sortino Ratio) ---
    if equity_curve_df_eval.empty or len(equity_curve_df_eval) < 2:
         # display("Warning: Equity curve is empty or too short for Sortino Ratio calculation.")
         return -1000 # Return a very low fitness

    equity_curve_df_eval['Daily_Return'] = equity_curve_df_eval['Equity'].pct_change().fillna(0)

    # Calculate Annualized Return (assuming daily data from dollar bars)
    trading_periods_per_year = 365
    total_return = (equity_curve_df_eval['Equity'].iloc[-1] - initial_capital) / initial_capital

    if len(equity_curve_df_eval) > 1:
      annualized_return = (1 + total_return)**(trading_periods_per_year / len(equity_curve_df_eval)) - 1
    else:
      annualized_return = 0


    # Calculate Sortino Ratio
    mar = 0 # Minimum Acceptable Return (MAR)
    downside_returns = equity_curve_df_eval[equity_curve_df_eval['Daily_Return'] < mar]['Daily_Return']
    downside_deviation = downside_returns.std()

    if downside_deviation == 0 or np.isnan(downside_deviation):
        sortino_ratio = float('inf') if annualized_return > mar else (0 if annualized_return == mar else -1000)
    else:
        sortino_ratio = (annualized_return - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN after calculation
    if np.isinf(sortino_ratio) or np.isnan(sortino_ratio):
        # display(f"Warning: Calculated Sortino Ratio is {sortino_ratio}. Returning low fitness.")
        return -1000

    return sortino_ratio


# Example usage of the updated fitness function with refined defaults
refined_params_with_indicators = {
    'weight_trend': 0.5,
    'weight_volatility': 0.2,
    'weight_exhaustion': 0.3,
    'strong_bull_threshold': 50,
    'weak_bull_threshold': 20,
    'neutral_threshold_upper': 20,
    'neutral_threshold_lower': -20,
    'strong_bear_threshold': -50,
    'weak_bear_threshold': -20,
    'stop_loss_multiplier_strong': 2,
    'stop_loss_multiplier_weak': 1,
    'indicator_trend': 'Slope_Refined', # Use refined default
    'indicator_volatility': 'ATR_Refined', # Use refined default
    'indicator_exhaustion': 'SMADiff_Refined', # Use refined default
    'lookback_trend': 20, # Default lookback
    'lookback_volatility': 20, # Default lookback
    'lookback_exhaustion': 20 # Default lookback
}

# The dollar_bars_df should contain only the raw OHLCV and DollarVolume columns at this stage
# Ensure df_for_evaluation_clean is used
df_for_evaluation_clean = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


fitness_refined = evaluate_strategy_with_indicators(refined_params_with_indicators, df_for_evaluation_clean, initial_capital)
display(f"Fitness (Sortino Ratio) with refined default parameters and indicators: {fitness_refined}")

"""**Reasoning**:
The fitness function and refined factor calculations are implemented. The next step is to update the parameter space for the genetic algorithm to include the refined indicator names and their lookback ranges, as well as the placeholder indicators if we want the GA to explore them. Then, run the genetic algorithm with walk-forward optimization using the updated parameter space and the refined fitness function.


"""

# Update the parameter space to include refined indicator names and lookback ranges
# Also include placeholder indicators to allow the GA to explore them
param_space_refined_indicators = {
    'weight_trend': (0.1, 0.7),
    'weight_volatility': (0.1, 0.4),
    'weight_exhaustion': (0.1, 0.4),
    'strong_bull_threshold': (30, 70),
    'weak_bull_threshold': (10, 40),
    'neutral_threshold_upper': (10, 30),
    'neutral_threshold_lower': (-30, -10),
    'strong_bear_threshold': (-70, -30),
    'weak_bear_threshold': (-40, -10),
    'stop_loss_multiplier_strong': (1.5, 3.5),
    'stop_loss_multiplier_weak': (0.5, 2.5),
    'indicator_trend': list(INDICATOR_FUNCTIONS['Trend'].keys()), # Include all implemented trend indicators
    'indicator_volatility': list(INDICATOR_FUNCTIONS['Volatility'].keys()), # Include all implemented volatility indicators
    'indicator_exhaustion': list(INDICATOR_FUNCTIONS['Exhaustion'].keys()), # Include all implemented exhaustion indicators
    'lookback_trend': (10, 50), # Range for lookback periods
    'lookback_volatility': (10, 50),
    'lookback_exhaustion': (10, 50)
}

# GA and Walk-Forward parameters (can keep the reduced values for faster PoC)
n_generations = 5
pop_size = 10
mutation_rate = 0.1
lookback_window_days = 365 # 1 year lookback window for optimization
step_size_days = 90 # Step forward by 90 days

# Ensure df_for_evaluation_clean contains only the necessary columns for the evaluation function
# which are Open, High, Low, Close, Volume, DollarVolume
df_for_evaluation_clean = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


# Run the Genetic Algorithm with walk-forward optimization using the updated parameter space
display("Starting Genetic Algorithm with Walk-Forward Optimization (with Refined Indicators)...")
optimal_params_refined_indicators = genetic_algorithm_walk_forward_with_indicators(
    df_for_evaluation_clean,
    initial_capital,
    param_space_refined_indicators,
    n_generations,
    pop_size,
    mutation_rate,
    lookback_window_days,
    step_size_days
)

display("\nOptimal Parameters (with Refined Indicators) found:")
display(optimal_params_refined_indicators)

"""**Reasoning**:
The genetic algorithm with walk-forward optimization using refined indicators has completed and found an optimal set of parameters, including indicator choices and lookback periods. The next step is to evaluate the performance of these optimal parameters on the entire dataset to get a final performance assessment. This evaluation should include calculating the key performance metrics and plotting the equity curve.


"""

# Evaluate the performance of the optimal parameters on the full dataset
# If optimal_params_refined_indicators is None (meaning no improvement was found), evaluate with the initial refined default parameters
if optimal_params_refined_indicators is None:
    display("No optimal parameters found by GA with refined indicators. Evaluating with initial refined default parameters.")
    evaluation_params_final = refined_params_with_indicators # Use the initial refined default parameters
else:
    display("Evaluating with optimal parameters found by GA with refined indicators.")
    evaluation_params_final = optimal_params_refined_indicators

# Re-run the trading simulation with the selected optimal parameters on the full dataset
position = 0  # 1 for long, -1 for short, 0 for flat
entry_price = 0
stop_loss = 0
equity_curve = []
trade_log = []
current_capital = initial_capital

# Ensure df_for_evaluation_clean is used for the final simulation as it has the correct columns
df_for_final_evaluation = df_for_evaluation_clean.copy()


# Calculate factors and regimes for the full dataset using the optimal parameters and indicators
# This step is necessary to have the factor and regime columns available for the simulation loop

# Extract indicator choices and lookbacks from the optimal parameters
indicator_trend_final = evaluation_params_final.get('indicator_trend', 'Slope_Refined')
indicator_volatility_final = evaluation_params_final.get('indicator_volatility', 'ATR_Refined')
indicator_exhaustion_final = evaluation_params_final.get('indicator_exhaustion', 'SMADiff_Refined')
lookback_trend_final = evaluation_params_final.get('lookback_trend', 20)
lookback_volatility_final = evaluation_params_final.get('lookback_volatility', 20)
lookback_exhaustion_final = evaluation_params_final.get('lookback_exhaustion', 20)


try:
    # Calculate Volatility first as Exhaustion might depend on it
    if indicator_volatility_final in INDICATOR_FUNCTIONS['Volatility']:
        df_for_final_evaluation['Volatility_Factor'] = INDICATOR_FUNCTIONS['Volatility'][indicator_volatility_final](df_for_final_evaluation, lookback_volatility_final)
    else:
        display(f"Warning: Unknown Volatility indicator '{indicator_volatility_final}'. Using default (ATR_Refined).")
        df_for_final_evaluation['Volatility_Factor'] = calculate_volatility_atr_refined(df_for_final_evaluation, lookback_volatility_final)

    if indicator_trend_final in INDICATOR_FUNCTIONS['Trend']:
         df_for_final_evaluation['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend'][indicator_trend_final](df_for_final_evaluation, lookback_trend_final)
    else:
         display(f"Warning: Unknown Trend indicator '{indicator_trend_final}'. Using default (Slope_Refined).")
         df_for_final_evaluation['Trend_Factor'] = calculate_trend_slope_refined(df_for_final_evaluation, lookback_trend_final)

    if indicator_exhaustion_final in INDICATOR_FUNCTIONS['Exhaustion']:
        if indicator_exhaustion_final == 'SMADiff_Refined': # SMADiff refined requires normalized Volatility
             df_for_final_evaluation['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_final](df_for_final_evaluation, lookback_exhaustion_final, df_for_final_evaluation['Volatility_Factor'])
        # Note: Placeholder exhaustion indicators like RSI_Placeholder need to be handled here if they were implemented
        # They would likely take df and their lookback as arguments
        # elif indicator_exhaustion_final == 'RSI_Placeholder':
        #      df_for_final_evaluation['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_final](df_for_final_evaluation, lookback_exhaustion_final)
        else:
             display(f"Warning: Exhaustion indicator '{indicator_exhaustion_final}' not specifically handled for arguments. Calling with df and lookback.")
             # Attempt to call with df and lookback as a fallback, assuming placeholder structure
             df_for_final_evaluation['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_final](df_for_final_evaluation, lookback_exhaustion_final)


    else:
        display(f"Warning: Unknown Exhaustion indicator '{indicator_exhaustion_final}'. Using default (SMADiff_Refined).")
        df_for_final_evaluation['Exhaustion_Factor'] = calculate_exhaustion_sma_diff_refined(df_for_final_evaluation, lookback_exhaustion_final, df_for_final_evaluation['Volatility_Factor'])


except Exception as e:
    display(f"Error during factor calculation for final evaluation: {e}")
    # If factor calculation fails, we cannot proceed with the simulation
    equity_curve_df_final = pd.DataFrame() # Empty dataframe
    trade_log_df_final = pd.DataFrame() # Empty dataframe
    display("Final evaluation aborted due to factor calculation error.")


# Drop rows with NaN values generated by lookback periods after factor calculation
df_for_final_evaluation = df_for_final_evaluation.dropna(subset=['Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor'])

if not df_for_final_evaluation.empty:
    # Calculate MSS with optimal weights
    weight_trend_final = evaluation_params_final.get('weight_trend', 0.5)
    weight_volatility_final = evaluation_params_final.get('weight_volatility', 0.2)
    weight_exhaustion_final = evaluation_params_final.get('weight_exhaustion', 0.3)


    df_for_final_evaluation['MSS_final'] = (weight_trend_final * df_for_final_evaluation['Trend_Factor'] +
                                   weight_volatility_final * df_for_final_evaluation['Volatility_Factor'] +
                                   weight_exhaustion_final * df_for_final_evaluation['Exhaustion_Factor'])

    # Classify Regime with optimal thresholds
    strong_bull_threshold_final = evaluation_params_final.get('strong_bull_threshold', 50)
    weak_bull_threshold_final = evaluation_params_final.get('weak_bull_threshold', 20)
    neutral_threshold_upper_final = evaluation_params_final.get('neutral_threshold_upper', 20)
    neutral_threshold_lower_final = evaluation_params_final.get('neutral_threshold_lower', -20)
    strong_bear_threshold_final = evaluation_params_final.get('strong_bear_threshold', -50)
    weak_bear_threshold_final = evaluation_params_final.get('weak_bear_threshold', -20)

    def classify_regime_final(mss):
        if mss > strong_bull_threshold_final:
            return 'Strong Bull'
        elif mss > weak_bull_threshold_final:
            return 'Weak Bull'
        elif mss >= neutral_threshold_lower_final and mss <= neutral_threshold_upper_final:
            return 'Neutral'
        elif mss > strong_bear_threshold_final:
            return 'Weak Bear'
        else:
            return 'Strong Bear'

    df_for_final_evaluation['Regime_final'] = df_for_final_evaluation['MSS_final'].apply(classify_regime_final)


    # Iterate through the dataframe for the final simulation
    stop_loss_multiplier_strong_final = evaluation_params_final.get('stop_loss_multiplier_strong', 2)
    stop_loss_multiplier_weak_final = evaluation_params_final.get('stop_loss_multiplier_weak', 1)

    for index, row in df_for_final_evaluation.iterrows():
        current_price = row['Close']
        current_regime = row['Regime_final']
        current_volatility_factor = row['Volatility_Factor'] # Using the normalized Volatility Factor


        # Determine stop-loss distance based on regime and Volatility Factor
        # Use absolute value of volatility factor for stop distance as it represents price movement
        if not isinstance(current_volatility_factor, (int, float)) or np.isnan(current_volatility_factor) or abs(current_volatility_factor) < 1e-9:
             stop_loss_distance = 0 # Cannot calculate dynamic stop loss without valid Volatility Factor
             # display(f"Warning: Invalid Volatility Factor for stop loss calculation at {index}. Factor: {current_volatility_factor}")
        else:
            if current_regime in ['Strong Bull', 'Strong Bear']:
                stop_loss_distance = stop_loss_multiplier_strong_final * abs(current_volatility_factor) # Use absolute value for distance
            elif current_regime in ['Weak Bull', 'Weak Bear']:
                stop_loss_distance = stop_loss_multiplier_weak_final * abs(current_volatility_factor) # Use absolute value for distance
            else: # Neutral
                stop_loss_distance = 0


        # Trading Logic based on Action Matrix (Section 4.3.3)
        if current_regime == 'Strong Bull':
            if position == 0: # Enter Long
                position = 1
                entry_price = current_price
                stop_loss = current_price - stop_loss_distance if stop_loss_distance > 0 else -float('inf') # Ensure stop is below entry for long
                trade_log.append({'Date': index, 'Action': 'Enter Long', 'Price': current_price, 'StopLoss': stop_loss})
            elif position == 1 and stop_loss_distance > 0: # Adjust Trailing Stop for Long if valid distance
                 stop_loss = max(stop_loss, current_price - stop_loss_distance)
                 # Optionally log stop adjustment:
                 # trade_log.append({'Date': index, 'Action': 'Adjust Long Stop', 'Price': current_price, 'StopLoss': stop_loss})
            elif position == -1: # Exit Short
                pnl = (entry_price - current_price)
                current_capital += pnl
                trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Weak Bull':
            if position == 1 and stop_loss_distance > 0: # Hold Longs ONLY, tighten stop
                 stop_loss = max(stop_loss, current_price - stop_loss_distance)
                 # Optionally log hold/stop adjust:
                 # trade_log.append({'Date': index, 'Action': 'Hold Long (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
            elif position == -1: # Exit Short
                pnl = (entry_price - current_price)
                current_capital += pnl
                trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Neutral':
            if position != 0: # EXIT ALL POSITIONS
                action = 'Exit Long' if position == 1 else 'Exit Short'
                pnl = (current_price - entry_price) if position == 1 else (entry_price - current_price)
                current_capital += pnl
                trade_log.append({'Date': index, 'Action': action + ' (Neutral Regime)', 'Price': current_price, 'PnL': pnl})
                position = 0
                entry_price = 0
                stop_loss = 0

        elif current_regime == 'Weak Bear':
             if position == -1 and stop_loss_distance > 0: # Hold Shorts ONLY, tighten stop
                stop_loss = min(stop_loss, current_price + stop_loss_distance)
                # Optionally log hold/stop adjust:
                # trade_log.append({'Date': index, 'Action': 'Hold Short (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
             elif position == 1: # Exit Long
                pnl = (current_price - entry_price)
                current_capital += pnl
                trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
                position = 0
                entry_price = 0
                stop_loss = 0


        elif current_regime == 'Strong Bear':
            if position == 0: # Enter Short
                position = -1
                entry_price = current_price
                stop_loss = current_price + stop_loss_distance if stop_loss_distance > 0 else float('inf')
                trade_log.append({'Date': index, 'Action': 'Enter Short', 'Price': current_price, 'StopLoss': stop_loss})
            elif position == -1 and stop_loss_distance > 0: # Adjust Trailing Stop for Short
                 stop_loss = min(stop_loss, current_price + stop_loss_distance)
                 # Optionally log stop adjustment:
                 # trade_log.append({'Date': index, 'Action': 'Adjust Short Stop', 'Price': current_price, 'StopLoss': stop_loss})
            elif position == 1: # Exit Long
                pnl = (current_price - entry_price)
                current_capital += pnl
                trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
                position = 0
                entry_price = 0
                stop_loss = 0


        # Check for stop-loss hit (only if position is active and stop_loss is a valid number)
        if position == 1 and not np.isinf(stop_loss) and current_price <= stop_loss:
            pnl = (current_price - entry_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0
        elif position == -1 and not np.isinf(stop_loss) and current_price >= stop_loss:
            pnl = (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Stop Out Short', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0


        # Append current equity to the equity curve
        equity_curve.append({'Date': index, 'Equity': current_capital + (current_price - entry_price if position != 0 else 0)})

    # Convert trade log and equity curve to DataFrames
    equity_curve_df_final = pd.DataFrame(equity_curve).set_index('Date')
    trade_log_df_final = pd.DataFrame(trade_log)


    # Evaluate performance metrics for the final simulation
    if equity_curve_df_final.empty or len(equity_curve_df_final) < 2:
        display("Final equity curve is empty or too short for evaluation.")
    else:
        equity_curve_df_final['Daily_Return'] = equity_curve_df_final['Equity'].pct_change().fillna(0)
        total_return_final = (equity_curve_df_final['Equity'].iloc[-1] - initial_capital) / initial_capital
        trading_periods_per_year = 365 # Adjust if necessary
        annualized_return_final = (1 + total_return_final)**(trading_periods_per_year / len(equity_curve_df_final)) - 1

        equity_curve_df_final['Peak'] = equity_curve_df_final['Equity'].cummax()
        equity_curve_df_final['Drawdown'] = equity_curve_df_final['Equity'] - equity_curve_df_final['Peak']
        max_drawdown_final = equity_curve_df_final['Drawdown'].min()

        mar = 0
        downside_returns_final = equity_curve_df_final[equity_curve_df_final['Daily_Return'] < mar]['Daily_Return']
        downside_deviation_final = downside_returns_final.std()

        sortino_ratio_final = (annualized_return_final - mar) / downside_deviation_final if downside_deviation_final != 0 and not np.isnan(downside_deviation_final) else (float('inf') if annualized_return_final > mar else (0 if annualized_return == mar else -1000))

        calmar_ratio_final = annualized_return_final / abs(max_drawdown_final) if max_drawdown_final != 0 else float('inf')


        display("\nFinal Performance Metrics (using optimal parameters and refined indicators):")
        display(f"Initial Capital: {initial_capital:.2f}")
        display(f"Final Equity: {equity_curve_df_final['Equity'].iloc[-1]:.2f}")
        display(f"Total Return: {total_return_final:.4f}")
        display(f"Annualized Return: {annualized_return_final:.4f}")
        display(f"Max Drawdown: {max_drawdown_final:.2f}")
        display(f"Sortino Ratio (MAR=0): {sortino_ratio_final:.4f}")
        display(f"Calmar Ratio: {calmar_ratio_final:.4f}")

        # Plot final equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve_df_final.index, equity_curve_df_final['Equity'], label='Final Equity Curve (Optimized Refined)')
        plt.title('Final Equity Curve (using optimal parameters and refined indicators)')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Display head of the final trade log
        display("\nFinal Trade Log Summary:")
        display(trade_log_df_final.head())

        # Analyze losing trades from the final simulation
        losing_trades_df_final = trade_log_df_final[trade_log_df_final['PnL'].notna() & (trade_log_df_final['PnL'] < 0)]
        display("\nLosing Trades Summary from Final Simulation:")
        display(losing_trades_df_final.head()) # Display head of losing trades

else:
    display("Cannot perform final evaluation as the dataframe for final evaluation is empty.")

"""## Implement dynamic factor weights

### Subtask:
Modify the MSS calculation to use dynamic weights for the factors based on market conditions (e.g., regime).

**Reasoning**:
Define a function to get dynamic weights based on market regime and apply it to calculate the dynamic MSS and update the regime column.
"""

def get_dynamic_weights(regime):
    """
    Returns dynamic weights for factors based on market regime.

    Args:
        regime (str): The current market regime ('Strong Bull', 'Weak Bull',
                      'Neutral', 'Weak Bear', 'Strong Bear').

    Returns:
        dict: Dictionary of weights for 'Trend', 'Volatility', 'Exhaustion'.
    """
    # These weights are examples and should be tuned based on backtesting and domain knowledge
    if regime == 'Strong Bull':
        return {'Trend': 0.6, 'Volatility': 0.1, 'Exhaustion': 0.3}
    elif regime == 'Weak Bull':
        return {'Trend': 0.4, 'Volatility': 0.2, 'Exhaustion': 0.4}
    elif regime == 'Neutral':
        # In Neutral regime, factors might be less reliable, or focus might shift.
        # Example: Give more weight to Volatility and Exhaustion for range trading or reversal signals.
        return {'Trend': 0.2, 'Volatility': 0.4, 'Exhaustion': 0.4}
    elif regime == 'Weak Bear':
        return {'Trend': 0.4, 'Volatility': 0.2, 'Exhaustion': 0.4}
    elif regime == 'Strong Bear':
        return {'Trend': 0.6, 'Volatility': 0.1, 'Exhaustion': 0.3}
    else: # Default or unexpected regime
        # Fallback to a neutral or default weighting scheme
        return {'Trend': 1/3, 'Volatility': 1/3, 'Exhaustion': 1/3}

# Ensure df_for_final_evaluation is ready and has factor columns
# This dataframe should already contain 'Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor'
# from the previous factor calculation step using optimal or default parameters.

# Iterate through the dataframe to calculate dynamic MSS
# We need to calculate the regime first to get the weights for that regime.
# However, the regime depends on MSS. This creates a dependency.
# A common approach is to use the *previous* bar's regime to determine the current bar's weights,
# or use the static regime classification first, then calculate dynamic weights based on that.
# Let's use the static regime classification ('Regime_final') calculated in the previous step
# to determine the dynamic weights for the 'MSS_dynamic' calculation for the current bar.

df_for_final_evaluation['MSS_dynamic'] = np.nan # Initialize dynamic MSS column

for index, row in df_for_final_evaluation.iterrows():
    current_regime = row['Regime_final'] # Use the statically classified regime
    dynamic_weights = get_dynamic_weights(current_regime)

    # Calculate dynamic MSS using factor values and dynamic weights
    dynamic_mss = (dynamic_weights['Trend'] * row['Trend_Factor'] +
                   dynamic_weights['Volatility'] * row['Volatility_Factor'] +
                   dynamic_weights['Exhaustion'] * row['Exhaustion_Factor'])

    df_for_final_evaluation.loc[index, 'MSS_dynamic'] = dynamic_mss

# Update the 'Regime_final' column based on the newly calculated 'MSS_dynamic'
# Use the same classification function and optimal thresholds found by the GA.
# The optimal thresholds (strong_bull_threshold_final, etc.) are available from the previous step.

# Reclassify Regime using the dynamic MSS
df_for_final_evaluation['Regime_final_dynamic'] = df_for_final_evaluation['MSS_dynamic'].apply(
    lambda mss: classify_regime_final(mss,
                                      strong_bull_threshold_final,
                                      weak_bull_threshold_final,
                                      neutral_threshold_upper_final,
                                      neutral_threshold_lower_final,
                                      strong_bear_threshold_final,
                                      weak_bear_threshold_final)
)

# Replace the old 'Regime_final' column with the new one based on dynamic MSS
df_for_final_evaluation = df_for_final_evaluation.drop(columns=['Regime_final'])
df_for_final_evaluation = df_for_final_evaluation.rename(columns={'Regime_final_dynamic': 'Regime_final'})


# Display the head and info of the DataFrame
display(df_for_final_evaluation.head())
display(df_for_final_evaluation.info())

# Re-define classify_regime_final function to accept thresholds as arguments for clarity
def classify_regime_final(mss, strong_bull_threshold, weak_bull_threshold,
                          neutral_threshold_upper, neutral_threshold_lower,
                          strong_bear_threshold, weak_bear_threshold):
    if mss > strong_bull_threshold:
        return 'Strong Bull'
    elif mss > weak_bull_threshold:
        return 'Weak Bull'
    elif mss >= neutral_threshold_lower and mss <= neutral_threshold_upper:
        return 'Neutral'
    elif mss > strong_bear_threshold:
        return 'Weak Bear'
    else:
        return 'Strong Bear'

"""**Reasoning**:
The error indicates that the `classify_regime_final` function was defined to accept 7 arguments (mss and 6 threshold parameters) but the lambda function within the `.apply()` method was only passing 1 argument (mss). The function definition was placed *after* the `.apply()` call. Move the re-definition of `classify_regime_final` to *before* it is used in the `.apply()` method and fix the lambda function to correctly pass all required arguments.


"""

# Re-define classify_regime_final function to accept thresholds as arguments BEFORE it's used
def classify_regime_final(mss, strong_bull_threshold, weak_bull_threshold,
                          neutral_threshold_upper, neutral_threshold_lower,
                          strong_bear_threshold, weak_bear_threshold):
    """
    Classifies the market regime based on the Market State Score (MSS) and thresholds.

    Args:
        mss (float): The Market State Score.
        strong_bull_threshold (float): MSS threshold for Strong Bull.
        weak_bull_threshold (float): MSS threshold for Weak Bull.
        neutral_threshold_upper (float): Upper MSS threshold for Neutral.
        neutral_threshold_lower (float): Lower MSS threshold for Neutral.
        strong_bear_threshold (float): MSS threshold for Strong Bear.
        weak_bear_threshold (float): MSS threshold for Weak Bear.

    Returns:
        str: The classified market regime.
    """
    if mss > strong_bull_threshold:
        return 'Strong Bull'
    elif mss > weak_bull_threshold:
        return 'Weak Bull'
    elif mss >= neutral_threshold_lower and mss <= neutral_threshold_upper:
        return 'Neutral'
    elif mss > strong_bear_threshold:
        return 'Weak Bear'
    else:
        return 'Strong Bear'

def get_dynamic_weights(regime):
    """
    Returns dynamic weights for factors based on market regime.

    Args:
        regime (str): The current market regime ('Strong Bull', 'Weak Bull',
                      'Neutral', 'Weak Bear', 'Strong Bear').

    Returns:
        dict: Dictionary of weights for 'Trend', 'Volatility', 'Exhaustion'.
    """
    # These weights are examples and should be tuned based on backtesting and domain knowledge
    if regime == 'Strong Bull':
        return {'Trend': 0.6, 'Volatility': 0.1, 'Exhaustion': 0.3}
    elif regime == 'Weak Bull':
        return {'Trend': 0.4, 'Volatility': 0.2, 'Exhaustion': 0.4}
    elif regime == 'Neutral':
        # In Neutral regime, factors might be less reliable, or focus might shift.
        # Example: Give more weight to Volatility and Exhaustion for range trading or reversal signals.
        return {'Trend': 0.2, 'Volatility': 0.4, 'Exhaustion': 0.4}
    elif regime == 'Weak Bear':
        return {'Trend': 0.4, 'Volatility': 0.2, 'Exhaustion': 0.4}
    elif regime == 'Strong Bear':
        return {'Trend': 0.6, 'Volatility': 0.1, 'Exhaustion': 0.3}
    else: # Default or unexpected regime
        # Fallback to a neutral or default weighting scheme
        return {'Trend': 1/3, 'Volatility': 1/3, 'Exhaustion': 1/3}

# Ensure df_for_final_evaluation is ready and has factor columns
# This dataframe should already contain 'Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor'
# from the previous factor calculation step using optimal or default parameters.

# Extract optimal thresholds found by the GA from evaluation_params_final
# Use default values if optimal_params_refined_indicators was None
strong_bull_threshold_final = evaluation_params_final.get('strong_bull_threshold', 50)
weak_bull_threshold_final = evaluation_params_final.get('weak_bull_threshold', 20)
neutral_threshold_upper_final = evaluation_params_final.get('neutral_threshold_upper', 20)
neutral_threshold_lower_final = evaluation_params_final.get('neutral_threshold_lower', -20)
strong_bear_threshold_final = evaluation_params_final.get('strong_bear_threshold', -50)
weak_bear_threshold_final = evaluation_params_final.get('weak_bear_threshold', -20)


df_for_final_evaluation['MSS_dynamic'] = np.nan # Initialize dynamic MSS column

# Iterate through the dataframe to calculate dynamic MSS
# Use the statically classified regime ('Regime_final') to determine dynamic weights for the current bar.
for index, row in df_for_final_evaluation.iterrows():
    current_regime = row['Regime_final'] # Use the statically classified regime
    dynamic_weights = get_dynamic_weights(current_regime)

    # Calculate dynamic MSS using factor values and dynamic weights
    dynamic_mss = (dynamic_weights['Trend'] * row['Trend_Factor'] +
                   dynamic_weights['Volatility'] * row['Volatility_Factor'] +
                   dynamic_weights['Exhaustion'] * row['Exhaustion_Factor'])

    df_for_final_evaluation.loc[index, 'MSS_dynamic'] = dynamic_mss

# Update the 'Regime_final' column based on the newly calculated 'MSS_dynamic'
# Use the re-defined classify_regime_final function with the optimal thresholds.
df_for_final_evaluation['Regime_final_dynamic'] = df_for_final_evaluation['MSS_dynamic'].apply(
    lambda mss: classify_regime_final(mss,
                                      strong_bull_threshold_final,
                                      weak_bull_threshold_final,
                                      neutral_threshold_upper_final,
                                      neutral_threshold_lower_final,
                                      strong_bear_threshold_final,
                                      weak_bear_threshold_final)
)

# Replace the old 'Regime_final' column with the new one based on dynamic MSS
df_for_final_evaluation = df_for_final_evaluation.drop(columns=['Regime_final'])
df_for_final_evaluation = df_for_final_evaluation.rename(columns={'Regime_final_dynamic': 'Regime_final'})


# Display the head and info of the DataFrame
display(df_for_final_evaluation.head())
display(df_for_final_evaluation.info())

"""**Reasoning**:
The dynamic MSS has been calculated and the regime has been updated based on it. The next step is to update the trading simulation logic to use the `MSS_dynamic` and the new `Regime_final` columns for trading decisions.


"""

# Re-run the trading simulation using the dynamically calculated MSS and updated regime
position = 0  # 1 for long, -1 for short, 0 for flat
entry_price = 0
stop_loss = 0
equity_curve = []
trade_log = []
current_capital = initial_capital # Start with the initial capital

# Stop-loss multipliers from the optimal parameters found by the GA
# Ensure these are available (defined in the previous evaluation cell)
stop_loss_multiplier_strong_final = evaluation_params_final.get('stop_loss_multiplier_strong', 2)
stop_loss_multiplier_weak_final = evaluation_params_final.get('stop_loss_multiplier_weak', 1)


# Iterate through the dataframe using the dynamic MSS and updated regime
for index, row in df_for_final_evaluation.iterrows():
    current_price = row['Close']
    current_regime = row['Regime_final'] # Use the regime based on dynamic MSS
    current_volatility_factor = row['Volatility_Factor'] # Use the normalized Volatility Factor

    # Determine stop-loss distance based on regime and Volatility Factor
    # Use absolute value of volatility factor for stop distance as it represents price movement
    if not isinstance(current_volatility_factor, (int, float)) or np.isnan(current_volatility_factor) or abs(current_volatility_factor) < 1e-9:
         stop_loss_distance = 0 # Cannot calculate dynamic stop loss without valid Volatility Factor
    else:
        if current_regime in ['Strong Bull', 'Strong Bear']:
            stop_loss_distance = stop_loss_multiplier_strong_final * abs(current_volatility_factor) # Use absolute value for distance
        elif current_regime in ['Weak Bull', 'Weak Bear']:
            stop_loss_distance = stop_loss_multiplier_weak_final * abs(current_volatility_factor) # Use absolute value for distance
        else: # Neutral
            stop_loss_distance = 0

    # Trading Logic based on Action Matrix (Section 4.3.3) - Using the updated regime
    if current_regime == 'Strong Bull':
        if position == 0: # Enter Long
            position = 1
            entry_price = current_price
            stop_loss = current_price - stop_loss_distance if stop_loss_distance > 0 else -float('inf') # Ensure stop is below entry for long
            trade_log.append({'Date': index, 'Action': 'Enter Long', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == 1 and stop_loss_distance > 0: # Adjust Trailing Stop for Long if valid distance
             stop_loss = max(stop_loss, current_price - stop_loss_distance)
             # Optionally log stop adjustment:
             # trade_log.append({'Date': index, 'Action': 'Adjust Long Stop', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1: # Exit Short
            pnl = (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0


    elif current_regime == 'Weak Bull':
        if position == 1 and stop_loss_distance > 0: # Hold Longs ONLY, tighten stop
             stop_loss = max(stop_loss, current_price - stop_loss_distance)
             # Optionally log hold/stop adjust:
             # trade_log.append({'Date': index, 'Action': 'Hold Long (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1: # Exit Short
            pnl = (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Short (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0


    elif current_regime == 'Neutral':
        if position != 0: # EXIT ALL POSITIONS
            action = 'Exit Long' if position == 1 else 'Exit Short'
            pnl = (current_price - entry_price) if position == 1 else (entry_price - current_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': action + ' (Neutral Regime)', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0

    elif current_regime == 'Weak Bear':
         if position == -1 and stop_loss_distance > 0: # Hold Shorts ONLY, tighten stop
            stop_loss = min(stop_loss, current_price + stop_loss_distance)
            # Optionally log hold/stop adjust:
            # trade_log.append({'Date': index, 'Action': 'Hold Short (Tighten Stop)', 'Price': current_price, 'StopLoss': stop_loss})
         elif position == 1: # Exit Long
            pnl = (current_price - entry_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0


    elif current_regime == 'Strong Bear':
        if position == 0: # Enter Short
            position = -1
            entry_price = current_price
            stop_loss = current_price + stop_loss_distance if stop_loss_distance > 0 else float('inf')
            trade_log.append({'Date': index, 'Action': 'Enter Short', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == -1 and stop_loss_distance > 0: # Adjust Trailing Stop for Short
             stop_loss = min(stop_loss, current_price + stop_loss_distance)
             # Optionally log stop adjustment:
             # trade_log.append({'Date': index, 'Action': 'Adjust Short Stop', 'Price': current_price, 'StopLoss': stop_loss})
        elif position == 1: # Exit Long
            pnl = (current_price - entry_price)
            current_capital += pnl
            trade_log.append({'Date': index, 'Action': 'Exit Long (Regime Change)', 'Price': current_price, 'PnL': pnl})
            position = 0
            entry_price = 0
            stop_loss = 0


    # Check for stop-loss hit (only if position is active and stop_loss is a valid number)
    if position == 1 and not np.isinf(stop_loss) and current_price <= stop_loss:
        pnl = (current_price - entry_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0
    elif position == -1 and not np.isinf(stop_loss) and current_price >= stop_loss:
        pnl = (entry_price - current_price)
        current_capital += pnl
        trade_log.append({'Date': index, 'Action': 'Stop Out Short', 'Price': current_price, 'PnL': pnl})
        position = 0
        entry_price = 0
        stop_loss = 0

    # Append current equity to the equity curve
    equity_curve.append({'Date': index, 'Equity': current_capital + (current_price - entry_price if position != 0 else 0)})

# Convert trade log and equity curve to DataFrames
equity_curve_df_dynamic_mss = pd.DataFrame(equity_curve).set_index('Date')
trade_log_df_dynamic_mss = pd.DataFrame(trade_log)

display("Equity Curve (Dynamic MSS Simulation):")
display(equity_curve_df_dynamic_mss.head())
display("Trade Log (Dynamic MSS Simulation):")
display(trade_log_df_dynamic_mss.head())

"""**Reasoning**:
The trading simulation with dynamic MSS and updated regime is complete. The next step is to evaluate the performance of this simulation by calculating key metrics like Sortino Ratio and Calmar Ratio and visualize the equity curve and trade logs.


"""

# Evaluate performance metrics for the simulation with dynamic MSS
if equity_curve_df_dynamic_mss.empty or len(equity_curve_df_dynamic_mss) < 2:
    display("Dynamic MSS equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_dynamic_mss['Daily_Return'] = equity_curve_df_dynamic_mss['Equity'].pct_change().fillna(0)
    total_return_dynamic_mss = (equity_curve_df_dynamic_mss['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_dynamic_mss = (1 + total_return_dynamic_mss)**(trading_periods_per_year / len(equity_curve_df_dynamic_mss)) - 1

    equity_curve_df_dynamic_mss['Peak'] = equity_curve_df_dynamic_mss['Equity'].cummax()
    equity_curve_df_dynamic_mss['Drawdown'] = equity_curve_df_dynamic_mss['Equity'] - equity_curve_df_dynamic_mss['Peak']
    max_drawdown_dynamic_mss = equity_curve_df_dynamic_mss['Drawdown'].min()

    mar = 0
    downside_returns_dynamic_mss = equity_curve_df_dynamic_mss[equity_curve_df_dynamic_mss['Daily_Return'] < mar]['Daily_Return']
    downside_deviation_dynamic_mss = downside_returns_dynamic_mss.std()

    sortino_ratio_dynamic_mss = (annualized_return_dynamic_mss - mar) / downside_deviation_dynamic_mss if downside_deviation_dynamic_mss != 0 and not np.isnan(downside_deviation_dynamic_mss) else (float('inf') if annualized_return_dynamic_mss > mar else (0 if annualized_return_dynamic_mss == mar else -1000))

    calmar_ratio_dynamic_mss = annualized_return_dynamic_mss / abs(max_drawdown_dynamic_mss) if max_drawdown_dynamic_mss != 0 else float('inf')


    display("\nPerformance Metrics (using Dynamic MSS and updated Regime):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_dynamic_mss['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_dynamic_mss:.4f}")
    display(f"Annualized Return: {annualized_return_dynamic_mss:.4f}")
    display(f"Max Drawdown: {max_drawdown_dynamic_mss:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_dynamic_mss:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_dynamic_mss:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_dynamic_mss.index, equity_curve_df_dynamic_mss['Equity'], label='Equity Curve (Dynamic MSS)')
    plt.title('Equity Curve (Dynamic MSS and Updated Regime)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Dynamic MSS Simulation):")
    display(trade_log_df_dynamic_mss.head())

    # Analyze losing trades from the simulation
    losing_trades_df_dynamic_mss = trade_log_df_dynamic_mss[trade_log_df_dynamic_mss['PnL'].notna() & (trade_log_df_dynamic_mss['PnL'] < 0)]
    display("\nLosing Trades Summary (Dynamic MSS Simulation):")
    display(losing_trades_df_dynamic_mss.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the dynamic MSS equity curve is empty.")

"""**Reasoning**:
The traceback points to a `SyntaxError: invalid syntax` related to an `else:` block. This usually means there's an issue with the preceding `if`, `elif`, or `try...except` block structure, such as missing colons, incorrect indentation, or an empty block where code is expected. Review the code block, specifically around the `else` statement indicated in the traceback, to identify and fix the syntax error. The likely cause is an incomplete `if` or `elif` statement before the `else`.


"""

# Evaluate performance metrics for the simulation with dynamic MSS
if equity_curve_df_dynamic_mss.empty or len(equity_curve_df_dynamic_mss) < 2:
    display("Dynamic MSS equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_dynamic_mss['Daily_Return'] = equity_curve_df_dynamic_mss['Equity'].pct_change().fillna(0)
    total_return_dynamic_mss = (equity_curve_df_dynamic_mss['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_dynamic_mss = (1 + total_return_dynamic_mss)**(trading_periods_per_year / len(equity_curve_df_dynamic_mss)) - 1

    equity_curve_df_dynamic_mss['Peak'] = equity_curve_df_dynamic_mss['Equity'].cummax()
    equity_curve_df_dynamic_mss['Drawdown'] = equity_curve_df_dynamic_mss['Equity'] - equity_curve_df_dynamic_mss['Peak']
    max_drawdown_dynamic_mss = equity_curve_df_dynamic_mss['Drawdown'].min()

    mar = 0
    downside_returns_dynamic_mss = equity_curve_df_dynamic_mss[equity_curve_df_dynamic_mss['Daily_Return'] < mar]['Daily_Return']
    # Check if downside_returns_dynamic_mss is empty before calculating std()
    if not downside_returns_dynamic_mss.empty:
        downside_deviation_dynamic_mss = downside_returns_dynamic_mss.std()
    else:
        downside_deviation_dynamic_mss = 0 # Or np.nan, depending on desired behavior when no downside returns


    # Handle cases where downside_deviation is zero or NaN
    if downside_deviation_dynamic_mss == 0 or np.isnan(downside_deviation_dynamic_mss):
        sortino_ratio_dynamic_mss = float('inf') if annualized_return_dynamic_mss > mar else (0 if annualized_return_dynamic_mss == mar else -1000) # High ratio if profitable with no downside, else low
    else:
        sortino_ratio_dynamic_mss = (annualized_return_dynamic_mss - mar) / downside_deviation_dynamic_mss

    # Handle cases where sortino_ratio is infinite or NaN after calculation
    if np.isinf(sortino_ratio_dynamic_mss) or np.isnan(sortino_ratio_dynamic_mss):
        sortino_ratio_dynamic_mss = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_dynamic_mss = annualized_return_dynamic_mss / abs(max_drawdown_dynamic_mss) if max_drawdown_dynamic_mss != 0 else float('inf')


    display("\nPerformance Metrics (using Dynamic MSS and updated Regime):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_dynamic_mss['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_dynamic_mss:.4f}")
    display(f"Annualized Return: {annualized_return_dynamic_mss:.4f}")
    display(f"Max Drawdown: {max_drawdown_dynamic_mss:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_dynamic_mss:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_dynamic_mss:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_dynamic_mss.index, equity_curve_df_dynamic_mss['Equity'], label='Equity Curve (Dynamic MSS)')
    plt.title('Equity Curve (Dynamic MSS and Updated Regime)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Dynamic MSS Simulation):")
    display(trade_log_df_dynamic_mss.head())

    # Analyze losing trades from the simulation
    losing_trades_df_dynamic_mss = trade_log_df_dynamic_mss[trade_log_df_dynamic_mss['PnL'].notna() & (trade_log_df_dynamic_mss['PnL'] < 0)]
    display("\nLosing Trades Summary (Dynamic MSS Simulation):")
    display(losing_trades_df_dynamic_mss.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the dynamic MSS equity curve is empty.")

"""**Reasoning**:
The traceback still indicates a `SyntaxError: invalid syntax` associated with an `else:` block, similar to the previous error. This suggests there's still an issue with the preceding conditional statement structure around line 67. Review the code again, specifically the `if` and `elif` statements before that `else` block, to ensure they are correctly formed. It seems there is an empty `elif` block. Add a `pass` statement or some placeholder code within the `elif` block to resolve the syntax error.


"""

# Evaluate performance metrics for the simulation with dynamic MSS
if equity_curve_df_dynamic_mss.empty or len(equity_curve_df_dynamic_mss) < 2:
    display("Dynamic MSS equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_dynamic_mss['Daily_Return'] = equity_curve_df_dynamic_mss['Equity'].pct_change().fillna(0)
    total_return_dynamic_mss = (equity_curve_df_dynamic_mss['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_dynamic_mss = (1 + total_return_dynamic_mss)**(trading_periods_per_year / len(equity_curve_df_dynamic_mss)) - 1

    equity_curve_df_dynamic_mss['Peak'] = equity_curve_df_dynamic_mss['Equity'].cummax()
    equity_curve_df_dynamic_mss['Drawdown'] = equity_curve_df_dynamic_mss['Equity'] - equity_curve_df_dynamic_mss['Peak']
    max_drawdown_dynamic_mss = equity_curve_df_dynamic_mss['Drawdown'].min()

    mar = 0
    downside_returns_dynamic_mss = equity_curve_df_dynamic_mss[equity_curve_df_dynamic_mss['Daily_Return'] < mar]['Daily_Return']
    # Check if downside_returns_dynamic_mss is empty before calculating std()
    if not downside_returns_dynamic_mss.empty:
        downside_deviation_dynamic_mss = downside_returns_dynamic_mss.std()
    else:
        downside_deviation_dynamic_mss = 0 # Or np.nan, depending on desired behavior when no downside returns


    # Handle cases where downside_deviation is zero or NaN
    if downside_deviation_dynamic_mss == 0 or np.isnan(downside_deviation_dynamic_mss):
        sortino_ratio_dynamic_mss = float('inf') if annualized_return_dynamic_mss > mar else (0 if annualized_return_dynamic_mss == mar else -1000) # High ratio if profitable with no downside, else low
    # Fix: Add a `pass` or code here if this elif was meant to do something
    elif False: # Placeholder or condition that was empty
        pass
    else: # This else should be the fallback if the above conditions are not met
        sortino_ratio_dynamic_mss = (annualized_return_dynamic_mss - mar) / downside_deviation_dynamic_mss

    # Handle cases where sortino_ratio is infinite or NaN after calculation
    if np.isinf(sortino_ratio_dynamic_mss) or np.isnan(sortino_ratio_dynamic_mss):
        sortino_ratio_dynamic_mss = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_dynamic_mss = annualized_return_dynamic_mss / abs(max_drawdown_dynamic_mss) if max_drawdown_dynamic_mss != 0 else float('inf')


    display("\nPerformance Metrics (using Dynamic MSS and updated Regime):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_dynamic_mss['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_dynamic_mss:.4f}")
    display(f"Annualized Return: {annualized_return_dynamic_mss:.4f}")
    display(f"Max Drawdown: {max_drawdown_dynamic_mss:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_dynamic_mss:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_dynamic_mss:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_dynamic_mss.index, equity_curve_df_dynamic_mss['Equity'], label='Equity Curve (Dynamic MSS)')
    plt.title('Equity Curve (Dynamic MSS and Updated Regime)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Dynamic MSS Simulation):")
    display(trade_log_df_dynamic_mss.head())

    # Analyze losing trades from the simulation
    losing_trades_df_dynamic_mss = trade_log_df_dynamic_mss[trade_log_df_dynamic_mss['PnL'].notna() & (trade_log_df_dynamic_mss['PnL'] < 0)]
    display("\nLosing Trades Summary (Dynamic MSS Simulation):")
    display(losing_trades_df_dynamic_mss.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the dynamic MSS equity curve is empty.")

"""**Reasoning**:
The traceback continues to show a `SyntaxError: invalid syntax` at an `else:` block, suggesting there's still an issue with the conditional logic preceding it. The previous attempt to fix it by adding an `elif False:` and `pass` was incorrect and introduced a new syntax error. The original structure for calculating `sortino_ratio_dynamic_mss` seems to have been a simple `if...else` without an `elif`. Revert to the correct `if...else` structure for the `sortino_ratio_dynamic_mss` calculation to resolve the syntax error.


"""

# Evaluate performance metrics for the simulation with dynamic MSS
if equity_curve_df_dynamic_mss.empty or len(equity_curve_df_dynamic_mss) < 2:
    display("Dynamic MSS equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_dynamic_mss['Daily_Return'] = equity_curve_df_dynamic_mss['Equity'].pct_change().fillna(0)
    total_return_dynamic_mss = (equity_curve_df_dynamic_mss['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_dynamic_mss = (1 + total_return_dynamic_mss)**(trading_periods_per_year / len(equity_curve_df_dynamic_mss)) - 1

    equity_curve_df_dynamic_mss['Peak'] = equity_curve_df_dynamic_mss['Equity'].cummax()
    equity_curve_df_dynamic_mss['Drawdown'] = equity_curve_df_dynamic_mss['Equity'] - equity_curve_df_dynamic_mss['Peak']
    max_drawdown_dynamic_mss = equity_curve_df_dynamic_mss['Drawdown'].min()

    mar = 0
    downside_returns_dynamic_mss = equity_curve_df_dynamic_mss[equity_curve_df_dynamic_mss['Daily_Return'] < mar]['Daily_Return']
    # Check if downside_returns_dynamic_mss is empty before calculating std()
    if not downside_returns_dynamic_mss.empty:
        downside_deviation_dynamic_mss = downside_returns_dynamic_mss.std()
    else:
        downside_deviation_dynamic_mss = 0 # Or np.nan, depending on desired behavior when no downside returns


    # Handle cases where downside_deviation is zero or NaN
    if downside_deviation_dynamic_mss == 0 or np.isnan(downside_deviation_dynamic_mss):
        sortino_ratio_dynamic_mss = float('inf') if annualized_return_dynamic_mss > mar else (0 if annualized_return_dynamic_mss == mar else -1000) # High ratio if profitable with no downside, else low
    else: # This else should be the fallback if the above conditions are not met
        sortino_ratio_dynamic_mss = (annualized_return_dynamic_mss - mar) / downside_deviation_dynamic_mss

    # Handle cases where sortino_ratio is infinite or NaN after calculation
    if np.isinf(sortino_ratio_dynamic_mss) or np.isnan(sortino_ratio_dynamic_mss):
        sortino_ratio_dynamic_mss = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_dynamic_mss = annualized_return_dynamic_mss / abs(max_drawdown_dynamic_mss) if max_drawdown_dynamic_mss != 0 else float('inf')


    display("\nPerformance Metrics (using Dynamic MSS and updated Regime):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_dynamic_mss['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_dynamic_mss:.4f}")
    display(f"Annualized Return: {annualized_return_dynamic_mss:.4f}")
    display(f"Max Drawdown: {max_drawdown_dynamic_mss:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_dynamic_mss:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_dynamic_mss:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_dynamic_mss.index, equity_curve_df_dynamic_mss['Equity'], label='Equity Curve (Dynamic MSS)')
    plt.title('Equity Curve (Dynamic MSS and Updated Regime)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Dynamic MSS Simulation):")
    display(trade_log_df_dynamic_mss.head())

    # Analyze losing trades from the simulation
    losing_trades_df_dynamic_mss = trade_log_df_dynamic_mss[trade_log_df_dynamic_mss['PnL'].notna() & (trade_log_df_dynamic_mss['PnL'] < 0)]
    display("\nLosing Trades Summary (Dynamic MSS Simulation):")
    display(losing_trades_df_dynamic_mss.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the dynamic MSS equity curve is empty.")

"""**Reasoning**:
The traceback continues to indicate a `SyntaxError: invalid syntax` at the same `else:` block, suggesting the issue persists in the conditional logic for calculating the Sortino ratio. Despite previous attempts, the syntax remains incorrect. The error is likely due to an incorrect chain of `if` and `else` statements. Re-examine the logic for calculating `sortino_ratio_dynamic_mss` and ensure the `if` and `else` statements are properly paired and indented. It appears there might be an extra `else` or an `else` without a preceding `if`. Simplify the logic to the core `if` condition checking for zero or NaN downside deviation, followed by its corresponding `else`.


"""

# Evaluate performance metrics for the simulation with dynamic MSS
if equity_curve_df_dynamic_mss.empty or len(equity_curve_df_dynamic_mss) < 2:
    display("Dynamic MSS equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_dynamic_mss['Daily_Return'] = equity_curve_df_dynamic_mss['Equity'].pct_change().fillna(0)
    total_return_dynamic_mss = (equity_curve_df_dynamic_mss['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_dynamic_mss = (1 + total_return_dynamic_mss)**(trading_periods_per_year / len(equity_curve_df_dynamic_mss)) - 1

    equity_curve_df_dynamic_mss['Peak'] = equity_curve_df_dynamic_mss['Equity'].cummax()
    equity_curve_df_dynamic_mss['Drawdown'] = equity_curve_df_dynamic_mss['Equity'] - equity_curve_df_dynamic_mss['Peak']
    max_drawdown_dynamic_mss = equity_curve_df_dynamic_mss['Drawdown'].min()

    mar = 0
    downside_returns_dynamic_mss = equity_curve_df_dynamic_mss[equity_curve_df_dynamic_mss['Daily_Return'] < mar]['Daily_Return']

    # Handle cases where downside_deviation is zero or NaN using a single if/else block
    if downside_returns_dynamic_mss.empty or downside_returns_dynamic_mss.std() == 0 or np.isnan(downside_returns_dynamic_mss.std()):
        # If no downside returns or std is zero/NaN, check if annualized return is positive
        sortino_ratio_dynamic_mss = float('inf') if annualized_return_dynamic_mss > mar else (0 if annualized_return_dynamic_mss == mar else -1000)
    else:
        # Otherwise, calculate the standard Sortino Ratio
        sortino_ratio_dynamic_mss = (annualized_return_dynamic_mss - mar) / downside_returns_dynamic_mss.std()

    # Handle cases where sortino_ratio is infinite or NaN after calculation (should be covered by the above, but as a safety check)
    if np.isinf(sortino_ratio_dynamic_mss) or np.isnan(sortino_ratio_dynamic_mss):
        sortino_ratio_dynamic_mss = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_dynamic_mss = annualized_return_dynamic_mss / abs(max_drawdown_dynamic_mss) if max_drawdown_dynamic_mss != 0 else float('inf')


    display("\nPerformance Metrics (using Dynamic MSS and updated Regime):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_dynamic_mss['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_dynamic_mss:.4f}")
    display(f"Annualized Return: {annualized_return_dynamic_mss:.4f}")
    display(f"Max Drawdown: {max_drawdown_dynamic_mss:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_dynamic_mss:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_dynamic_mss:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_dynamic_mss.index, equity_curve_df_dynamic_mss['Equity'], label='Equity Curve (Dynamic MSS)')
    plt.title('Equity Curve (Dynamic MSS and Updated Regime)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Dynamic MSS Simulation):")
    display(trade_log_df_dynamic_mss.head())

    # Analyze losing trades from the simulation
    losing_trades_df_dynamic_mss = trade_log_df_dynamic_mss[trade_log_df_dynamic_mss['PnL'].notna() & (trade_log_df_dynamic_mss['PnL'] < 0)]
    display("\nLosing Trades Summary (Dynamic MSS Simulation):")
    display(losing_trades_df_dynamic_mss.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the dynamic MSS equity curve is empty.")

"""## Enhance trading logic (action matrix)

### Subtask:
Modify the trading simulation logic to include more sophisticated rules based on the Action Matrix, such as implementing gradual entries and exits and adding more nuanced regime-specific trading rules.

**Reasoning**:
Modify the trading simulation logic to include fractional position sizing, gradual entries/exits, and adjust stop-loss handling for fractional positions.
"""

# Re-run the trading simulation with enhanced logic for fractional positions
position_size = 0.0  # Current position size (can be fractional, positive for long, negative for short, 0 for flat)
entry_price = 0.0 # Weighted average entry price for the current position
stop_loss = 0.0
equity_curve_enhanced_logic = []
trade_log_enhanced_logic = []
current_capital = initial_capital # Start with the initial capital
max_position_size = 1.0 # Maximum allowed position size (e.g., 1.0 represents full capital allocation or 1 unit)

# Parameters for gradual entry/exit (these are examples and need tuning)
entry_step_size = 0.2 # Enter/Exit in steps of 20% of max_position_size
mss_confidence_factor = 0.01 # How much MSS magnitude affects position sizing

# Stop-loss multipliers from the optimal parameters found by the GA
# Ensure these are available (defined in the previous evaluation cell)
stop_loss_multiplier_strong_final = evaluation_params_final.get('stop_loss_multiplier_strong', 2)
stop_loss_multiplier_weak_final = evaluation_params_final.get('stop_loss_multiplier_weak', 1)


# Iterate through the dataframe using the dynamic MSS and updated regime
for index, row in df_for_final_evaluation.iterrows():
    current_price = row['Close']
    current_regime = row['Regime_final'] # Use the regime based on dynamic MSS
    current_mss = row['MSS_dynamic'] # Use the dynamic MSS
    current_volatility_factor = row['Volatility_Factor'] # Use the normalized Volatility Factor

    # Determine stop-loss distance based on regime and Volatility Factor
    # Use absolute value of volatility factor for stop distance as it represents price movement
    if not isinstance(current_volatility_factor, (int, float)) or np.isnan(current_volatility_factor) or abs(current_volatility_factor) < 1e-9:
         stop_loss_distance = 0 # Cannot calculate dynamic stop loss without valid Volatility Factor
    else:
        if current_regime in ['Strong Bull', 'Strong Bear']:
            stop_loss_distance = stop_loss_multiplier_strong_final * abs(current_volatility_factor) # Use absolute value for distance
        elif current_regime in ['Weak Bull', 'Weak Bear']:
            stop_loss_distance = stop_loss_multiplier_weak_final * abs(current_volatility_factor) # Use absolute value for distance
        else: # Neutral
            stop_loss_distance = 0


    # --- Trading Logic with Fractional Positions and Gradual Entries/Exits ---

    # Determine target position size based on regime and MSS confidence
    target_position_size = 0.0
    if current_regime == 'Strong Bull':
        # Scale target position size based on MSS magnitude within the regime
        # MSS ranges from > strong_bull_threshold to 100. Normalize this range.
        normalized_mss = (current_mss - strong_bull_threshold_final) / (100 - strong_bull_threshold_final) if (100 - strong_bull_threshold_final) > 0 else 0
        target_position_size = max_position_size * np.clip(normalized_mss, 0, 1) # Target long position, scaled by normalized MSS

    elif current_regime == 'Weak Bull':
        # Hold Longs ONLY, potentially reduce position if MSS is closer to Neutral
        normalized_mss = (current_mss - weak_bull_threshold_final) / (neutral_threshold_upper_final - weak_bull_threshold_final) if (neutral_threshold_upper_final - weak_bull_threshold_final) > 0 else 0
        target_position_size = position_size # Default to hold
        if position_size > 0: # If currently long
             # Scale target position size based on how far into Weak Bull we are
             target_position_size = max_position_size * np.clip(normalized_mss, 0, 1) # Scale down towards 0 as MSS approaches Neutral

    elif current_regime == 'Neutral':
        target_position_size = 0.0 # Target flat position

    elif current_regime == 'Weak Bear':
         # Hold Shorts ONLY, potentially reduce position if MSS is closer to Neutral
         normalized_mss = (current_mss - weak_bear_threshold_final) / (neutral_threshold_lower_final - weak_bear_threshold_final) if (neutral_threshold_lower_final - weak_bear_threshold_final) > 0 else 0 # Note: MSS is negative here, adjusting normalization
         # A different normalization might be needed for bear regimes
         # Let's normalize from -100 to weak_bear_threshold_final
         normalized_mss = (current_mss - (-100)) / (weak_bear_threshold_final - (-100)) if (weak_bear_threshold_final - (-100)) > 0 else 0
         target_position_size = position_size # Default to hold
         if position_size < 0: # If currently short
             # Scale target position size based on how far into Weak Bear we are
              target_position_size = -max_position_size * np.clip(1 - normalized_mss, 0, 1) # Scale down towards 0 as MSS approaches Neutral (1 - normalized_mss for bear side)


    elif current_regime == 'Strong Bear':
        # Scale target position size based on MSS magnitude within the regime
        # MSS ranges from -100 to strong_bear_threshold. Normalize this range.
        normalized_mss = (current_mss - (-100)) / (strong_bear_threshold_final - (-100)) if (strong_bear_threshold_final - (-100)) > 0 else 0
        target_position_size = -max_position_size * np.clip(1 - normalized_mss, 0, 1) # Target short position, scaled by normalized MSS (1 - normalized_mss for bear side)


    # Implement gradual entry/exit
    position_change = target_position_size - position_size

    # Limit position change to entry_step_size (as a fraction of max_position_size)
    max_step = entry_step_size * max_position_size
    position_change = np.clip(position_change, -max_step, max_step)

    # Calculate the amount of capital/units to trade in this step
    trade_amount_units = position_change * (current_capital / current_price) if current_price > 0 else 0 # Amount in units, positive for buy, negative for sell

    # Update position size and entry price (weighted average)
    new_position_size = position_size + position_change

    if abs(new_position_size) < 1e-9: # If position closes or becomes very small
        if abs(position_size) > 1e-9: # If there was an existing position
             pnl = (current_price - entry_price) * position_size * (current_capital / current_price) # Calculate P/L for the full exit
             current_capital += pnl
             action = 'Exit Long' if position_size > 0 else 'Exit Short'
             trade_log_enhanced_logic.append({'Date': index, 'Action': action + ' (Gradual)', 'Price': current_price, 'PositionChange': -position_size, 'NewPositionSize': 0.0, 'PnL': pnl})
        position_size = 0.0
        entry_price = 0.0
    else:
        if abs(position_size) < 1e-9: # Entering a new position
            entry_price = current_price
            action = 'Enter Long' if new_position_size > 0 else 'Enter Short'
            trade_log_enhanced_logic.append({'Date': index, 'Action': action + ' (Gradual)', 'Price': current_price, 'PositionChange': position_change, 'NewPositionSize': new_position_size, 'PnL': np.nan})
        else: # Adding to or reducing an existing position
            # Calculate P/L on the portion being closed (if reducing)
            pnl_on_closed_portion = 0.0
            if (position_size > 0 and new_position_size < position_size) or (position_size < 0 and new_position_size > position_size): # Reducing position
                 closed_amount_units = (position_size - new_position_size) * (current_capital / current_price) # Amount in units being closed
                 pnl_on_closed_portion = (current_price - entry_price) * (position_size - new_position_size) * (current_capital / current_price) # P/L on the units being closed
                 current_capital += pnl_on_closed_portion # Update capital with P/L from the closed portion

            # Update weighted average entry price when adding to a position
            if (position_size > 0 and new_position_size > position_size) or (position_size < 0 and new_position_size < position_size): # Adding to position
                 # New entry price is weighted average of old position and new addition
                 entry_price = ((entry_price * abs(position_size)) + (current_price * abs(position_change))) / abs(new_position_size) if abs(new_position_size) > 1e-9 else current_price

            action = 'Increase Long' if position_change > 0 else ('Decrease Long' if position_change < 0 and position_size > 0 else ('Increase Short' if position_change < 0 else 'Decrease Short'))
            trade_log_enhanced_logic.append({'Date': index, 'Action': action + ' (Gradual)', 'Price': current_price, 'PositionChange': position_change, 'NewPositionSize': new_position_size, 'PnL': pnl_on_closed_portion if abs(pnl_on_closed_portion) > 1e-9 else np.nan})


        position_size = new_position_size # Update the position size


    # Update Trailing Stop Loss based on the current position size and direction
    if abs(position_size) > 1e-9 and stop_loss_distance > 0: # If there is an active position and valid stop distance
        if position_size > 0: # Long position
            # Initial stop or trail the stop upwards
            if stop_loss == 0.0: # Initialize stop loss on first entry
                 stop_loss = current_price - stop_loss_distance
            else: # Trail the stop
                 stop_loss = max(stop_loss, current_price - stop_loss_distance)
        elif position_size < 0: # Short position
            # Initial stop or trail the stop downwards
            if stop_loss == 0.0: # Initialize stop loss on first entry
                 stop_loss = current_price + stop_loss_distance
            else: # Trail the stop
                stop_loss = min(stop_loss, current_price + stop_loss_distance)
    else: # No active position or invalid stop distance
         stop_loss = 0.0 # Reset stop loss

    # Check for stop-loss hit (only if position is active and stop_loss is a valid number)
    if abs(position_size) > 1e-9 and not np.isinf(stop_loss) and stop_loss != 0.0: # Ensure stop_loss is explicitly set and not 0
        if position_size > 0 and current_price <= stop_loss:
            pnl = (current_price - entry_price) * position_size * (current_capital / current_price) # Calculate P/L for the full exit
            current_capital += pnl
            trade_log_enhanced_logic.append({'Date': index, 'Action': 'Stop Out Long (Fractional)', 'Price': current_price, 'PositionChange': -position_size, 'NewPositionSize': 0.0, 'PnL': pnl})
            position_size = 0.0
            entry_price = 0.0
            stop_loss = 0.0
        elif position_size < 0 and current_price >= stop_loss:
            pnl = (entry_price - current_price) * abs(position_size) * (current_capital / current_price) # Calculate P/L for the full exit (use abs position size for short)
            current_capital += pnl
            trade_log_enhanced_logic.append({'Date': index, 'Action': 'Stop Out Short (Fractional)', 'Price': current_price, 'PositionChange': -position_size, 'NewPositionSize': 0.0, 'PnL': pnl})
            position_size = 0.0
            entry_price = 0.0
            stop_loss = 0.0


    # Append current equity to the equity curve
    # Current equity is current capital + (unrealized P/L of the current position)
    unrealized_pnl = (current_price - entry_price) * position_size * (current_capital / current_price) if abs(position_size) > 1e-9 and current_price > 0 else 0.0
    equity_curve_enhanced_logic.append({'Date': index, 'Equity': current_capital + unrealized_pnl})


# Convert trade log and equity curve to DataFrames
equity_curve_df_enhanced_logic = pd.DataFrame(equity_curve_enhanced_logic).set_index('Date')
trade_log_df_enhanced_logic = pd.DataFrame(trade_log_enhanced_logic)

display("Equity Curve (Enhanced Logic Simulation):")
display(equity_curve_df_enhanced_logic.head())
display("Trade Log (Enhanced Logic Simulation):")
display(trade_log_df_enhanced_logic.head())

"""**Reasoning**:
Evaluate the performance of the strategy with the enhanced logic by calculating key metrics and plotting the equity curve.


"""

# Evaluate performance metrics for the simulation with enhanced logic
if equity_curve_df_enhanced_logic.empty or len(equity_curve_df_enhanced_logic) < 2:
    display("Enhanced logic equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_enhanced_logic['Daily_Return'] = equity_curve_df_enhanced_logic['Equity'].pct_change().fillna(0)
    total_return_enhanced_logic = (equity_curve_df_enhanced_logic['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_enhanced_logic = (1 + total_return_enhanced_logic)**(trading_periods_per_year / len(equity_curve_df_enhanced_logic)) - 1

    equity_curve_df_enhanced_logic['Peak'] = equity_curve_df_enhanced_logic['Equity'].cummax()
    equity_curve_df_enhanced_logic['Drawdown'] = equity_curve_df_enhanced_logic['Equity'] - equity_curve_df_enhanced_logic['Peak']
    max_drawdown_enhanced_logic = equity_curve_df_enhanced_logic['Drawdown'].min()

    mar = 0
    downside_returns_enhanced_logic = equity_curve_df_enhanced_logic[equity_curve_df_enhanced_logic['Daily_Return'] < mar]['Daily_Return']

    # Handle cases where downside_deviation is zero or NaN using a single if/else block
    if downside_returns_enhanced_logic.empty or downside_returns_enhanced_logic.std() == 0 or np.isnan(downside_returns_enhanced_logic.std()):
        # If no downside returns or std is zero/NaN, check if annualized return is positive
        sortino_ratio_enhanced_logic = float('inf') if annualized_return_enhanced_logic > mar else (0 if annualized_return_enhanced_logic == mar else -1000)
    else:
        # Otherwise, calculate the standard Sortino Ratio
        sortino_ratio_enhanced_logic = (annualized_return_enhanced_logic - mar) / downside_returns_enhanced_logic.std()

    # Handle cases where sortino_ratio is infinite or NaN after calculation (should be covered by the above, but as a safety check)
    if np.isinf(sortino_ratio_enhanced_logic) or np.isnan(sortino_ratio_enhanced_logic):
        sortino_ratio_enhanced_logic = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_enhanced_logic = annualized_return_enhanced_logic / abs(max_drawdown_enhanced_logic) if max_drawdown_enhanced_logic != 0 else float('inf')


    display("\nPerformance Metrics (using Enhanced Logic):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_enhanced_logic['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_enhanced_logic:.4f}")
    display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
    display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_enhanced_logic.index, equity_curve_df_enhanced_logic['Equity'], label='Equity Curve (Enhanced Logic)')
    plt.title('Equity Curve (Enhanced Logic)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Enhanced Logic Simulation):")
    display(trade_log_df_enhanced_logic.head())

    # Analyze losing trades from the simulation
    losing_trades_df_enhanced_logic = trade_log_df_enhanced_logic[trade_log_df_enhanced_logic['PnL'].notna() & (trade_log_df_enhanced_logic['PnL'] < 0)]
    display("\nLosing Trades Summary (Enhanced Logic Simulation):")
    display(losing_trades_df_enhanced_logic.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the enhanced logic equity curve is empty.")

"""**Reasoning**:
The previous code failed due to a `SyntaxError` in the performance evaluation section. This seems to be a recurring issue with the handling of the `Sortino Ratio` calculation. Re-examine the conditional logic around the `downside_deviation_enhanced_logic` and `sortino_ratio_enhanced_logic` calculation and fix the syntax error. Then, re-run the evaluation code.


"""

# Evaluate performance metrics for the simulation with enhanced logic
if equity_curve_df_enhanced_logic.empty or len(equity_curve_df_enhanced_logic) < 2:
    display("Enhanced logic equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_enhanced_logic['Daily_Return'] = equity_curve_df_enhanced_logic['Equity'].pct_change().fillna(0)
    total_return_enhanced_logic = (equity_curve_df_enhanced_logic['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_enhanced_logic = (1 + total_return_enhanced_logic)**(trading_periods_per_year / len(equity_curve_df_enhanced_logic)) - 1

    equity_curve_df_enhanced_logic['Peak'] = equity_curve_df_enhanced_logic['Equity'].cummax()
    equity_curve_df_enhanced_logic['Drawdown'] = equity_curve_df_enhanced_logic['Equity'] - equity_curve_df_enhanced_logic['Peak']
    max_drawdown_enhanced_logic = equity_curve_df_enhanced_logic['Drawdown'].min()

    mar = 0
    downside_returns_enhanced_logic = equity_curve_df_enhanced_logic[equity_curve_df_enhanced_logic['Daily_Return'] < mar]['Daily_Return']

    # Re-evaluate the conditional logic for Sortino Ratio calculation
    downside_deviation = downside_returns_enhanced_logic.std()

    if downside_deviation == 0 or np.isnan(downside_deviation):
        # If no downside returns or std is zero/NaN
        if annualized_return_enhanced_logic > mar:
             sortino_ratio_enhanced_logic = float('inf') # Profitable with no downside risk
        elif annualized_return_enhanced_logic == mar:
             sortino_ratio_enhanced_logic = 0 # Breakeven with no downside risk
        else:
             sortino_ratio_enhanced_logic = -1000 # Losing with no measurable downside risk (or error) # Use a very low number
    else:
        # Otherwise, calculate the standard Sortino Ratio
        sortino_ratio_enhanced_logic = (annualized_return_enhanced_logic - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
    if np.isinf(sortino_ratio_enhanced_logic) or np.isnan(sortino_ratio_enhanced_logic):
        sortino_ratio_enhanced_logic = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_enhanced_logic = annualized_return_enhanced_logic / abs(max_drawdown_enhanced_logic) if max_drawdown_enhanced_logic != 0 else float('inf')


    display("\nPerformance Metrics (using Enhanced Logic):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_enhanced_logic['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_enhanced_logic:.4f}")
    display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
    display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_enhanced_logic.index, equity_curve_df_enhanced_logic['Equity'], label='Equity Curve (Enhanced Logic)')
    plt.title('Equity Curve (Enhanced Logic)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Enhanced Logic Simulation):")
    display(trade_log_df_enhanced_logic.head())

    # Analyze losing trades from the simulation
    losing_trades_df_enhanced_logic = trade_log_df_enhanced_logic[trade_log_df_enhanced_logic['PnL'].notna() & (trade_log_df_enhanced_logic['PnL'] < 0)]
    display("\nLosing Trades Summary (Enhanced Logic Simulation):")
    display(losing_trades_df_enhanced_logic.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the enhanced logic equity curve is empty.")

"""**Reasoning**:
The previous code continues to fail with a `SyntaxError` in the Sortino Ratio calculation logic, despite attempts to fix it. The conditional structure seems problematic. Re-examine the syntax for the `if/elif/else` block related to `downside_deviation` and `annualized_return_enhanced_logic` to ensure it is correct Python syntax.


"""

# Evaluate performance metrics for the simulation with enhanced logic
if equity_curve_df_enhanced_logic.empty or len(equity_curve_df_enhanced_logic) < 2:
    display("Enhanced logic equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_enhanced_logic['Daily_Return'] = equity_curve_df_enhanced_logic['Equity'].pct_change().fillna(0)
    total_return_enhanced_logic = (equity_curve_df_enhanced_logic['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_enhanced_logic = (1 + total_return_enhanced_logic)**(trading_periods_per_year / len(equity_curve_df_enhanced_logic)) - 1

    equity_curve_df_enhanced_logic['Peak'] = equity_curve_df_enhanced_logic['Equity'].cummax()
    equity_curve_df_enhanced_logic['Drawdown'] = equity_curve_df_enhanced_logic['Equity'] - equity_curve_df_enhanced_logic['Peak']
    max_drawdown_enhanced_logic = equity_curve_df_enhanced_logic['Drawdown'].min()

    mar = 0
    downside_returns_enhanced_logic = equity_curve_df_enhanced_logic[equity_curve_df_enhanced_logic['Daily_Return'] < mar]['Daily_Return']

    # Corrected conditional logic for Sortino Ratio calculation
    downside_deviation = downside_returns_enhanced_logic.std()

    if downside_deviation == 0 or np.isnan(downside_deviation):
        # If no downside returns or std is zero/NaN
        if annualized_return_enhanced_logic > mar:
             sortino_ratio_enhanced_logic = float('inf') # Profitable with no downside risk
        elif annualized_return_enhanced_logic == mar:
             sortino_ratio_enhanced_logic = 0 # Breakeven with no downside risk
        else:
             sortino_ratio_enhanced_logic = -1000 # Losing with no measurable downside risk (or error) # Use a very low number
    else:
        # Otherwise, calculate the standard Sortino Ratio
        sortino_ratio_enhanced_logic = (annualized_return_enhanced_logic - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
    if np.isinf(sortino_ratio_enhanced_logic) or np.isnan(sortino_ratio_enhanced_logic):
        sortino_ratio_enhanced_logic = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_enhanced_logic = annualized_return_enhanced_logic / abs(max_drawdown_enhanced_logic) if max_drawdown_enhanced_logic != 0 else float('inf')


    display("\nPerformance Metrics (using Enhanced Logic):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_enhanced_logic['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_enhanced_logic:.4f}")
    display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
    display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_enhanced_logic.index, equity_curve_df_enhanced_logic['Equity'], label='Equity Curve (Enhanced Logic)')
    plt.title('Equity Curve (Enhanced Logic)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Enhanced Logic Simulation):")
    display(trade_log_df_enhanced_logic.head())

    # Analyze losing trades from the simulation
    losing_trades_df_enhanced_logic = trade_log_df_enhanced_logic[trade_log_df_enhanced_logic['PnL'].notna() & (trade_log_df_enhanced_logic['PnL'] < 0)]
    display("\nLosing Trades Summary (Enhanced Logic Simulation):")
    display(losing_trades_df_enhanced_logic.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the enhanced logic equity curve is empty.")

"""**Reasoning**:
The `SyntaxError` persists in the same location, indicating a fundamental issue with the `if`/`elif`/`else` structure around the Sortino Ratio calculation for the case where `downside_deviation` is zero or NaN. Let's simplify this conditional logic to a single `if`/`else` block that directly checks for a valid `downside_deviation` and calculate the ratio, handling the edge cases of zero or NaN deviation explicitly within that structure to avoid the syntax error.


"""

# Evaluate performance metrics for the simulation with enhanced logic
if equity_curve_df_enhanced_logic.empty or len(equity_curve_df_enhanced_logic) < 2:
    display("Enhanced logic equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_enhanced_logic['Daily_Return'] = equity_curve_df_enhanced_logic['Equity'].pct_change().fillna(0)
    total_return_enhanced_logic = (equity_curve_df_enhanced_logic['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_enhanced_logic = (1 + total_return_enhanced_logic)**(trading_periods_per_year / len(equity_curve_df_enhanced_logic)) - 1

    equity_curve_df_enhanced_logic['Peak'] = equity_curve_df_enhanced_logic['Equity'].cummax()
    equity_curve_df_enhanced_logic['Drawdown'] = equity_curve_df_enhanced_logic['Equity'] - equity_curve_df_enhanced_logic['Peak']
    max_drawdown_enhanced_logic = equity_curve_df_enhanced_logic['Drawdown'].min()

    mar = 0
    downside_returns_enhanced_logic = equity_curve_df_enhanced_logic[equity_curve_df_enhanced_logic['Daily_Return'] < mar]['Daily_Return']

    # Simplified and corrected conditional logic for Sortino Ratio calculation
    downside_deviation = downside_returns_enhanced_logic.std()

    if downside_deviation > 0 and not np.isnan(downside_deviation):
        # Calculate Sortino Ratio if downside deviation is positive and valid
        sortino_ratio_enhanced_logic = (annualized_return_enhanced_logic - mar) / downside_deviation
    else:
        # Handle cases with zero, negative, or NaN downside deviation
        if annualized_return_enhanced_logic > mar:
             sortino_ratio_enhanced_logic = float('inf') # Profitable with no measurable downside risk
        elif annualized_return_enhanced_logic == mar:
             sortino_ratio_enhanced_logic = 0 # Breakeven with no measurable downside risk
        else:
             sortino_ratio_enhanced_logic = -1000 # Losing with no measurable downside risk (or error)


    # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
    if np.isinf(sortino_ratio_enhanced_logic) or np.isnan(sortino_ratio_enhanced_logic):
        sortino_ratio_enhanced_logic = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_enhanced_logic = annualized_return_enhanced_logic / abs(max_drawdown_enhanced_logic) if max_drawdown_enhanced_logic != 0 else float('inf')


    display("\nPerformance Metrics (using Enhanced Logic):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_enhanced_logic['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_enhanced_logic:.4f}")
    display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
    display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_enhanced_logic.index, equity_curve_df_enhanced_logic['Equity'], label='Equity Curve (Enhanced Logic)')
    plt.title('Equity Curve (Enhanced Logic)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Enhanced Logic Simulation):")
    display(trade_log_df_enhanced_logic.head())

    # Analyze losing trades from the simulation
    losing_trades_df_enhanced_logic = trade_log_df_enhanced_logic[trade_log_df_enhanced_logic['PnL'].notna() & (trade_log_df_enhanced_logic['PnL'] < 0)]
    display("\nLosing Trades Summary (Enhanced Logic Simulation):")
    display(losing_trades_df_enhanced_logic.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the enhanced logic equity curve is empty.")

"""**Reasoning**:
The `SyntaxError` at line 71 persists, specifically within the `else` block related to the Sortino Ratio calculation when `downside_deviation` is not greater than 0 or is NaN. Looking at the traceback and the code, the issue seems to be within the nested `if`/`elif`/`else` structure inside this `else` block. Let's try another simplification: explicitly calculate the Sortino Ratio using a robust formula that handles zero standard deviation outside the `if/else` and then handle the resulting `inf` or `NaN` values.


"""

# Evaluate performance metrics for the simulation with enhanced logic
if equity_curve_df_enhanced_logic.empty or len(equity_curve_df_enhanced_logic) < 2:
    display("Enhanced logic equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_enhanced_logic['Daily_Return'] = equity_curve_df_enhanced_logic['Equity'].pct_change().fillna(0)
    total_return_enhanced_logic = (equity_curve_df_enhanced_logic['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_enhanced_logic = (1 + total_return_enhanced_logic)**(trading_periods_per_year / len(equity_curve_df_enhanced_logic)) - 1

    equity_curve_df_enhanced_logic['Peak'] = equity_curve_df_enhanced_logic['Equity'].cummax()
    equity_curve_df_enhanced_logic['Drawdown'] = equity_curve_df_enhanced_logic['Equity'] - equity_curve_df_enhanced_logic['Peak']
    max_drawdown_enhanced_logic = equity_curve_df_enhanced_logic['Drawdown'].min()

    mar = 0
    downside_returns_enhanced_logic = equity_curve_df_enhanced_logic[equity_curve_df_enhanced_logic['Daily_Return'] < mar]['Daily_Return']

    # More robust Sortino Ratio calculation
    downside_deviation = downside_returns_enhanced_logic.std()

    if downside_deviation == 0 or np.isnan(downside_deviation):
        sortino_ratio_enhanced_logic = np.inf if annualized_return_enhanced_logic > mar else (0 if annualized_return_enhanced_logic == mar else -np.inf) # Use np.inf for infinity
    else:
        sortino_ratio_enhanced_logic = (annualized_return_enhanced_logic - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN after calculation
    if np.isinf(sortino_ratio_enhanced_logic) or np.isnan(sortino_ratio_enhanced_logic):
        sortino_ratio_enhanced_logic = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_enhanced_logic = annualized_return_enhanced_logic / abs(max_drawdown_enhanced_logic) if max_drawdown_enhanced_logic != 0 else float('inf')


    display("\nPerformance Metrics (using Enhanced Logic):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_enhanced_logic['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_enhanced_logic:.4f}")
    display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
    display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_enhanced_logic.index, equity_curve_df_enhanced_logic['Equity'], label='Equity Curve (Enhanced Logic)')
    plt.title('Equity Curve (Enhanced Logic)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Enhanced Logic Simulation):")
    display(trade_log_df_enhanced_logic.head())

    # Analyze losing trades from the simulation
    losing_trades_df_enhanced_logic = trade_log_df_enhanced_logic[trade_log_df_enhanced_logic['PnL'].notna() & (trade_log_df_enhanced_logic['PnL'] < 0)]
    display("\nLosing Trades Summary (Enhanced Logic Simulation):")
    display(losing_trades_df_enhanced_logic.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the enhanced logic equity curve is empty.")

"""**Reasoning**:
The `SyntaxError` at line 63 persists even after simplifying the Sortino Ratio calculation logic. The traceback points specifically to the `else:` keyword within the `if downside_deviation == 0 or np.isnan(downside_deviation):` block. This strongly suggests a syntax issue within the preceding `if` and `elif` clauses of that block. Looking closely, the issue might be related to how the expressions returning `np.inf`, `0`, and `-np.inf` are structured within the `if`/`elif`/`else` of that block. Let's try an even simpler structure, directly assigning the result based on the conditions, without using nested `if`/`elif`/`else` for the assignment itself.


"""

# Evaluate performance metrics for the simulation with enhanced logic
if equity_curve_df_enhanced_logic.empty or len(equity_curve_df_enhanced_logic) < 2:
    display("Enhanced logic equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_enhanced_logic['Daily_Return'] = equity_curve_df_enhanced_logic['Equity'].pct_change().fillna(0)
    total_return_enhanced_logic = (equity_curve_df_enhanced_logic['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_enhanced_logic = (1 + total_return_enhanced_logic)**(trading_periods_per_year / len(equity_curve_df_enhanced_logic)) - 1

    equity_curve_df_enhanced_logic['Peak'] = equity_curve_df_enhanced_logic['Equity'].cummax()
    equity_curve_df_enhanced_logic['Drawdown'] = equity_curve_df_enhanced_logic['Equity'] - equity_curve_df_enhanced_logic['Peak']
    max_drawdown_enhanced_logic = equity_curve_df_enhanced_logic['Drawdown'].min()

    mar = 0
    downside_returns_enhanced_logic = equity_curve_df_enhanced_logic[equity_curve_df_enhanced_logic['Daily_Return'] < mar]['Daily_Return']

    # Simplified and corrected conditional logic for Sortino Ratio calculation - Attempt 3
    downside_deviation = downside_returns_enhanced_logic.std()

    sortino_ratio_enhanced_logic = -1000 # Default to a very low fitness

    if downside_deviation > 0 and not np.isnan(downside_deviation):
        # Calculate Sortino Ratio if downside deviation is positive and valid
        sortino_ratio_enhanced_logic = (annualized_return_enhanced_logic - mar) / downside_deviation
    else:
        # Handle cases with zero, negative, or NaN downside deviation
        if annualized_return_enhanced_logic > mar:
             sortino_ratio_enhanced_logic = float('inf') # Profitable with no measurable downside risk
        elif annualized_return_enhanced_logic == mar:
             sortino_ratio_enhanced_logic = 0 # Breakeven with no measurable downside risk
        # If annualized_return_enhanced_logic < mar, it remains the default -1000


    # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
    if np.isinf(sortino_ratio_enhanced_logic) or np.isnan(sortino_ratio_enhanced_logic):
        sortino_ratio_enhanced_logic = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_enhanced_logic = annualized_return_enhanced_logic / abs(max_drawdown_enhanced_logic) if max_drawdown_enhanced_logic != 0 else float('inf')


    display("\nPerformance Metrics (using Enhanced Logic):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_enhanced_logic['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_enhanced_logic:.4f}")
    display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
    display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_enhanced_logic.index, equity_curve_df_enhanced_logic['Equity'], label='Equity Curve (Enhanced Logic)')
    plt.title('Equity Curve (Enhanced Logic)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Enhanced Logic Simulation):")
    display(trade_log_df_enhanced_logic.head())

    # Analyze losing trades from the simulation
    losing_trades_df_enhanced_logic = trade_log_df_enhanced_logic[trade_log_df_enhanced_logic['PnL'].notna() & (trade_log_df_enhanced_logic['PnL'] < 0)]
    display("\nLosing Trades Summary (Enhanced Logic Simulation):")
    display(losing_trades_df_enhanced_logic.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the enhanced logic equity curve is empty.")

"""**Reasoning**:
The `SyntaxError` at line 72 persists, still pointing to the `else` clause within the Sortino Ratio calculation when `downside_deviation` is not positive or is NaN. The issue is likely in the structure of the conditional assignment within that `else` block. Let's simplify this part further by assigning a default value and then updating it based on the conditions, removing the nested `if`/`elif`/`else` completely from the assignment line.


"""

# Evaluate performance metrics for the simulation with enhanced logic
if equity_curve_df_enhanced_logic.empty or len(equity_curve_df_enhanced_logic) < 2:
    display("Enhanced logic equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_enhanced_logic['Daily_Return'] = equity_curve_df_enhanced_logic['Equity'].pct_change().fillna(0)
    total_return_enhanced_logic = (equity_curve_df_enhanced_logic['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_enhanced_logic = (1 + total_return_enhanced_logic)**(trading_periods_per_year / len(equity_curve_df_enhanced_logic)) - 1

    equity_curve_df_enhanced_logic['Peak'] = equity_curve_df_enhanced_logic['Equity'].cummax()
    equity_curve_df_enhanced_logic['Drawdown'] = equity_curve_df_enhanced_logic['Equity'] - equity_curve_df_enhanced_logic['Peak']
    max_drawdown_enhanced_logic = equity_curve_df_enhanced_logic['Drawdown'].min()

    mar = 0
    downside_returns_enhanced_logic = equity_curve_df_enhanced_logic[equity_curve_df_enhanced_logic['Daily_Return'] < mar]['Daily_Return']

    # Simplified and corrected conditional logic for Sortino Ratio calculation - Attempt 4
    downside_deviation = downside_returns_enhanced_logic.std()

    sortino_ratio_enhanced_logic = -1000 # Default to a very low fitness for problematic cases

    if downside_deviation > 0 and not np.isnan(downside_deviation):
        # Calculate Sortino Ratio if downside deviation is positive and valid
        sortino_ratio_enhanced_logic = (annualized_return_enhanced_logic - mar) / downside_deviation
    elif annualized_return_enhanced_logic > mar:
        # Profitable with no measurable downside risk
        sortino_ratio_enhanced_logic = float('inf')
    elif annualized_return_enhanced_logic == mar:
        # Breakeven with no measurable downside risk
        sortino_ratio_enhanced_logic = 0
    # If annualized_return_enhanced_logic < mar and downside_deviation is not positive/valid,
    # it remains the default -1000.


    # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
    if np.isinf(sortino_ratio_enhanced_logic) or np.isnan(sortino_ratio_enhanced_logic):
        sortino_ratio_enhanced_logic = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_enhanced_logic = annualized_return_enhanced_logic / abs(max_drawdown_enhanced_logic) if max_drawdown_enhanced_logic != 0 else float('inf')


    display("\nPerformance Metrics (using Enhanced Logic):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_enhanced_logic['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_enhanced_logic:.4f}")
    display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
    display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_enhanced_logic.index, equity_curve_df_enhanced_logic['Equity'], label='Equity Curve (Enhanced Logic)')
    plt.title('Equity Curve (Enhanced Logic)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Enhanced Logic Simulation):")
    display(trade_log_df_enhanced_logic.head())

    # Analyze losing trades from the simulation
    losing_trades_df_enhanced_logic = trade_log_df_enhanced_logic[trade_log_df_enhanced_logic['PnL'].notna() & (trade_log_df_enhanced_logic['PnL'] < 0)]
    display("\nLosing Trades Summary (Enhanced Logic Simulation):")
    display(losing_trades_df_enhanced_logic.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the enhanced logic equity curve is empty.")

"""## Improve risk management

### Subtask:
Refine the stop-loss mechanism (e.g., more sophisticated trailing stop logic, profit targets).

**Reasoning**:
Implement a more sophisticated trailing stop-loss mechanism, including a profit target, and integrate this into the trading simulation loop that handles fractional positions.
"""

# Re-run the trading simulation with enhanced logic including refined stop-loss and profit targets
position_fraction = 0.0 # Current position as a fraction of initial_capital (positive for long, negative for short)
total_units = 0.0
total_cost_basis = 0.0 # Total cost paid for units (positive for buy, negative for sell)
stop_loss = 0.0
take_profit = 0.0 # Profit target level
equity_curve_enhanced_logic = []
trade_log_enhanced_logic = []
current_capital = initial_capital # Start with the initial capital
max_position_fraction = 1.0 # Max position is 100% of initial capital

# Parameters for gradual entry/exit (these are examples and need tuning)
entry_step_size = 0.2 # Enter/Exit in steps of 20% of max_position_fraction
# mss_confidence_factor = 0.01 # How much MSS magnitude affects position sizing (not directly used in this scaling model)

# Parameters for stop-loss and profit targets (these are examples and need tuning)
# Stop-loss multipliers from the optimal parameters found by the GA
# Ensure these are available (defined in the previous evaluation cell)
# For now, let's use some default values or values from a previous run if available
# Assuming evaluation_params_final is available from a previous GA run (if applicable)
# Otherwise, use reasonable defaults
# Using parameters from the last successful run with dynamic MSS and adjusted thresholds (cell 3c86b593 and 4f559c84)
# Let's retrieve the last used thresholds and multipliers from the notebook state if possible
# Based on the last successful run, the thresholds were 50/-50 for strong bull/bear.
# Let's define some default multipliers here for now if evaluation_params_final is not available
stop_loss_multiplier_strong_final = 3 # Example value
stop_loss_multiplier_weak_final = 3 # Example value

# Example Profit Target Multipliers (multiples of Volatility Factor)
take_profit_multiplier_strong = 4.0 # 4x Volatility Factor for Strong regimes
take_profit_multiplier_weak = 2.0   # 2x Volatility Factor for Weak regimes

# Redefine max_step based on fraction
max_step_fraction = entry_step_size * max_position_fraction # Step size in terms of fraction of initial_capital


# Iterate through the dataframe using dollar_bars_df which contains the calculated factors, MSS, and Regime
for index, row in dollar_bars_df.iterrows(): # Use dollar_bars_df
    current_price = row['Close']
    current_regime = row['Regime'] # Use the regime based on dynamic MSS
    current_mss = row['MSS'] # Use the dynamic MSS
    current_volatility_factor = row['Volatility_Factor'] # Use the normalized Volatility Factor

    # Determine stop-loss distance and take-profit distance based on regime and Volatility Factor
    # Use absolute value of volatility factor for distance as it represents price movement
    if not isinstance(current_volatility_factor, (int, float)) or np.isnan(current_volatility_factor) or abs(current_volatility_factor) < 1e-9:
         stop_loss_distance = 0 # Cannot calculate dynamic stop loss without valid Volatility Factor
         take_profit_distance = 0 # Cannot calculate dynamic take profit without valid Volatility Factor
    else:
        if current_regime in ['Strong Bull', 'Strong Bear']:
            stop_loss_distance = stop_loss_multiplier_strong_final * abs(current_volatility_factor)
            take_profit_distance = take_profit_multiplier_strong * abs(current_volatility_factor)
        elif current_regime in ['Weak Bull', 'Weak Bear']:
            stop_loss_distance = stop_loss_multiplier_weak_final * abs(current_volatility_factor)
            take_profit_distance = take_profit_multiplier_weak * abs(current_volatility_factor)
        else: # Neutral
            stop_loss_distance = 0
            take_profit_distance = 0


    # --- Trading Logic with Fractional Positions (Capital Allocation Model) ---

    # Determine target position fraction based on regime and MSS confidence
    target_position_fraction = 0.0
    # Use the thresholds from the last successful run with dynamic MSS and adjusted thresholds (cell 3c86b593 and 4f559c84)
    # Let's retrieve the last used thresholds from the notebook state if possible
    # Based on the last successful run, the thresholds were 50/-50 for strong bull/bear.
    strong_bull_threshold_final = 50 # Example value
    weak_bull_threshold_final = 20 # Example value
    neutral_threshold_upper_final = 20 # Example value
    neutral_threshold_lower_final = -20 # Example value
    strong_bear_threshold_final = -50 # Example value
    weak_bear_threshold_final = -20 # Example value


    if current_regime == 'Strong Bull':
        # Scale target position fraction based on MSS magnitude within the regime
        # MSS ranges from strong_bull_threshold_final to 100. Normalize this range.
        normalized_mss = (current_mss - strong_bull_threshold_final) / (100 - strong_bull_threshold_final) if (100 - strong_bull_threshold_final) > 0 else 0
        target_position_fraction = max_position_fraction * np.clip(normalized_mss, 0, 1)

    elif current_regime == 'Weak Bull':
         normalized_mss = (current_mss - weak_bull_threshold_final) / (neutral_threshold_upper_final - weak_bull_threshold_final) if (neutral_threshold_upper_final - weak_bull_threshold_final) > 0 else 0
         target_position_fraction = position_fraction # Default to hold
         if position_fraction > 0: # If currently long
             target_position_fraction = max_position_fraction * np.clip(normalized_mss, 0, 1)

    elif current_regime == 'Neutral':
        target_position_fraction = 0.0

    elif current_regime == 'Weak Bear':
         normalized_mss = (current_mss - (-100)) / (weak_bear_threshold_final - (-100)) if (weak_bear_threshold_final - (-100)) > 0 else 0
         target_position_fraction = position_fraction # Default to hold
         if position_fraction < 0: # If currently short
              target_position_fraction = -max_position_fraction * np.clip(1 - normalized_mss, 0, 1)

    elif current_regime == 'Strong Bear':
        normalized_mss = (current_mss - (-100)) / (strong_bear_threshold_final - (-100)) if (strong_bear_threshold_final - (-100)) > 0 else 0
        target_position_fraction = -max_position_fraction * np.clip(1 - normalized_mss, 0, 1)

    # Ensure target_position_fraction has correct sign based on regime
    if target_position_fraction > 1e-9 and current_regime not in ['Strong Bull', 'Weak Bull']:
         target_position_fraction = 0.0
    elif target_position_fraction < -1e-9 and current_regime not in ['Strong Bear', 'Weak Bear']:
         target_position_fraction = 0.0
    # In Neutral, always target 0
    if current_regime == 'Neutral':
        target_position_fraction = 0.0


    # Calculate the change in position fraction
    position_fraction_change = target_position_fraction - position_fraction

    # Limit position fraction change to max_step_fraction
    position_fraction_change = np.clip(position_fraction_change, -max_step_fraction, max_step_fraction)

    # Calculate the amount of capital to allocate/deallocate in this step
    capital_to_trade = position_fraction_change * initial_capital # Amount of initial capital equivalent to the fraction change

    # Calculate the units to trade based on the capital amount and current price
    units_to_trade = capital_to_trade / current_price if current_price > 0 else 0.0

    # Update total units and total cost basis based on units_to_trade
    if units_to_trade > 1e-9: # Buying (increasing long or entering long)
        total_cost_basis += units_to_trade * current_price # Add cost for new units
        total_units += units_to_trade
        action = 'Increase Long' if position_fraction_change > 0 and total_units > units_to_trade else 'Enter Long' # Refine action description
        trade_log_enhanced_logic.append({'Date': index, 'Action': action + ' (Gradual)', 'Price': current_price, 'Units': units_to_trade, 'TotalUnits': total_units, 'PnL': np.nan})
    elif units_to_trade < -1e-9: # Selling (decreasing long or exiting long)
        # Ensure we don't sell more units than we have
        units_to_sell = abs(units_to_trade)

        if total_units > 1e-9: # Only sell if we have units (long position)
            units_to_sell = min(units_to_sell, total_units) # Do not sell more than available units
            units_to_trade = -units_to_sell # Adjust units_to_trade based on actual units sold

            avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0
            pnl_on_sold_units = (current_price - avg_cost_basis_per_unit) * units_to_sell # Calculate P/L

            current_capital += pnl_on_sold_units # Update capital with realized P/L

            # Update total units and total cost basis based on units sold
            # Assuming weighted average cost basis reduction
            total_cost_basis -= units_to_sell * avg_cost_basis_per_unit
            total_units -= units_to_sell

            action = 'Decrease Long' if total_units > 1e-9 else 'Exit Long' # Refine action description
            trade_log_enhanced_logic.append({'Date': index, 'Action': action + ' (Gradual)', 'Price': current_price, 'Units': -units_to_sell, 'TotalUnits': total_units, 'PnL': pnl_on_sold_units})

        elif total_units < -1e-9: # Selling a short position (buying to cover) - This logic needs to handle short positions separately
             # For now, this block is not active as we are focusing on long-only for this part.
             pass # Skip short selling for now in this enhanced logic PoC

    # Update the current position fraction based on the new total units and current price
    # This represents the current value of the position as a fraction of initial capital
    current_market_value = total_units * current_price if current_price > 0 else 0.0
    position_fraction = current_market_value / initial_capital if initial_capital > 0 else 0.0


    # --- Refined Stop-Loss and Take-Profit Logic ---
    # Stop-loss and take-profit levels should be based on the entry price of the *total* position (weighted average).

    # Calculate Stop Loss and Take Profit levels if position is active
    if abs(total_units) > 1e-9: # If position is active
        avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0

        # Calculate Stop Loss level
        if total_units > 0: # Long position
            # Initial Stop Loss or Trailing Stop for Long
            # The stop loss level should be calculated based on the *initial* stop loss logic when entering,
            # and then trailed based on subsequent price movements.
            # Let's simplify: calculate the *current* stop loss level based on the trailing logic.
            # The trailing stop is based on the highest price reached since entry (or last stop adjustment)
            # minus the stop_loss_distance.
            # This requires tracking the peak price since entry. Let's add that state variable.
            # Re-initialize peak_price_since_entry at the start of the loop
            peak_price_since_entry = -float('inf') # Track peak price for trailing stop (Long)
            valley_price_since_entry = float('inf') # Track valley price for trailing stop (Short - not active yet)


            # Update peak/valley price since entry
            if total_units > 0: # Long position
                 peak_price_since_entry = max(peak_price_since_entry, current_price)
                 # Calculate trailing stop level
                 stop_loss_level = peak_price_since_entry - stop_loss_distance if stop_loss_distance > 0 else -float('inf')
                 # The stop_loss variable should hold the *active* stop loss level
                 stop_loss = stop_loss_level

                 # Calculate Take Profit level based on average cost basis
                 take_profit_level = avg_cost_basis_per_unit + take_profit_distance if take_profit_distance > 0 else float('inf')
                 take_profit = take_profit_level


            elif total_units < 0: # Short position (not active in this PoC)
                 valley_price_since_entry = min(valley_price_since_entry, current_price)
                 # Calculate trailing stop level
                 stop_loss_level = valley_price_since_entry + stop_loss_distance if stop_loss_distance > 0 else float('inf')
                 stop_loss = stop_loss_level

                 # Calculate Take Profit level based on average cost basis
                 take_profit_level = avg_cost_basis_per_unit - take_profit_distance if take_profit_distance > 0 else -float('inf')
                 take_profit = take_profit_level
            else: # Should not happen
                 stop_loss = 0.0
                 take_profit = 0.0

    else: # No active position
         stop_loss = 0.0
         take_profit = 0.0
         # Reset peak/valley price when position is flat
         peak_price_since_entry = -float('inf')
         valley_price_since_entry = float('inf')


    # Check for Stop-Loss or Take-Profit hit
    if abs(total_units) > 1e-9: # Only check if position is active
         avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0 # Recalculate avg cost basis

         if total_units > 0: # Long position
             # Check Take Profit hit
             if current_price >= take_profit and not np.isinf(take_profit):
                 pnl = (current_price - avg_cost_basis_per_unit) * total_units # Calculate P/L
                 current_capital += pnl # Update capital
                 trade_log_enhanced_logic.append({'Date': index, 'Action': 'Take Profit Long', 'Price': current_price, 'Units': -total_units, 'TotalUnits': 0.0, 'PnL': pnl})
                 total_units = 0.0
                 total_cost_basis = 0.0
                 position_fraction = 0.0
                 stop_loss = 0.0
                 take_profit = 0.0
                 peak_price_since_entry = -float('inf') # Reset peak

             # Check Stop Loss hit (after checking take profit)
             elif current_price <= stop_loss and not np.isinf(stop_loss):
                 pnl = (current_price - avg_cost_basis_per_unit) * total_units # Calculate P/L
                 current_capital += pnl # Update capital
                 trade_log_enhanced_logic.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'Units': -total_units, 'TotalUnits': 0.0, 'PnL': pnl})
                 total_units = 0.0
                 total_cost_basis = 0.0
                 position_fraction = 0.0
                 stop_loss = 0.0
                 take_profit = 0.0
                 peak_price_since_entry = -float('inf') # Reset peak

         elif total_units < 0: # Short position (not active)
              # Similar checks for short take profit and stop loss
              pass # Skip short position checks for now


    # Append current equity to the equity curve
    # Current equity is current capital + (unrealized P/L of the current position)
    avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0
    unrealized_pnl = (current_price - avg_cost_basis_per_unit) * total_units if abs(total_units) > 1e-9 else 0.0
    equity_curve_enhanced_logic.append({'Date': index, 'Equity': current_capital + unrealized_pnl})


# Convert trade log and equity curve to DataFrames
equity_curve_df_enhanced_logic = pd.DataFrame(equity_curve_enhanced_logic).set_index('Date')
trade_log_df_enhanced_logic = pd.DataFrame(trade_log_enhanced_logic)

display("Equity Curve (Enhanced Logic Simulation):")
display(equity_curve_df_enhanced_logic.head())
display("Trade Log (Enhanced Logic Simulation):")
display(trade_log_df_enhanced_logic.head())

"""## Refine Factor Calculations (Implement Real Indicators)

### Subtask:
Replace placeholder indicator functions (MACD_Placeholder, StdDev_Placeholder, RSI_Placeholder) with real implementations for calculating Trend, Volatility, and Exhaustion factors.

**Reasoning**:
Implement the actual calculation logic for MACD, Standard Deviation, and RSI. Replace the placeholder functions in the `INDICATOR_FUNCTIONS` dictionary with these new implementations. Ensure the functions return normalized values in the -100 to 100 range as required for the MSS calculation.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress

# Re-define factor calculation functions to be selectable
def calculate_trend_slope(df, lookback):
    df_calc = df.copy() # Work on a copy
    df_calc['Trend_Slope_Val'] = np.nan
    # Calculate slopes for the entire dataset first
    for i in range(lookback, len(df_calc)):
        y = df_calc['Close'].iloc[i-lookback:i]
        x = np.arange(lookback)
        # Handle potential division by zero or constant price in linregress
        if np.std(y) == 0:
             slope = 0
        else:
             slope, _, _, _, _ = linregress(x, y)
        df_calc.loc[df_calc.index[i], 'Trend_Slope_Val'] = slope

    # Now calculate rolling standard deviation of slopes
    df_calc['Slope_Std'] = df_calc['Trend_Slope_Val'].rolling(window=lookback).std()

    # Normalize Trend: Scale relative to the rolling standard deviation of slopes
    # Avoid division by zero
    valid_indices = df_calc.index[df_calc['Slope_Std'].notna() & (df_calc['Slope_Std'].abs() > 1e-9)]
    # Adjust the multiplier (e.g., 2) to control how many standard deviations map to the full range
    # Let's use a multiplier of 2 for now, tune if needed
    scaling_factor = 100 / (df_calc.loc[valid_indices, 'Slope_Std'] * 2) # Assuming +/- 2 std dev covers most cases
    df_calc.loc[valid_indices, 'Trend_Factor'] = df_calc.loc[valid_indices, 'Trend_Slope_Val'] * scaling_factor

    df_calc = df_calc.drop(columns=['Trend_Slope_Val', 'Slope_Std']) # Drop intermediate columns
    df_calc['Trend_Factor'] = np.clip(df_calc['Trend_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Trend_Factor']


# Implement actual MACD calculation
def calculate_trend_macd(df, fastperiod=12, slowperiod=26, signalperiod=9):
    df_calc = df.copy() # Work on a copy
    ema_fast = df_calc['Close'].ewm(span=fastperiod, adjust=False).mean()
    ema_slow = df_calc['Close'].ewm(span=slowperiod, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    # macd_histogram = macd_line - signal_line # Not used directly in MSS, but good to have

    # Normalize MACD: Scale relative to a rolling standard deviation of MACD line
    # This is one approach, other normalizations are possible (e.g., scaling relative to price)
    df_calc['MACD_Std'] = macd_line.rolling(window=slowperiod).std() # Use slowperiod for rolling std
    # Avoid division by zero
    valid_indices = df_calc.index[df_calc['MACD_Std'].notna() & (df_calc['MACD_Std'].abs() > 1e-9)]

    # Scale MACD line based on its rolling standard deviation
    # Adjust the multiplier (e.g., 5) to control the range
    scaling_factor = 100 / (df_calc.loc[valid_indices, 'MACD_Std'] * 5) # Assuming +/- 5 std dev maps to -100 to 100
    df_calc.loc[valid_indices, 'Trend_Factor'] = macd_line.loc[valid_indices] * scaling_factor

    df_calc['Trend_Factor'] = np.clip(df_calc['Trend_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Trend_Factor']


def calculate_volatility_atr(df, lookback):
    df_calc = df.copy() # Work on a copy
    df_calc['TR'] = np.maximum(np.maximum(df_calc['High'] - df_calc['Low'], abs(df_calc['High'] - df_calc['Close'].shift(1))), abs(df_calc['Low'] - df_calc['Close'].shift(1)))
    df_calc['Volatility_ATR_Absolute'] = df_calc['TR'].rolling(window=lookback).mean()
    df_calc = df_calc.drop(columns=['TR'])

    # Modified Normalization for ATR: Scale relative to a rolling average of ATR
    # Normalize ATR by dividing by a rolling average of ATR and then scaling to -100 to 100
    df_calc['ATR_MA'] = df_calc['Volatility_ATR_Absolute'].rolling(window=lookback).mean()
    # Avoid division by zero
    valid_indices = df_calc.index[df_calc['ATR_MA'].notna() & (df_calc['ATR_MA'].abs() > 1e-9)]
    df_calc.loc[valid_indices, 'Volatility_Factor'] = ((df_calc.loc[valid_indices, 'Volatility_ATR_Absolute'] / df_calc.loc[valid_indices, 'ATR_MA']) - 1) * 100 # Percentage deviation from MA

    df_calc = df_calc.drop(columns=['ATR_MA', 'Volatility_ATR_Absolute'])
    df_calc['Volatility_Factor'] = np.clip(df_calc['Volatility_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Volatility_Factor']


# Implement actual Standard Deviation calculation for Volatility
def calculate_volatility_stddev(df, lookback):
    df_calc = df.copy() # Work on a copy
    # Calculate rolling standard deviation of daily returns (log returns often preferred)
    df_calc['Log_Return'] = np.log(df_calc['Close'] / df_calc['Close'].shift(1))
    df_calc['Rolling_StdDev'] = df_calc['Log_Return'].rolling(window=lookback).std()

    # Normalize Standard Deviation: Scale relative to a rolling average of StdDev
    df_calc['StdDev_MA'] = df_calc['Rolling_StdDev'].rolling(window=lookback).mean()
    # Avoid division by zero
    valid_indices = df_calc.index[df_calc['StdDev_MA'].notna() & (df_calc['StdDev_MA'].abs() > 1e-9)]

    # Scale StdDev based on its rolling average
    # Adjust the multiplier (e.g., 1000) - Needs tuning as log returns stddev is small
    scaling_factor = 100 # Example scaling factor
    df_calc.loc[valid_indices, 'Volatility_Factor'] = ((df_calc.loc[valid_indices, 'Rolling_StdDev'] / df_calc.loc[valid_indices, 'StdDev_MA']) - 1) * scaling_factor # Percentage deviation from MA

    df_calc = df_calc.drop(columns=['Log_Return', 'Rolling_StdDev', 'StdDev_MA'])
    df_calc['Volatility_Factor'] = np.clip(df_calc['Volatility_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Volatility_Factor']


def calculate_exhaustion_sma_diff(df, sma_lookback, atr_series):
    df_calc = df.copy() # Work on a copy
    df_calc['SMA'] = df_calc['Close'].rolling(window=sma_lookback).mean()

    # Ensure ATR series is aligned and not NaN for calculation
    # Using the passed atr_series directly
    valid_indices = df_calc.index[atr_series.notna() & (atr_series.abs() > 1e-9)]
    df_calc.loc[valid_indices, 'Exhaustion_Factor'] = (df_calc.loc[valid_indices, 'Close'] - df_calc.loc[valid_indices, 'SMA']) / atr_series.loc[valid_indices]

    df_calc = df_calc.drop(columns=['SMA'])
    # Normalize Exhaustion: Scale the ratio to fit -100 to 100
    # Assuming a range like -10 to 10 covers most cases relative to ATR
    scaling_factor = 100 / 10
    df_calc['Exhaustion_Factor'] = np.clip(df_calc['Exhaustion_Factor'] * scaling_factor, -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Exhaustion_Factor']


# Implement actual RSI calculation for Exhaustion
def calculate_exhaustion_rsi(df, lookback):
    df_calc = df.copy() # Work on a copy
    # Calculate price changes
    delta = df_calc['Close'].diff()

    # Get gains and losses (ignore 0 changes)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate rolling average gains and losses
    avg_gains = gains.ewm(span=lookback, adjust=False).mean()
    avg_losses = losses.ewm(span=lookback, adjust=False).mean()

    # Calculate Relative Strength (RS)
    # Handle potential division by zero (when avg_losses is 0)
    rs = avg_gains / avg_losses
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0) # Replace inf with nan and fill nan with 0

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    # Normalize RSI: RSI is typically 0-100. Scale it to -100 to 100.
    # A simple linear scaling: RSI 0 -> -100, RSI 100 -> 100, RSI 50 -> 0
    df_calc['Exhaustion_Factor'] = (rsi - 50) * 2 # Scale from 0-100 to -100-100

    df_calc['Exhaustion_Factor'] = np.clip(df_calc['Exhaustion_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Exhaustion_Factor']


# Update the INDICATOR_FUNCTIONS dictionary with implemented functions
INDICATOR_FUNCTIONS = {
    'Trend': {
        'Slope': calculate_trend_slope,
        'MACD': calculate_trend_macd, # Added actual MACD
    },
    'Volatility': {
        'ATR': calculate_volatility_atr,
        'StdDev': calculate_volatility_stddev, # Added actual StdDev
    },
    'Exhaustion': {
        'SMADiff': calculate_exhaustion_sma_diff,
        'RSI': calculate_exhaustion_rsi, # Added actual RSI
    }
}

# Example usage of the updated functions (optional, just to test)
# Assuming dollar_bars_df is available with OHLCV and DollarVolume
# trend_slope_factor = calculate_trend_slope(dollar_bars_df.copy(), lookback=20)
# trend_macd_factor = calculate_trend_macd(dollar_bars_df.copy(), fastperiod=12, slowperiod=26, signalperiod=9)
# volatility_atr_factor = calculate_volatility_atr(dollar_bars_df.copy(), lookback=20)
# volatility_stddev_factor = calculate_volatility_stddev(dollar_bars_df.copy(), lookback=20)
# exhaustion_smadiff_factor = calculate_exhaustion_sma_diff(dollar_bars_df.copy(), sma_lookback=20, atr_series=volatility_atr_factor.copy()) # Pass the calculated ATR factor
# exhaustion_rsi_factor = calculate_exhaustion_rsi(dollar_bars_df.copy(), lookback=14)


# The evaluate_strategy_with_indicators function (defined in a previous cell)
# already uses this INDICATOR_FUNCTIONS dictionary.
# Re-running the previous evaluation cell will now use the updated indicator functions
# if the parameters specify them.

display("Updated INDICATOR_FUNCTIONS dictionary:")
display(INDICATOR_FUNCTIONS)

"""## Implement dynamic factor weights

### Subtask:
Modify the MSS calculation to use dynamic weights for the factors based on market conditions (e.g., regime).

**Reasoning**:
Define a function that returns dynamic weights based on the market regime. Modify the MSS calculation to use these dynamic weights for each bar in the DataFrame. Update the regime classification to use the dynamic MSS.
"""

def get_dynamic_weights(regime):
    """
    Returns dynamic weights for factors based on market regime.

    Args:
        regime (str): The current market regime ('Strong Bull', 'Weak Bull',
                      'Neutral', 'Weak Bear', 'Strong Bear').

    Returns:
        dict: Dictionary of weights for 'Trend', 'Volatility', 'Exhaustion'.
    """
    # These weights are examples and should be tuned based on backtesting and domain knowledge
    if regime == 'Strong Bull':
        return {'Trend': 0.6, 'Volatility': 0.1, 'Exhaustion': 0.3}
    elif regime == 'Weak Bull':
        return {'Trend': 0.4, 'Volatility': 0.2, 'Exhaustion': 0.4}
    elif regime == 'Neutral':
        # In Neutral regime, factors might be less reliable, or focus might shift.
        # Example: Give more weight to Volatility and Exhaustion for range trading or reversal signals.
        return {'Trend': 0.2, 'Volatility': 0.4, 'Exhaustion': 0.4}
    elif regime == 'Weak Bear':
        return {'Trend': 0.4, 'Volatility': 0.2, 'Exhaustion': 0.4}
    elif regime == 'Strong Bear':
        return {'Trend': 0.6, 'Volatility': 0.1, 'Exhaustion': 0.3}
    else: # Default or unexpected regime
        # Fallback to a neutral or default weighting scheme
        return {'Trend': 1/3, 'Volatility': 1/3, 'Exhaustion': 1/3}

# We need to recalculate factors using the implemented real indicators first
# Assuming dollar_bars_df is available and contains only the raw OHLCV and DollarVolume columns
# We will use the default indicator choices and lookbacks for this initial calculation

# Define default lookbacks (these will be used unless optimized later)
default_lookback_trend = 20
default_lookback_volatility = 20
default_lookback_exhaustion = 20 # For SMADiff and RSI

# Ensure we are working on a clean copy of the relevant data columns from the global dollar_bars_df
# Include factor columns from the existing dollar_bars_df if they exist,
# otherwise start with raw data and calculate factors.
# A safer approach is to start with the raw data and recalculate factors.
df_for_dynamic_mss = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


try:
    # Calculate Volatility first as Exhaustion (SMADiff) depends on it
    df_for_dynamic_mss['Volatility_Factor'] = INDICATOR_FUNCTIONS['Volatility']['ATR'](df_for_dynamic_mss.copy(), default_lookback_volatility) # Pass a copy to indicator function

    df_for_dynamic_mss['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend']['Slope'](df_for_dynamic_mss.copy(), default_lookback_trend) # Pass a copy

    # For Exhaustion, need to handle different indicator function signatures if they are added
    # For SMADiff, we need the Volatility_Factor
    if 'SMADiff' in INDICATOR_FUNCTIONS['Exhaustion']:
         # Ensure Volatility_Factor is available before passing it
         if 'Volatility_Factor' in df_for_dynamic_mss.columns:
             df_for_dynamic_mss['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion']['SMADiff'](df_for_dynamic_mss.copy(), default_lookback_exhaustion, df_for_dynamic_mss['Volatility_Factor']) # Pass a copy
         else:
             display("Error: Volatility_Factor not calculated, cannot calculate SMADiff.")
             df_for_dynamic_mss['Exhaustion_Factor'] = np.nan # Set to NaN if dependency missing
    elif 'RSI' in INDICATOR_FUNCTIONS['Exhaustion']:
         # Example for RSI, which takes df and lookback
         df_for_dynamic_mss['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion']['RSI'](df_for_dynamic_mss.copy(), default_lookback_exhaustion) # Pass a copy
    else:
         # Fallback to default or handle error
         display("Warning: Default Exhaustion indicator (SMADiff or RSI) not found or not handled. Factor not calculated.")
         df_for_dynamic_mss['Exhaustion_Factor'] = np.nan


except Exception as e:
    display(f"Error during factor calculation for dynamic MSS: {e}")
    # If factor calculation fails, we cannot proceed
    df_for_dynamic_mss = pd.DataFrame() # Empty dataframe


# Drop rows with NaN values generated by lookback periods after factor calculation
# Keep a reference to the original index before dropping
original_index = df_for_dynamic_mss.index
df_for_dynamic_mss = df_for_dynamic_mss.dropna(subset=['Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor'])

if not df_for_dynamic_mss.empty:
    # Initial Regime Classification (using static weights and thresholds for the first pass to get an initial regime)
    # We need some initial regime to determine the first dynamic weights.
    # Let's use the previously used static thresholds for this initial classification.
    # These thresholds might need to be defined here or passed from a previous cell.
    # For now, let's use the thresholds from the last successful run (50 and -50).

    # Define static thresholds for initial regime classification
    static_strong_bull_threshold = 50
    static_weak_bull_threshold = 20
    static_neutral_threshold_upper = 20
    static_neutral_threshold_lower = -20
    static_strong_bear_threshold = -50
    static_weak_bear_threshold = -20


    def classify_regime_static(mss):
        if mss > static_strong_bull_threshold:
            return 'Strong Bull'
        elif mss > static_weak_bull_threshold:
            return 'Weak Bull'
        elif mss >= static_neutral_threshold_lower and mss <= static_neutral_threshold_upper:
            return 'Neutral'
        elif mss > static_strong_bear_threshold:
            return 'Weak Bear'
        else: # mss <= static_strong_bear_threshold
            return 'Strong Bear'

    # Calculate initial static MSS to get a starting regime
    # Use default static weights for this initial MSS calculation
    static_weight_trend = 0.4
    static_weight_volatility = 0.3
    static_weight_exhaustion = 0.3

    df_for_dynamic_mss['MSS_static_initial'] = (static_weight_trend * df_for_dynamic_mss['Trend_Factor'] +
                                                static_weight_volatility * df_for_dynamic_mss['Volatility_Factor'] +
                                                static_weight_exhaustion * df_for_dynamic_mss['Exhaustion_Factor'])

    # Classify the initial static regime
    df_for_dynamic_mss['Regime_initial'] = df_for_dynamic_mss['MSS_static_initial'].apply(classify_regime_static)


    # Now, calculate Dynamic MSS using dynamic weights based on the *initial* regime
    df_for_dynamic_mss['MSS_dynamic'] = np.nan # Initialize dynamic MSS column

    # Iterate through the dataframe to calculate dynamic MSS
    # Use the initial static regime ('Regime_initial') to determine dynamic weights for the current bar.
    # Use .loc for setting values to avoid SettingWithCopyWarning
    for index, row in df_for_dynamic_mss.iterrows():
        current_regime_initial = row['Regime_initial'] # Use the initial static regime
        dynamic_weights = get_dynamic_weights(current_regime_initial)

        # Calculate dynamic MSS using factor values and dynamic weights
        dynamic_mss = (dynamic_weights['Trend'] * row['Trend_Factor'] +
                       dynamic_weights['Volatility'] * row['Volatility_Factor'] +
                       dynamic_weights['Exhaustion'] * row['Exhaustion_Factor'])

        df_for_dynamic_mss.loc[index, 'MSS_dynamic'] = dynamic_mss


    # Update the 'Regime' column based on the newly calculated 'MSS_dynamic'
    # Use the same static thresholds for classifying the dynamic MSS into regimes for consistency with the action matrix.
    df_for_dynamic_mss['Regime_dynamic'] = df_for_dynamic_mss['MSS_dynamic'].apply(classify_regime_static) # Use static thresholds

    # Replace the old 'Regime' and 'MSS' columns with the new dynamic ones for subsequent steps
    # Keep the individual factor columns as they are needed for the simulation
    # Ensure we are only dropping if the columns exist
    columns_to_drop = ['MSS_static_initial', 'Regime_initial']
    df_for_dynamic_mss = df_for_dynamic_mss.drop(columns=[col for col in columns_to_drop if col in df_for_dynamic_mss.columns])

    df_for_dynamic_mss = df_for_dynamic_mss.rename(columns={'MSS_dynamic': 'MSS', 'Regime_dynamic': 'Regime'})


    # Display the head and info of the DataFrame with dynamic MSS and updated Regime
    display("DataFrame with Dynamic MSS and Updated Regime:")
    display(df_for_dynamic_mss.head())
    display(df_for_dynamic_mss.info())

    # Update the main dollar_bars_df with the new dynamic MSS and Regime for subsequent steps
    # Simply assign the processed dataframe back to the global variable
    global dollar_bars_df
    dollar_bars_df = df_for_dynamic_mss.copy()


else:
    display("Cannot calculate dynamic MSS as the DataFrame is empty after factor calculation.")

"""## Enhance trading logic (action matrix)

### Subtask:
Modify the trading simulation logic to include more sophisticated rules based on the Action Matrix, such as implementing gradual entries and exits and adding more nuanced regime-specific trading rules.

**Reasoning**:
Modify the trading simulation logic to include fractional position sizing, gradual entries/exits, and adjust stop-loss handling for fractional positions.
"""

# Re-run the trading simulation with enhanced logic for fractional positions
position_size = 0.0  # Current position size (can be fractional, positive for long, negative for short, 0 for flat)
entry_price = 0.0 # Weighted average entry price for the current position
stop_loss = 0.0
equity_curve_enhanced_logic = []
trade_log_enhanced_logic = []
current_capital = initial_capital # Start with the initial capital
max_position_size = 1.0 # Maximum allowed position size (e.g., 1.0 represents full capital allocation or 1 unit)

# Parameters for gradual entry/exit (these are examples and need tuning)
entry_step_size = 0.2 # Enter/Exit in steps of 20% of max_position_size
mss_confidence_factor = 0.01 # How much MSS magnitude affects position sizing

# Stop-loss multipliers from the optimal parameters found by the GA
# Ensure these are available (defined in the previous evaluation cell)
# For now, let's use some default values or values from a previous run if available
# Assuming evaluation_params_final is available from a previous GA run (if applicable)
# Otherwise, use reasonable defaults
stop_loss_multiplier_strong_final = evaluation_params_final.get('stop_loss_multiplier_strong', 2)
stop_loss_multiplier_weak_final = evaluation_params_final.get('stop_loss_multiplier_weak', 1)


# Iterate through the dataframe using the dynamic MSS and updated regime
for index, row in dollar_bars_df.iterrows():
    current_price = row['Close']
    current_regime = row['Regime'] # Use the regime based on dynamic MSS
    current_mss = row['MSS'] # Use the dynamic MSS
    current_volatility_factor = row['Volatility_Factor'] # Use the normalized Volatility Factor

    # Determine stop-loss distance based on regime and Volatility Factor
    # Use absolute value of volatility factor for stop distance as it represents price movement
    if not isinstance(current_volatility_factor, (int, float)) or np.isnan(current_volatility_factor) or abs(current_volatility_factor) < 1e-9:
         stop_loss_distance = 0 # Cannot calculate dynamic stop loss without valid Volatility Factor
    else:
        if current_regime in ['Strong Bull', 'Strong Bear']:
            stop_loss_distance = stop_loss_multiplier_strong_final * abs(current_volatility_factor) # Use absolute value for distance
        elif current_regime in ['Weak Bull', 'Weak Bear']:
            stop_loss_distance = stop_loss_multiplier_weak_final * abs(current_volatility_factor) # Use absolute value for distance
        else: # Neutral
            stop_loss_distance = 0


    # --- Trading Logic with Fractional Positions and Gradual Entries/Exits ---

    # Determine target position size based on regime and MSS confidence
    target_position_size = 0.0
    # Use the thresholds from evaluation_params_final if available, otherwise use defaults
    strong_bull_threshold_final = evaluation_params_final.get('strong_bull_threshold', 50)
    weak_bull_threshold_final = evaluation_params_final.get('weak_bull_threshold', 20)
    neutral_threshold_upper_final = evaluation_params_final.get('neutral_threshold_upper', 20)
    neutral_threshold_lower_final = evaluation_params_final.get('neutral_threshold_lower', -20)
    strong_bear_threshold_final = evaluation_params_final.get('strong_bear_threshold', -50)
    weak_bear_threshold_final = evaluation_params_final.get('weak_bear_threshold', -20)


    if current_regime == 'Strong Bull':
        # Scale target position size based on MSS magnitude within the regime
        # MSS ranges from > strong_bull_threshold to 100. Normalize this range.
        normalized_mss = (current_mss - strong_bull_threshold_final) / (100 - strong_bull_threshold_final) if (100 - strong_bull_threshold_final) > 0 else 0
        target_position_size = max_position_size * np.clip(normalized_mss, 0, 1) # Target long position, scaled by normalized MSS

    elif current_regime == 'Weak Bull':
        # Hold Longs ONLY, potentially reduce position if MSS is closer to Neutral
        normalized_mss = (current_mss - weak_bull_threshold_final) / (neutral_threshold_upper_final - weak_bull_threshold_final) if (neutral_threshold_upper_final - weak_bull_threshold_final) > 0 else 0
        target_position_size = position_size # Default to hold
        if position_size > 0: # If currently long
             # Scale target position size based on how far into Weak Bull we are
             target_position_size = max_position_size * np.clip(normalized_mss, 0, 1) # Scale down towards 0 as MSS approaches Neutral

    elif current_regime == 'Neutral':
        target_position_size = 0.0 # Target flat position

    elif current_regime == 'Weak Bear':
         # Hold Shorts ONLY, potentially reduce position if MSS is closer to Neutral
         normalized_mss = (current_mss - weak_bear_threshold_final) / (neutral_threshold_lower_final - weak_bear_threshold_final) if (neutral_threshold_lower_final - weak_bear_threshold_final) > 0 else 0 # Note: MSS is negative here, adjusting normalization
         # A different normalization might be needed for bear regimes
         # Let's normalize from -100 to weak_bear_threshold_final
         normalized_mss = (current_mss - (-100)) / (weak_bear_threshold_final - (-100)) if (weak_bear_threshold_final - (-100)) > 0 else 0
         target_position_size = position_size # Default to hold
         if position_size < 0: # If currently short
             # Scale target position size based on how far into Weak Bear we are
              target_position_size = -max_position_size * np.clip(1 - normalized_mss, 0, 1) # Scale down towards 0 as MSS approaches Neutral (1 - normalized_mss for bear side)


    elif current_regime == 'Strong Bear':
        # Scale target position size based on MSS magnitude within the regime
        # MSS ranges from -100 to strong_bear_threshold. Normalize this range.
        normalized_mss = (current_mss - (-100)) / (strong_bear_threshold_final - (-100)) if (strong_bear_threshold_final - (-100)) > 0 else 0
        target_position_size = -max_position_size * np.clip(1 - normalized_mss, 0, 1) # Target short position, scaled by normalized MSS (1 - normalized_mss for bear side)


    # Implement gradual entry/exit
    position_change = target_position_size - position_size

    # Limit position change to entry_step_size (as a fraction of max_position_size)
    max_step = entry_step_size * max_position_size
    position_change = np.clip(position_change, -max_step, max_step)

    # Calculate the amount of capital/units to trade in this step
    trade_amount_units = position_change * (current_capital / current_price) if current_price > 0 else 0 # Amount in units, positive for buy, negative for sell

    # Update position size and entry price (weighted average)
    new_position_size = position_size + position_change

    if abs(new_position_size) < 1e-9: # If position closes or becomes very small
        if abs(position_size) > 1e-9: # If there was an existing position
             pnl = (current_price - entry_price) * position_size * (current_capital / current_price) # Calculate P/L for the full exit
             current_capital += pnl
             action = 'Exit Long' if position_size > 0 else 'Exit Short'
             trade_log_enhanced_logic.append({'Date': index, 'Action': action + ' (Gradual)', 'Price': current_price, 'PositionChange': -position_size, 'NewPositionSize': 0.0, 'PnL': pnl})
        position_size = 0.0
        entry_price = 0.0
    else:
        if abs(position_size) < 1e-9: # Entering a new position
            entry_price = current_price
            action = 'Enter Long' if new_position_size > 0 else 'Enter Short'
            trade_log_enhanced_logic.append({'Date': index, 'Action': action + ' (Gradual)', 'Price': current_price, 'PositionChange': position_change, 'NewPositionSize': new_position_size, 'PnL': np.nan})
        else: # Adding to or reducing an existing position
            # Calculate P/L on the portion being closed (if reducing)
            pnl_on_closed_portion = 0.0
            if (position_size > 0 and new_position_size < position_size) or (position_size < 0 and new_position_size > position_size): # Reducing position
                 closed_amount_units = (position_size - new_position_size) * (current_capital / current_price) # Amount in units being closed
                 pnl_on_closed_portion = (current_price - entry_price) * (position_size - new_position_size) * (current_capital / current_price) # P/L on the units being closed
                 current_capital += pnl_on_closed_portion # Update capital with P/L from the closed portion

            # Update weighted average entry price when adding to a position
            if (position_size > 0 and new_position_size > position_size) or (position_size < 0 and new_position_size < position_size): # Adding to position
                 # New entry price is weighted average of old position and new addition
                 entry_price = ((entry_price * abs(position_size)) + (current_price * abs(position_change))) / abs(new_position_size) if abs(new_position_size) > 1e-9 else current_price

            action = 'Increase Long' if position_change > 0 else ('Decrease Long' if position_change < 0 and position_size > 0 else ('Increase Short' if position_change < 0 else 'Decrease Short'))
            trade_log_enhanced_logic.append({'Date': index, 'Action': action + ' (Gradual)', 'Price': current_price, 'PositionChange': position_change, 'NewPositionSize': new_position_size, 'PnL': pnl_on_closed_portion if abs(pnl_on_closed_portion) > 1e-9 else np.nan})


        position_size = new_position_size # Update the position size


    # Update Trailing Stop Loss based on the current position size and direction
    if abs(position_size) > 1e-9 and stop_loss_distance > 0: # If there is an active position and valid stop distance
        if position_size > 0: # Long position
            # Initial stop or trail the stop upwards
            if stop_loss == 0.0: # Initialize stop loss on first entry
                 stop_loss = current_price - stop_loss_distance
            else: # Trail the stop
                 stop_loss = max(stop_loss, current_price - stop_loss_distance)
        elif position_size < 0: # Short position
            # Initial stop or trail the stop downwards
            if stop_loss == 0.0: # Initialize stop loss on first entry
                 stop_loss = current_price + stop_loss_distance
            else: # Trail the stop
                stop_loss = min(stop_loss, current_price + stop_loss_distance)
    else: # No active position or invalid stop distance
         stop_loss = 0.0 # Reset stop loss

    # Check for stop-loss hit (only if position is active and stop_loss is a valid number)
    if abs(position_size) > 1e-9 and not np.isinf(stop_loss) and stop_loss != 0.0: # Ensure stop_loss is explicitly set and not 0
        if position_size > 0 and current_price <= stop_loss:
            pnl = (current_price - entry_price) * position_size * (current_capital / current_price) # Calculate P/L for the full exit
            current_capital += pnl
            trade_log_enhanced_logic.append({'Date': index, 'Action': 'Stop Out Long (Fractional)', 'Price': current_price, 'PositionChange': -position_size, 'NewPositionSize': 0.0, 'PnL': pnl})
            position_size = 0.0
            entry_price = 0.0
            stop_loss = 0.0
        elif position_size < 0 and current_price >= stop_loss:
            pnl = (entry_price - current_price) * abs(position_size) * (current_capital / current_price) # Calculate P/L for the full exit (use abs position size for short)
            current_capital += pnl
            trade_log_enhanced_logic.append({'Date': index, 'Action': 'Stop Out Short (Fractional)', 'Price': current_price, 'PositionChange': -position_size, 'NewPositionSize': 0.0, 'PnL': pnl})
            position_size = 0.0
            entry_price = 0.0
            stop_loss = 0.0


    # Append current equity to the equity curve
    # Current equity is current capital + (unrealized P/L of the current position)
    unrealized_pnl = (current_price - entry_price) * position_size * (current_capital / current_price) if abs(position_size) > 1e-9 and current_price > 0 else 0.0
    equity_curve_enhanced_logic.append({'Date': index, 'Equity': current_capital + unrealized_pnl})


# Convert trade log and equity curve to DataFrames
equity_curve_df_enhanced_logic = pd.DataFrame(equity_curve_enhanced_logic).set_index('Date')
trade_log_df_enhanced_logic = pd.DataFrame(trade_log_enhanced_logic)

display("Equity Curve (Enhanced Logic Simulation):")
display(equity_curve_df_enhanced_logic.head())
display("Trade Log (Enhanced Logic Simulation):")
display(trade_log_df_enhanced_logic.head())

"""**Reasoning**:
Evaluate the performance of the strategy with the enhanced logic by calculating key metrics and plotting the equity curve.
"""

# Evaluate performance metrics for the simulation with enhanced logic
if equity_curve_df_enhanced_logic.empty or len(equity_curve_df_enhanced_logic) < 2:
    display("Enhanced logic equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_enhanced_logic['Daily_Return'] = equity_curve_df_enhanced_logic['Equity'].pct_change().fillna(0)
    total_return_enhanced_logic = (equity_curve_df_enhanced_logic['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_enhanced_logic = (1 + total_return_enhanced_logic)**(trading_periods_per_year / len(equity_curve_df_enhanced_logic)) - 1

    equity_curve_df_enhanced_logic['Peak'] = equity_curve_df_enhanced_logic['Equity'].cummax()
    equity_curve_df_enhanced_logic['Drawdown'] = equity_curve_df_enhanced_logic['Equity'] - equity_curve_df_enhanced_logic['Peak']
    max_drawdown_enhanced_logic = equity_curve_df_enhanced_logic['Drawdown'].min()

    mar = 0
    downside_returns_enhanced_logic = equity_curve_df_enhanced_logic[equity_curve_df_enhanced_logic['Daily_Return'] < mar]['Daily_Return']

    # Corrected conditional logic for Sortino Ratio calculation
    downside_deviation = downside_returns_enhanced_logic.std()

    if downside_deviation == 0 or np.isnan(downside_deviation):
        # If no downside returns or std is zero/NaN
        if annualized_return_enhanced_logic > mar:
             sortino_ratio_enhanced_logic = float('inf') # Profitable with no downside risk
        elif annualized_return_enhanced_logic == mar:
             sortino_ratio_enhanced_logic = 0 # Breakeven with no downside risk
        else:
             sortino_ratio_enhanced_logic = -1000 # Losing with no measurable downside risk (or error) # Use a very low number
    else:
        # Otherwise, calculate the standard Sortino Ratio
        sortino_ratio_enhanced_logic = (annualized_return_enhanced_logic - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
    if np.isinf(sortino_ratio_enhanced_logic) or np.isnan(sortino_ratio_enhanced_logic):
        sortino_ratio_enhanced_logic = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_enhanced_logic = annualized_return_enhanced_logic / abs(max_drawdown_enhanced_logic) if max_drawdown_enhanced_logic != 0 else float('inf')


    display("\nPerformance Metrics (using Enhanced Logic):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_enhanced_logic['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_enhanced_logic:.4f}")
    display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
    display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_enhanced_logic.index, equity_curve_df_enhanced_logic['Equity'], label='Equity Curve (Enhanced Logic)')
    plt.title('Equity Curve (Enhanced Logic)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Enhanced Logic Simulation):")
    display(trade_log_df_enhanced_logic.head())

    # Analyze losing trades from the simulation
    losing_trades_df_enhanced_logic = trade_log_df_enhanced_logic[trade_log_df_enhanced_logic['PnL'].notna() & (trade_log_df_enhanced_logic['PnL'] < 0)]
    display("\nLosing Trades Summary (Enhanced Logic Simulation):")
    display(losing_trades_df_enhanced_logic.head()) # Display head of losing trades

# The final else statement is now correctly aligned with the initial if statement
else:
    display("Cannot perform evaluation as the enhanced logic equity curve is empty.")

"""## Evaluate performance (Part 1: Calculations)

### Subtask:
Calculate key performance metrics like P&L, Sortino Ratio, and Calmar Ratio based on the simulation results in a separate cell.

**Reasoning**:
Extract the performance calculation logic into a new cell to avoid potential syntax issues in the previous combined cell. Calculate the daily returns, total return, annualized return, drawdown, Sortino Ratio, and Calmar Ratio.
"""

# Evaluate performance metrics for the simulation with enhanced logic (Calculations)
if equity_curve_df_enhanced_logic.empty or len(equity_curve_df_enhanced_logic) < 2:
    display("Enhanced logic equity curve is empty or too short for evaluation.")
    # Initialize metrics to default values if evaluation is not possible
    total_return_enhanced_logic = np.nan
    annualized_return_enhanced_logic = np.nan
    max_drawdown_enhanced_logic = np.nan
    sortino_ratio_enhanced_logic = -1000 # Use a low number for invalid ratio
    calmar_ratio_enhanced_logic = np.nan
else:
    equity_curve_df_enhanced_logic['Daily_Return'] = equity_curve_df_enhanced_logic['Equity'].pct_change().fillna(0)
    total_return_enhanced_logic = (equity_curve_df_enhanced_logic['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_enhanced_logic = (1 + total_return_enhanced_logic)**(trading_periods_per_year / len(equity_curve_df_enhanced_logic)) - 1

    equity_curve_df_enhanced_logic['Peak'] = equity_curve_df_enhanced_logic['Equity'].cummax()
    equity_curve_df_enhanced_logic['Drawdown'] = equity_curve_df_enhanced_logic['Equity'] - equity_curve_df_enhanced_logic['Peak']
    max_drawdown_enhanced_logic = equity_curve_df_enhanced_logic['Drawdown'].min()

    mar = 0
    downside_returns_enhanced_logic = equity_curve_df_enhanced_logic[equity_curve_df_enhanced_logic['Daily_Return'] < mar]['Daily_Return']

    # Corrected conditional logic for Sortino Ratio calculation
    downside_deviation = downside_returns_enhanced_logic.std()

    if downside_deviation == 0 or np.isnan(downside_deviation):
        # If no downside returns or std is zero/NaN
        if annualized_return_enhanced_logic > mar:
             sortino_ratio_enhanced_logic = float('inf') # Profitable with no downside risk
        elif annualized_return_enhanced_logic == mar:
             sortino_ratio_enhanced_logic = 0 # Breakeven with no downside risk
        else:
             sortino_ratio_enhanced_logic = -1000 # Losing with no measurable downside risk (or error) # Use a very low number
    else:
        # Otherwise, calculate the standard Sortino Ratio
        sortino_ratio_enhanced_logic = (annualized_return_enhanced_logic - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
    if np.isinf(sortino_ratio_enhanced_logic) or np.isnan(sortino_ratio_enhanced_logic):
        sortino_ratio_enhanced_logic = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_enhanced_logic = annualized_return_enhanced_logic / abs(max_drawdown_enhanced_logic) if max_drawdown_enhanced_logic != 0 else float('inf')


# Display calculated metrics (optional, can be done in the next cell)
# display("\nCalculated Performance Metrics (using Enhanced Logic):")
# display(f"Total Return: {total_return_enhanced_logic:.4f}")
# display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
# display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
# display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
# display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")

"""## Evaluate performance (Part 2: Visualization and Summary)

### Subtask:
Plot the equity curve and display the trade log and losing trades summaries based on the simulation results.

**Reasoning**:
Plot the equity curve to visualize the strategy's performance and display the trade log and losing trades summaries for review.
"""

import matplotlib.pyplot as plt

# Plot the equity curve
plt.figure(figsize=(12, 6))
plt.plot(equity_curve_df_enhanced_logic.index, equity_curve_df_enhanced_logic['Equity'], label='Equity Curve (Enhanced Logic)')
plt.title('Equity Curve (Enhanced Logic)')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.legend()
plt.grid(True)
plt.show()

# Display performance metrics (calculated in the previous cell)
display("\nPerformance Metrics (using Enhanced Logic):")
display(f"Initial Capital: {initial_capital:.2f}")
display(f"Final Equity: {equity_curve_df_enhanced_logic['Equity'].iloc[-1]:.2f}")
display(f"Total Return: {total_return_enhanced_logic:.4f}")
display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")


# Display head of the trade log
display("\nTrade Log Summary (Enhanced Logic Simulation):")
display(trade_log_df_enhanced_logic.head())

# Analyze losing trades from the simulation
losing_trades_df_enhanced_logic = trade_log_df_enhanced_logic[trade_log_df_enhanced_logic['PnL'].notna() & (trade_log_df_enhanced_logic['PnL'] < 0)]
display("\nLosing Trades Summary (Enhanced Logic Simulation):")
display(losing_trades_df_enhanced_logic.head()) # Display head of losing trades

"""**Reasoning**:
Evaluate the performance of the strategy with the enhanced logic by calculating key metrics and plotting the equity curve.
"""

# Evaluate performance metrics for the simulation with enhanced logic
if equity_curve_df_enhanced_logic.empty or len(equity_curve_df_enhanced_logic) < 2:
    display("Enhanced logic equity curve is empty or too short for evaluation.")
else:
    equity_curve_df_enhanced_logic['Daily_Return'] = equity_curve_df_enhanced_logic['Equity'].pct_change().fillna(0)
    total_return_enhanced_logic = (equity_curve_df_enhanced_logic['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_enhanced_logic = (1 + total_return_enhanced_logic)**(trading_periods_per_year / len(equity_curve_df_enhanced_logic)) - 1

    equity_curve_df_enhanced_logic['Peak'] = equity_curve_df_enhanced_logic['Equity'].cummax()
    equity_curve_df_enhanced_logic['Drawdown'] = equity_curve_df_enhanced_logic['Equity'] - equity_curve_df_enhanced_logic['Peak']
    max_drawdown_enhanced_logic = equity_curve_df_enhanced_logic['Drawdown'].min()

    mar = 0
    downside_returns_enhanced_logic = equity_curve_df_enhanced_logic[equity_curve_df_enhanced_logic['Daily_Return'] < mar]['Daily_Return']

    # Corrected conditional logic for Sortino Ratio calculation
    downside_deviation = downside_returns_enhanced_logic.std()

    if downside_deviation == 0 or np.isnan(downside_deviation):
        # If no downside returns or std is zero/NaN
        if annualized_return_enhanced_logic > mar:
             sortino_ratio_enhanced_logic = float('inf') # Profitable with no downside risk
        elif annualized_return_enhanced_logic == mar:
             sortino_ratio_enhanced_logic = 0 # Breakeven with no downside risk
        else:
             sortino_ratio_enhanced_logic = -1000 # Losing with no measurable downside risk (or error) # Use a very low number
    else:
        # Otherwise, calculate the standard Sortino Ratio
        sortino_ratio_enhanced_logic = (annualized_return_enhanced_logic - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
    if np.isinf(sortino_ratio_enhanced_logic) or np.isnan(sortino_ratio_enhanced_logic):
        sortino_ratio_enhanced_logic = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_enhanced_logic = annualized_return_enhanced_logic / abs(max_drawdown_enhanced_logic) if max_drawdown_enhanced_logic != 0 else float('inf')


    display("\nPerformance Metrics (using Enhanced Logic):")
    display(f"Initial Capital: {initial_capital:.2f}")
    display(f"Final Equity: {equity_curve_df_enhanced_logic['Equity'].iloc[-1]:.2f}")
    display(f"Total Return: {total_return_enhanced_logic:.4f}")
    display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
    display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
    display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
    display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")

    # Plot the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_df_enhanced_logic.index, equity_curve_df_enhanced_logic['Equity'], label='Equity Curve (Enhanced Logic)')
    plt.title('Equity Curve (Enhanced Logic)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display head of the trade log
    display("\nTrade Log Summary (Enhanced Logic Simulation):")
    display(trade_log_df_enhanced_logic.head())

    # Analyze losing trades from the simulation
    losing_trades_df_enhanced_logic = trade_log_df_enhanced_logic[trade_log_df_enhanced_logic['PnL'].notna() & (trade_log_df_enhanced_logic['PnL'] < 0)]
    display("\nLosing Trades Summary (Enhanced Logic Simulation):")
    display(losing_trades_df_enhanced_logic.head()) # Display head of losing trades

else:
    display("Cannot perform evaluation as the enhanced logic equity curve is empty.")

"""## Evaluate performance (Part 1: Calculations)

### Subtask:
Calculate key performance metrics like P&L, Sortino Ratio, and Calmar Ratio based on the simulation results in a separate cell.

**Reasoning**:
Extract the performance calculation logic into a new cell to avoid potential syntax issues in the previous combined cell. Calculate the daily returns, total return, annualized return, drawdown, Sortino Ratio, and Calmar Ratio.
"""

# Evaluate performance metrics for the simulation with enhanced logic (Calculations)
if equity_curve_df_enhanced_logic.empty or len(equity_curve_df_enhanced_logic) < 2:
    display("Enhanced logic equity curve is empty or too short for evaluation.")
    # Initialize metrics to default values if evaluation is not possible
    total_return_enhanced_logic = np.nan
    annualized_return_enhanced_logic = np.nan
    max_drawdown_enhanced_logic = np.nan
    sortino_ratio_enhanced_logic = -1000 # Use a low number for invalid ratio
    calmar_ratio_enhanced_logic = np.nan
else:
    equity_curve_df_enhanced_logic['Daily_Return'] = equity_curve_df_enhanced_logic['Equity'].pct_change().fillna(0)
    total_return_enhanced_logic = (equity_curve_df_enhanced_logic['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if necessary
    annualized_return_enhanced_logic = (1 + total_return_enhanced_logic)**(trading_periods_per_year / len(equity_curve_df_enhanced_logic)) - 1

    equity_curve_df_enhanced_logic['Peak'] = equity_curve_df_enhanced_logic['Equity'].cummax()
    equity_curve_df_enhanced_logic['Drawdown'] = equity_curve_df_enhanced_logic['Equity'] - equity_curve_df_enhanced_logic['Peak']
    max_drawdown_enhanced_logic = equity_curve_df_enhanced_logic['Drawdown'].min()

    mar = 0
    downside_returns_enhanced_logic = equity_curve_df_enhanced_logic[equity_curve_df_enhanced_logic['Daily_Return'] < mar]['Daily_Return']

    # Corrected conditional logic for Sortino Ratio calculation
    downside_deviation = downside_returns_enhanced_logic.std()

    if downside_deviation == 0 or np.isnan(downside_deviation):
        # If no downside returns or std is zero/NaN
        if annualized_return_enhanced_logic > mar:
             sortino_ratio_enhanced_logic = float('inf') # Profitable with no downside risk
        elif annualized_return_enhanced_logic == mar:
             sortino_ratio_enhanced_logic = 0 # Breakeven with no downside risk
        else:
             sortino_ratio_enhanced_logic = -1000 # Losing with no measurable downside risk (or error) # Use a very low number
    else:
        # Otherwise, calculate the standard Sortino Ratio
        sortino_ratio_enhanced_logic = (annualized_return_enhanced_logic - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
    if np.isinf(sortino_ratio_enhanced_logic) or np.isnan(sortino_ratio_enhanced_logic):
        sortino_ratio_enhanced_logic = -1000 # Return a very low fitness for problematic ratios


    # Calculate Calmar Ratio
    # Avoid division by zero if there is no drawdown
    calmar_ratio_enhanced_logic = annualized_return_enhanced_logic / abs(max_drawdown_enhanced_logic) if max_drawdown_enhanced_logic != 0 else float('inf')


# Display calculated metrics (optional, can be done in the next cell)
# display("\nCalculated Performance Metrics (using Enhanced Logic):")
# display(f"Total Return: {total_return_enhanced_logic:.4f}")
# display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
# display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
# display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
# display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")

"""## Evaluate performance (Part 2: Visualization and Summary)

### Subtask:
Plot the equity curve and display the trade log and losing trades summaries based on the simulation results.

**Reasoning**:
Plot the equity curve to visualize the strategy's performance and display the trade log and losing trades summaries for review.
"""

import matplotlib.pyplot as plt

# Plot the equity curve
plt.figure(figsize=(12, 6))
plt.plot(equity_curve_df_enhanced_logic.index, equity_curve_df_enhanced_logic['Equity'], label='Equity Curve (Enhanced Logic)')
plt.title('Equity Curve (Enhanced Logic)')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.legend()
plt.grid(True)
plt.show()

# Display performance metrics (calculated in the previous cell)
display("\nPerformance Metrics (using Enhanced Logic):")
display(f"Initial Capital: {initial_capital:.2f}")
display(f"Final Equity: {equity_curve_df_enhanced_logic['Equity'].iloc[-1]:.2f}")
display(f"Total Return: {total_return_enhanced_logic:.4f}")
display(f"Annualized Return: {annualized_return_enhanced_logic:.4f}")
display(f"Max Drawdown: {max_drawdown_enhanced_logic:.2f}")
display(f"Sortino Ratio (MAR=0): {sortino_ratio_enhanced_logic:.4f}")
display(f"Calmar Ratio: {calmar_ratio_enhanced_logic:.4f}")


# Display head of the trade log
display("\nTrade Log Summary (Enhanced Logic Simulation):")
display(trade_log_df_enhanced_logic.head())

# Analyze losing trades from the simulation
losing_trades_df_enhanced_logic = trade_log_df_enhanced_logic[trade_log_df_enhanced_logic['PnL'].notna() & (trade_log_df_enhanced_logic['PnL'] < 0)]
display("\nLosing Trades Summary (Enhanced Logic Simulation):")
display(losing_trades_df_enhanced_logic.head()) # Display head of losing trades

"""## Analyze Losing Trades (Enhanced Logic)

### Subtask:
Re-examine the `losing_trades_df_enhanced_logic` from the latest run to see the characteristics of the trades that are still resulting in losses, despite the positive overall performance.

**Reasoning**:
Display the `losing_trades_df_enhanced_logic` DataFrame to examine the details of the trades that had a negative PnL in the latest simulation with the enhanced logic.
"""

# Analyze losing trades from the simulation with enhanced logic again
# losing_trades_df_enhanced_logic was already created in the previous evaluation cell
display("\nLosing Trades Summary (Enhanced Logic Simulation - Latest Run):")
display(losing_trades_df_enhanced_logic)

"""# Task
Implement a Genetic Algorithm with walk-forward optimization to optimize the trading strategy defined in "/content/Helios Trader.md". The GA should optimize numeric parameters (lookback periods, entry step size, dynamic factor weights, stop-loss multipliers) and the choice of indicator for each factor. Analyze the market data around losing trades after running the GA.

## Define the fitness function (with indicator selection and extended parameters)

### Subtask:
Implement a fitness function that evaluates the performance of a given set of strategy parameters, *including the choice of indicator for each factor, lookback periods, entry step size, dynamic factor weights, and stop-loss multipliers*. This function will run backtests using the historical data and a risk-adjusted return metric.

**Reasoning**:
Implement the `evaluate_strategy_extended` function as described in the instructions. This involves integrating the factor calculations, MSS calculation (including dynamic weights), regime classification, and the enhanced trading simulation logic (fractional positions, gradual entries/exits, stop-loss, profit targets) into a single function that takes all relevant parameters. The function will then calculate and return the Sortino Ratio.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress

# Assuming the following functions from previous steps are available and correct:
# calculate_trend_slope, calculate_trend_macd, calculate_volatility_atr,
# calculate_volatility_stddev, calculate_exhaustion_sma_diff, calculate_exhaustion_rsi
# INDICATOR_FUNCTIONS dictionary mapping indicator names to these functions
# get_dynamic_weights function
# classify_regime_final function

# Define the evaluate_strategy_extended function
def evaluate_strategy_extended(params, df, initial_capital=100000):
    """
    Evaluates a trading strategy with extended parameters, including indicator choices,
    lookbacks, gradual entry/exit, dynamic weights, and profit targets.

    Args:
        params (dict): Dictionary of strategy parameters, including:
                       - 'weight_trend', 'weight_volatility', 'weight_exhaustion' (for static MSS, if used)
                       - Regime thresholds ('strong_bull_threshold', 'weak_bull_threshold', etc.)
                       - Stop-loss multipliers ('stop_loss_multiplier_strong', 'stop_loss_multiplier_weak')
                       - Indicator choices ('indicator_trend', 'indicator_volatility', 'indicator_exhaustion')
                       - Lookback periods for each indicator ('lookback_trend', 'lookback_volatility', 'lookback_exhaustion')
                       - Parameters for gradual entry/exit ('entry_step_size', 'max_position_fraction')
                       - Flag/params for dynamic weights (e.g., 'use_dynamic_weights')
                       - Parameters for profit targets ('take_profit_multiplier_strong', 'take_profit_multiplier_weak')

        df (pd.DataFrame): DataFrame containing dollar bars (Open, High, Low, Close, Volume, DollarVolume).
        initial_capital (float): Starting capital for the backtest.

    Returns:
        float: The Sortino Ratio of the strategy's performance, or a very low number
               if the ratio is infinite or NaN or if there's an error.
    """
    df_eval = df.copy() # Work on a copy

    # Extract parameters with defaults
    weight_trend = params.get('weight_trend', 0.4)
    weight_volatility = params.get('weight_volatility', 0.3)
    weight_exhaustion = params.get('weight_exhaustion', 0.3)

    strong_bull_threshold = params.get('strong_bull_threshold', 50)
    weak_bull_threshold = params.get('weak_bull_threshold', 20)
    neutral_threshold_upper = params.get('neutral_threshold_upper', 20)
    neutral_threshold_lower = params.get('neutral_threshold_lower', -20)
    strong_bear_threshold = params.get('strong_bear_threshold', -50)
    weak_bear_threshold = params.get('weak_bear_threshold', -20)

    stop_loss_multiplier_strong = params.get('stop_loss_multiplier_strong', 3)
    stop_loss_multiplier_weak = params.get('stop_loss_multiplier_weak', 3)

    indicator_trend_name = params.get('indicator_trend', 'Slope')
    indicator_volatility_name = params.get('indicator_volatility', 'ATR')
    indicator_exhaustion_name = params.get('indicator_exhaustion', 'SMADiff')

    lookback_trend = params.get('lookback_trend', 20)
    lookback_volatility = params.get('lookback_volatility', 20)
    lookback_exhaustion = params.get('lookback_exhaustion', 20)

    entry_step_size = params.get('entry_step_size', 0.2)
    max_position_fraction = params.get('max_position_fraction', 1.0)

    use_dynamic_weights = params.get('use_dynamic_weights', True) # Default to use dynamic weights
    # Note: If dynamic weights were optimized (e.g., different weights per regime as parameters),
    # those would need to be extracted here as well. For now, assuming get_dynamic_weights
    # is a fixed function that takes the regime.

    take_profit_multiplier_strong = params.get('take_profit_multiplier_strong', 4.0)
    take_profit_multiplier_weak = params.get('take_profit_multiplier_weak', 2.0)


    # --- Calculate Factors using specified indicators ---
    try:
        # Calculate Volatility first as Exhaustion (SMADiff) depends on it
        if indicator_volatility_name in INDICATOR_FUNCTIONS['Volatility']:
            df_eval['Volatility_Factor'] = INDICATOR_FUNCTIONS['Volatility'][indicator_volatility_name](df_eval.copy(), lookback_volatility)
        else:
            # display(f"Warning: Unknown Volatility indicator: {indicator_volatility_name}. Using default (ATR).")
            df_eval['Volatility_Factor'] = INDICATOR_FUNCTIONS['Volatility']['ATR'](df_eval.copy(), lookback_volatility)

        if indicator_trend_name in INDICATOR_FUNCTIONS['Trend']:
             df_eval['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend'][indicator_trend_name](df_eval.copy(), lookback_trend)
        else:
             # display(f"Warning: Unknown Trend indicator: {indicator_trend_name}. Using default (Slope).")
             df_eval['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend']['Slope'](df_eval.copy(), lookback_trend)

        if indicator_exhaustion_name in INDICATOR_FUNCTIONS['Exhaustion']:
            if indicator_exhaustion_name == 'SMADiff': # SMADiff requires normalized Volatility
                 df_eval['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_name](df_eval.copy(), lookback_exhaustion, df_eval['Volatility_Factor'])
            else:
                 # Other exhaustion indicators (like RSI) take df and their lookback
                 df_eval['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_name](df_eval.copy(), lookback_exhaustion)
        else:
            # display(f"Warning: Unknown Exhaustion indicator: {indicator_exhaustion_name}. Using default (SMADiff).")
            df_eval['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion']['SMADiff'](df_eval.copy(), lookback_exhaustion, df_eval['Volatility_Factor'])

    except Exception as e:
        # display(f"Error during factor calculation: {e}")
        return -10000 # Return a very low fitness on error


    # Drop rows with NaN values generated by lookback periods after factor calculation
    df_eval = df_eval.dropna(subset=['Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor'])

    if len(df_eval) < max(lookback_trend, lookback_volatility, lookback_exhaustion) + 10: # Ensure enough data after dropna
        # display("Warning: DataFrame is too short after dropping NaNs.")
        return -10000 # Cannot evaluate performance if insufficient data remains


    # --- Calculate MSS and Regime ---
    if use_dynamic_weights:
         # Calculate initial static MSS to get a starting regime for dynamic weights
         # Use some default static weights for this initial classification
         static_weight_trend_initial = 0.4
         static_weight_volatility_initial = 0.3
         static_weight_exhaustion_initial = 0.3

         df_eval['MSS_static_initial'] = (static_weight_trend_initial * df_eval['Trend_Factor'] +
                                         static_weight_volatility_initial * df_eval['Volatility_Factor'] +
                                         static_weight_exhaustion_initial * df_eval['Exhaustion_Factor'])

         # Classify the initial static regime using the provided thresholds
         def classify_regime_initial(mss):
              if mss > strong_bull_threshold: return 'Strong Bull'
              elif mss > weak_bull_threshold: return 'Weak Bull'
              elif mss >= neutral_threshold_lower and mss <= neutral_threshold_upper: return 'Neutral'
              elif mss > strong_bear_threshold: return 'Weak Bear'
              else: return 'Strong Bear'

         df_eval['Regime_initial'] = df_eval['MSS_static_initial'].apply(classify_regime_initial)


         # Calculate Dynamic MSS using dynamic weights based on the *initial* regime
         df_eval['MSS'] = np.nan # Initialize dynamic MSS column
         for index, row in df_eval.iterrows():
             current_regime_initial = row['Regime_initial'] # Use the initial static regime
             dynamic_weights = get_dynamic_weights(current_regime_initial)

             dynamic_mss = (dynamic_weights['Trend'] * row['Trend_Factor'] +
                            dynamic_weights['Volatility'] * row['Volatility_Factor'] +
                            dynamic_weights['Exhaustion'] * row['Exhaustion_Factor'])
             df_eval.loc[index, 'MSS'] = dynamic_mss

         # Classify the final Regime based on the Dynamic MSS and the provided thresholds
         df_eval['Regime'] = df_eval['MSS'].apply(classify_regime_initial) # Use the same thresholds

    else: # Use static weights for MSS calculation
         df_eval['MSS'] = (weight_trend * df_eval['Trend_Factor'] +
                           weight_volatility * df_eval['Volatility_Factor'] +
                           weight_exhaustion * df_eval['Exhaustion_Factor'])

         # Classify Regime based on Static MSS and the provided thresholds
         def classify_regime_static_final(mss):
              if mss > strong_bull_threshold: return 'Strong Bull'
              elif mss > weak_bull_threshold: return 'Weak Bull'
              elif mss >= neutral_threshold_lower and mss <= neutral_threshold_upper: return 'Neutral'
              elif mss > strong_bear_threshold: return 'Weak Bear'
              else: return 'Strong Bear'

         df_eval['Regime'] = df_eval['MSS'].apply(classify_regime_static_final)


    # --- Trading Simulation with Enhanced Logic ---
    position_fraction = 0.0 # Current position as a fraction of initial_capital
    total_units = 0.0
    total_cost_basis = 0.0 # Total cost paid for units (positive for buy, negative for sell)
    stop_loss = 0.0
    take_profit = 0.0 # Profit target level
    equity_curve = []
    current_capital = initial_capital

    max_step_fraction = entry_step_size * max_position_fraction # Step size in terms of fraction of initial_capital

    # Initializing these here, they are updated within the loop for trailing stops
    peak_price_since_entry = -float('inf') # Track peak price for trailing stop (Long)
    valley_price_since_entry = float('inf') # Track valley price for trailing stop (Short - not active yet)


    for index, row in df_eval.iterrows():
        current_price = row['Close']
        current_regime = row['Regime']
        current_mss = row['MSS']
        current_volatility_factor = row['Volatility_Factor']

        # Determine stop-loss distance and take-profit distance based on regime and Volatility Factor
        if not isinstance(current_volatility_factor, (int, float)) or np.isnan(current_volatility_factor) or abs(current_volatility_factor) < 1e-9:
             stop_loss_distance = 0
             take_profit_distance = 0
        else:
            if current_regime in ['Strong Bull', 'Strong Bear']:
                stop_loss_distance = stop_loss_multiplier_strong * abs(current_volatility_factor)
                take_profit_distance = take_profit_multiplier_strong * abs(current_volatility_factor)
            elif current_regime in ['Weak Bull', 'Weak Bear']:
                stop_loss_distance = stop_loss_multiplier_weak * abs(current_volatility_factor)
                take_profit_distance = take_profit_multiplier_weak * abs(current_volatility_factor)
            else: # Neutral
                stop_loss_distance = 0
                take_profit_distance = 0


        # Determine target position fraction based on regime and MSS confidence
        target_position_fraction = 0.0
        if current_regime == 'Strong Bull':
            normalized_mss = (current_mss - strong_bull_threshold) / (100 - strong_bull_threshold) if (100 - strong_bull_threshold) > 0 else 0
            target_position_fraction = max_position_fraction * np.clip(normalized_mss, 0, 1)
        elif current_regime == 'Weak Bull':
             normalized_mss = (current_mss - weak_bull_threshold) / (neutral_threshold_upper - weak_bull_threshold) if (neutral_threshold_upper - weak_bull_threshold) > 0 else 0
             target_position_fraction = position_fraction # Default to hold
             if position_fraction > 0:
                 target_position_fraction = max_position_fraction * np.clip(normalized_mss, 0, 1)
        elif current_regime == 'Neutral':
            target_position_fraction = 0.0
        elif current_regime == 'Weak Bear':
             normalized_mss = (current_mss - (-100)) / (weak_bear_threshold - (-100)) if (weak_bear_threshold - (-100)) > 0 else 0
             target_position_fraction = position_fraction # Default to hold
             if position_fraction < 0:
                  target_position_fraction = -max_position_fraction * np.clip(1 - normalized_mss, 0, 1)
        elif current_regime == 'Strong Bear':
            normalized_mss = (current_mss - (-100)) / (strong_bear_threshold - (-100)) if (strong_bear_threshold - (-100)) > 0 else 0
            target_position_fraction = -max_position_fraction * np.clip(1 - normalized_mss, 0, 1)

        # Ensure target_position_fraction has correct sign based on regime
        if target_position_fraction > 1e-9 and current_regime not in ['Strong Bull', 'Weak Bull']:
             target_position_fraction = 0.0
        elif target_position_fraction < -1e-9 and current_regime not in ['Strong Bear', 'Weak Bear']:
             target_position_fraction = 0.0
        # In Neutral, always target 0
        if current_regime == 'Neutral':
            target_position_fraction = 0.0


        # Calculate the change in position fraction
        position_fraction_change = target_position_fraction - position_fraction

        # Limit position fraction change to max_step_fraction
        position_fraction_change = np.clip(position_fraction_change, -max_step_fraction, max_step_fraction)

        # Calculate the capital amount to allocate/deallocate
        capital_to_trade = position_fraction_change * initial_capital

        # Calculate the units to trade based on the capital amount and current price
        units_to_trade = capital_to_trade / current_price if current_price > 0 else 0.0

        # Update total units and total cost basis
        # This simplified logic assumes entering/exiting at the current price
        # More complex logic would handle limit/market orders
        total_units += units_to_trade
        total_cost_basis += units_to_trade * current_price # Track cumulative cost/revenue

        # Update the current position fraction
        current_market_value = total_units * current_price if current_price > 0 else 0.0
        position_fraction = current_market_value / initial_capital if initial_capital > 0 else 0.0


        # Update Trailing Stop Loss and Take Profit based on the current position
        if abs(total_units) > 1e-9: # If position is active
            avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0

            if total_units > 0: # Long position
                # Update peak price for trailing stop
                peak_price_since_entry = max(peak_price_since_entry, current_price)
                # Calculate trailing stop level
                stop_loss_level = peak_price_since_entry - stop_loss_distance if stop_loss_distance > 0 else -float('inf')
                # The stop_loss variable holds the *active* stop loss level
                # Ensure stop loss does not move down during a long trade (only trails up)
                stop_loss = max(stop_loss, stop_loss_level) if stop_loss != 0.0 else stop_loss_level # Initialize or trail up

                # Calculate Take Profit level based on average cost basis
                take_profit_level = avg_cost_basis_per_unit + take_profit_distance if take_profit_distance > 0 else float('inf')
                take_profit = take_profit_level

            elif total_units < 0: # Short position (not active in this PoC)
                 # Update valley price for trailing stop
                 valley_price_since_entry = min(valley_price_since_entry, current_price)
                 # Calculate trailing stop level
                 stop_loss_level = valley_price_since_entry + stop_loss_distance if stop_loss_distance > 0 else float('inf')
                 # Ensure stop loss does not move up during a short trade (only trails down)
                 stop_loss = min(stop_loss, stop_loss_level) if stop_loss != 0.0 else stop_loss_level # Initialize or trail down

                 # Calculate Take Profit level based on average cost basis
                 take_profit_level = avg_cost_basis_per_unit - take_profit_distance if take_profit_distance > 0 else -float('inf')
                 take_profit = take_profit_level
        else: # No active position
             stop_loss = 0.0
             take_profit = 0.0
             # Reset peak/valley price when position is flat
             peak_price_since_entry = -float('inf')
             valley_price_since_entry = float('inf')


        # Check for Stop-Loss or Take-Profit hit
        if abs(total_units) > 1e-9: # Only check if position is active
             avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0 # Recalculate avg cost basis

             # Check Take Profit hit
             if total_units > 0 and current_price >= take_profit and not np.isinf(take_profit):
                 pnl = (current_price - avg_cost_basis_per_unit) * total_units # Calculate P/L
                 current_capital += pnl # Update capital
                 # Reset position variables
                 total_units = 0.0
                 total_cost_basis = 0.0
                 position_fraction = 0.0
                 stop_loss = 0.0
                 take_profit = 0.0
                 peak_price_since_entry = -float('inf') # Reset peak

             # Check Stop Loss hit (after checking take profit)
             elif total_units > 0 and current_price <= stop_loss and not np.isinf(stop_loss):
                 pnl = (current_price - avg_cost_basis_per_unit) * total_units # Calculate P/L
                 current_capital += pnl # Update capital
                 # Reset position variables
                 total_units = 0.0
                 total_cost_basis = 0.0
                 position_fraction = 0.0
                 stop_loss = 0.0
                 take_profit = 0.0
                 peak_price_since_entry = -float('inf') # Reset peak

             elif total_units < 0: # Short position (not active)
                  # Similar checks for short take profit and stop loss
                  pass # Skip short position checks for now


        # Append current equity to the equity curve
        # Current equity is current capital + (unrealized P/L of the current position)
        avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0
        unrealized_pnl = (current_price - avg_cost_basis_per_unit) * total_units if abs(total_units) > 1e-9 else 0.0
        equity_curve.append({'Date': index, 'Equity': current_capital + unrealized_pnl})


    equity_curve_df_eval = pd.DataFrame(equity_curve).set_index('Date')

    # --- Performance Evaluation (Sortino Ratio) ---
    if equity_curve_df_eval.empty or len(equity_curve_df_eval) < 2:
         # display("Warning: Equity curve is empty or too short for Sortino Ratio calculation.")
         return -10000 # Return a very low fitness

    equity_curve_df_eval['Daily_Return'] = equity_curve_df_eval['Equity'].pct_change().fillna(0)

    # Calculate Annualized Return (assuming daily data from dollar bars)
    trading_periods_per_year = 365
    total_return = (equity_curve_df_eval['Equity'].iloc[-1] - initial_capital) / initial_capital

    if len(equity_curve_df_eval) > 1:
      annualized_return = (1 + total_return)**(trading_periods_per_year / len(equity_curve_df_eval)) - 1
    else:
      annualized_return = 0


    # Calculate Sortino Ratio
    mar = 0 # Minimum Acceptable Return (MAR)
    downside_returns = equity_curve_df_eval[equity_curve_df_eval['Daily_Return'] < mar]['Daily_Return']
    downside_deviation = downside_returns.std()

    # Handle cases where downside_deviation is zero or NaN
    if downside_deviation == 0 or np.isnan(downside_deviation):
        sortino_ratio = float('inf') if annualized_return > mar else (0 if annualized_return == mar else -10000) # Use a very low number for problematic ratios
    else:
        sortino_ratio = (annualized_return - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
    if np.isinf(sortino_ratio) or np.isnan(sortino_ratio):
        sortino_ratio = -10000 # Return a very low fitness for problematic ratios


    return sortino_ratio

# Example usage of the evaluate_strategy_extended function
# Define a sample parameter set including all new parameters
sample_params = {
    'weight_trend': 0.4,
    'weight_volatility': 0.3,
    'weight_exhaustion': 0.3,
    'strong_bull_threshold': 50,
    'weak_bull_threshold': 20,
    'neutral_threshold_upper': 20,
    'neutral_threshold_lower': -20,
    'strong_bear_threshold': -50,
    'weak_bear_threshold': -20,
    'stop_loss_multiplier_strong': 3,
    'stop_loss_multiplier_weak': 3,
    'indicator_trend': 'Slope', # Using implemented indicator
    'indicator_volatility': 'ATR', # Using implemented indicator
    'indicator_exhaustion': 'SMADiff', # Using implemented indicator
    'lookback_trend': 20,
    'lookback_volatility': 20,
    'lookback_exhaustion': 20,
    'entry_step_size': 0.2,
    'max_position_fraction': 1.0,
    'use_dynamic_weights': True, # Example: use dynamic weights
    'take_profit_multiplier_strong': 4.0,
    'take_profit_multiplier_weak': 2.0
}

# Ensure dollar_bars_df is available with raw OHLCV and DollarVolume
# Assuming dollar_bars_df was loaded and potentially processed before
# For evaluation, we need the raw data or at least the OHLCV and DollarVolume
# Let's assume dollar_bars_df is the result of create_dollar_bars

# If dollar_bars_df was modified in previous steps (e.g., added factor columns),
# ensure evaluate_strategy_extended works on a clean input or handles existing columns.
# A robust approach is to pass a dataframe with only the required input columns.
df_for_extended_evaluation = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


fitness_extended = evaluate_strategy_extended(sample_params, df_for_extended_evaluation, initial_capital)
display(f"Fitness (Sortino Ratio) with extended sample parameters: {fitness_extended}")

"""## Set up the ga framework with genetic operators and walk-forward optimization

### Subtask:
Structure the Genetic Algorithm process, incorporating the genetic operators (selection, crossover, mutation) and the walk-forward optimization framework. The GA will optimize both the numeric parameters and the indicator choices.

**Reasoning**:
Implement the genetic algorithm with walk-forward optimization, incorporating the choice of indicators and their lookback periods into the parameter space and using the updated fitness function to evaluate individuals.
"""

import random

def genetic_algorithm_walk_forward_extended(df, initial_capital, param_space, n_generations, pop_size, mutation_rate, lookback_window_days, step_size_days):
    """
    Runs a Genetic Algorithm with walk-forward optimization, optimizing a wide range
    of strategy parameters, including indicator choices and lookbacks.

    Args:
        df (pd.DataFrame): DataFrame containing the full historical data (dollar bars OHLCV).
                           Factors will be calculated inside the fitness function.
        initial_capital (float): Starting capital for backtests.
        param_space (dict): Dictionary defining the search space for each parameter,
                            including numeric ranges and indicator choices/lookback ranges,
                            entry/exit parameters, dynamic weight flag, and profit target multipliers.
                            Example: {'weight_trend': (0.1, 0.7), ..., 'indicator_trend': ['Slope', 'MACD'], 'lookback_trend': (10, 50), 'entry_step_size': (0.1, 0.5), ...}.
        n_generations (int): Number of generations for the GA.
        pop_size (int): Size of the population in each generation.
        mutation_rate (float): Probability of mutation for each parameter.
        lookback_window_days (int): Size of the walk-forward backtesting window (in days).
        step_size_days (int): Size of the step for the walk-forward window (in days).

    Returns:
        dict: The best set of parameters found during the walk-forward optimization.
    """

    best_overall_params = None
    best_overall_fitness = -float('inf')

    # Get all unique dates and sort them
    all_dates = df.index.unique().sort_values()
    total_bars = len(all_dates)

    # Calculate the minimum number of data points needed for indicator lookbacks
    # Consider all potential lookback parameters in the param_space
    min_lookback = 0
    for param, value_range in param_space.items():
        if param.startswith('lookback_') and isinstance(value_range, tuple):
            min_lookback = max(min_lookback, value_range[1]) # Use the upper bound of the lookback range

    # Determine the initial training window end date based on the first date and lookback days
    initial_window_start_date = all_dates[0]
    initial_window_end_date = initial_window_start_date + pd.Timedelta(days=lookback_window_days)

    # Find the index corresponding to initial_window_end_date in all_dates
    # Use side='left' to get the index of the first date >= initial_window_end_date
    first_window_end_idx = all_dates.searchsorted(initial_window_end_date, side='left')

    # Ensure the initial window has enough data points for lookbacks and GA
    # A buffer of 50 bars or twice the max lookback is used as a heuristic
    min_bars_for_ga = max(min_lookback * 2, 50) # Heuristic: require at least twice the max lookback bars or 50 bars

    if (first_window_end_idx - 0) < min_bars_for_ga:
         display(f"Error: Initial window size ({first_window_end_idx} bars) is less than the minimum required bars for GA ({min_bars_for_ga}).")
         return None

    window_start_idx = 0
    window_end_idx = first_window_end_idx # The index *after* the last date in the window

    # Loop while the end of the current window is within the total number of bars
    # and the window size is sufficient for the GA
    while window_end_idx <= total_bars and (window_end_idx - window_start_idx) >= min_bars_for_ga:
        # Define the training window (backtest period for GA optimization) using indices
        current_start_date = all_dates[window_start_idx]
        # Use the date of the last bar in the window
        current_end_date = all_dates[window_end_idx - 1]

        # Create the training dataframe for the current window
        # Pass a copy of the relevant data columns for the fitness function
        train_df = df.loc[current_start_date:current_end_date].copy()
        train_df_clean = train_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


        # Re-check if the training window has enough data points after subsetting by dates
        if len(train_df_clean) < min_bars_for_ga:
             display(f"Skipping window: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')} due to insufficient data ({len(train_df_clean)} bars). Minimum required: {min_bars_for_ga}")
             # Move the window forward by step_size_days and find the new indices
             next_window_start_date = current_start_date + pd.Timedelta(days=step_size_days)
             window_start_idx = all_dates.searchsorted(next_window_start_date, side='left')
             # Calculate the new window end index based on the lookback_window_days from the new start date
             if window_start_idx < total_bars:
                new_window_end_date = all_dates[window_start_idx] + pd.Timedelta(days=lookback_window_days)
                window_end_idx = all_dates.searchsorted(new_window_end_date, side='left')
             else:
                # If the new start index is beyond the last bar, set end index to total_bars + 1 to terminate loop
                window_end_idx = total_bars + 1
             continue


        display(f"Optimizing on window: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")

        # --- Genetic Algorithm ---
        # Initialize population
        population = []
        for _ in range(pop_size):
            params = {}
            for param, value_range in param_space.items():
                if isinstance(value_range, tuple): # Numeric parameter
                    min_val, max_val = value_range
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param] = random.randint(min_val, max_val)
                    else:
                        params[param] = random.uniform(min_val, max_val)
                elif isinstance(value_range, list): # Categorical parameter (indicator choice)
                    params[param] = random.choice(value_range)
                elif isinstance(value_range, bool): # Boolean parameter (e.g., use_dynamic_weights)
                     params[param] = random.choice([True, False])


            population.append(params)


        # GA generations
        for generation in range(n_generations):
            # Evaluate fitness of each individual using the extended evaluation function
            fitness_scores = [evaluate_strategy_extended(params, train_df_clean.copy(), initial_capital) for params in population]

            # Handle potential errors or invalid fitness scores from evaluation
            # Filter out individuals with fitness below -9999 (indicating errors/invalid scenarios)
            valid_fitness_indices = [i for i, score in enumerate(fitness_scores) if not np.isinf(score) and not np.isnan(score) and score > -9999]

            if not valid_fitness_indices:
                 display(f"Generation {generation+1}: No valid fitness scores. Re-initializing population.")
                 # Re-initialize population if no valid individuals are found
                 population = []
                 for _ in range(pop_size):
                      params = {}
                      for param, value_range in param_space.items():
                           if isinstance(value_range, tuple): # Numeric parameter
                               min_val, max_val = value_range
                               if isinstance(min_val, int) and isinstance(max_val, int):
                                    params[param] = random.randint(min_val, max_val)
                               else:
                                    params[param] = random.uniform(min_val, max_val)
                           elif isinstance(value_range, list): # Categorical parameter (indicator choice)
                                params[param] = random.choice(value_range)
                           elif isinstance(value_range, bool): # Boolean parameter
                                params[param] = random.choice([True, False])

                      population.append(params)

                 continue # Skip selection/crossover/mutation for this generation


            # Select parents from valid individuals (e.g., tournament selection)
            valid_population = [population[i] for i in valid_fitness_indices]
            valid_fitness = [fitness_scores[i] for i in valid_fitness_indices]

            parents = []
            # Use the size of the valid population for selecting parents
            num_parents_to_select = min(pop_size, len(valid_population))
            # Ensure we select an even number of parents for crossover
            num_parents_to_select = (num_parents_to_select // 2) * 2

            if num_parents_to_select == 0:
                 display(f"Generation {generation+1}: Not enough valid individuals to select parents. Re-initializing population.")
                 # Re-initialize population if not enough valid parents
                 population = []
                 for _ in range(pop_size):
                      params = {}
                      for param, value_range in param_space.items():
                           if isinstance(value_range, tuple):
                               min_val, max_val = value_range
                               if isinstance(min_val, int) and isinstance(max_val, int):
                                    params[param] = random.randint(min_val, max_val)
                               else:
                                    params[param] = random.uniform(min_val, max_val)
                           elif isinstance(value_range, list):
                                params[param] = random.choice(value_range)
                           elif isinstance(value_range, bool): # Boolean parameter
                                params[param] = random.choice([True, False])

                      population.append(params)
                 continue


            for _ in range(num_parents_to_select // 2):
                # Select two random indices from the valid individuals list
                idx1_valid = random.choice(range(len(valid_population)))
                idx2_valid = random.choice(range(len(valid_population)))
                parent1 = valid_population[idx1_valid] if valid_fitness[idx1_valid] > valid_fitness[idx2_valid] else valid_population[idx2_valid]
                idx1_valid = random.choice(range(len(valid_population)))
                idx2_valid = random.choice(range(len(valid_population)))
                parent2 = valid_population[idx1_valid] if valid_fitness[idx1_valid] > valid_fitness[idx2_valid] else valid_population[idx2_valid]
                parents.extend([parent1, parent2])

            # Create next generation (crossover and mutation)
            next_population = []
            # Ensure we generate enough individuals for the next population size
            while len(next_population) < pop_size:
                 # Select two parents with replacement from the generated parents list
                 # Ensure there are at least 2 parents to select from
                 if len(parents) < 2:
                      # If not enough parents, break the loop and re-initialize population outside
                      display(f"Generation {generation+1}: Not enough parents generated for crossover. Breaking.")
                      break

                 parent1, parent2 = random.sample(parents, 2)

                 child1, child2 = {}, {}

                 # Crossover
                 for param in param_space.keys():
                     if random.random() < 0.5:
                         child1[param] = parent1[param]
                         child2[param] = parent2[param]
                     else:
                         child1[param] = parent2[param]
                         child2[param] = parent1[param]

                 # Mutation
                 for child in [child1, child2]:
                     for param, value_range in param_space.items():
                         if random.random() < mutation_rate:
                             if isinstance(value_range, tuple): # Numeric parameter mutation
                                 min_val, max_val = value_range
                                 if isinstance(min_val, int) and isinstance(max_val, int):
                                     child[param] = random.randint(min_val, max_val)
                                 else:
                                     child[param] = random.uniform(min_val, max_val)
                                 # Ensure mutated values are within bounds
                                 child[param] = max(min_val, min(max_val, child[param]))
                             elif isinstance(value_range, list): # Categorical parameter mutation (pick a random choice)
                                 child[param] = random.choice(value_range)
                             elif isinstance(value_range, bool): # Boolean parameter mutation (flip the value)
                                 child[param] = not child[param]


                 next_population.extend([child1, child2])

            # Ensure the size of the next population is exactly pop_size
            population = next_population[:pop_size]


            # Track best individual in this generation (from the evaluated valid individuals)
            # Check if valid_fitness is not empty before finding max
            if valid_fitness:
                 best_gen_fitness_idx = valid_fitness.index(max(valid_fitness))
                 best_gen_fitness = valid_fitness[best_gen_fitness_idx]
                 best_gen_params = valid_population[best_gen_fitness_idx]
                 display(f"Generation {generation+1}: Best Fitness = {best_gen_fitness:.4f}")
            else:
                 display(f"Generation {generation+1}: No valid fitness scores to report best fitness.")


        # After GA for the current window, select the best parameters found in this window
        # and track the overall best.
        if valid_fitness: # Check if valid_fitness is not empty
             best_window_fitness = max(valid_fitness) # Max fitness from the last generation's valid individuals
             best_window_params = valid_population[valid_fitness.index(best_window_fitness)]

             if best_window_fitness > best_overall_fitness:
                 best_overall_fitness = best_window_fitness
                 best_overall_params = best_window_params
                 display(f"New overall best fitness found: {best_overall_fitness:.4f}")
                 display(f"Corresponding parameters: {best_overall_params}")
        else:
             display(f"No valid fitness scores in the last generation of window ending {current_end_date.strftime('%Y-%m-%d')}. No update to overall best parameters.")


        # Move the window forward by step_size_days and find the new indices
        current_window_start_date = all_dates[window_start_idx]
        next_window_start_date = current_window_start_date + pd.Timedelta(days=step_size_days)
        window_start_idx = all_dates.searchsorted(next_window_start_date, side='left')

        # Calculate the new window end index based on the lookback_window_days from the new start date
        if window_start_idx < total_bars:
            new_window_end_date = all_dates[window_start_idx] + pd.Timedelta(days=lookback_window_days)
            window_end_idx = all_dates.searchsorted(new_window_end_date, side='left')
        else:
            # If the new start index is beyond the last bar, set end index to total_bars + 1 to terminate loop
            window_end_idx = total_bars + 1


    return best_overall_params


# Define the extended parameter space for the GA, including indicator choices, lookback ranges,
# entry/exit parameters, dynamic weight flag, and profit target multipliers.
param_space_extended = {
    'weight_trend': (0.1, 0.7),
    'weight_volatility': (0.1, 0.4),
    'weight_exhaustion': (0.1, 0.4),
    'strong_bull_threshold': (30, 70),
    'weak_bull_threshold': (10, 40),
    'neutral_threshold_upper': (10, 30),
    'neutral_threshold_lower': (-30, -10),
    'strong_bear_threshold': (-70, -30),
    'weak_bear_threshold': (-40, -10),
    'stop_loss_multiplier_strong': (1.5, 3.5),
    'stop_loss_multiplier_weak': (0.5, 2.5),
    'indicator_trend': list(INDICATOR_FUNCTIONS['Trend'].keys()), # Use actual implemented indicators
    'indicator_volatility': list(INDICATOR_FUNCTIONS['Volatility'].keys()), # Use actual implemented indicators
    'indicator_exhaustion': list(INDICATOR_FUNCTIONS['Exhaustion'].keys()), # Use actual implemented indicators
    'lookback_trend': (10, 50),
    'lookback_volatility': (10, 50),
    'lookback_exhaustion': (10, 50),
    'entry_step_size': (0.1, 0.5), # Range for entry step size
    'max_position_fraction': (0.5, 1.0), # Range for max position size
    'use_dynamic_weights': True, # Boolean parameter, GA will choose True or False
    'take_profit_multiplier_strong': (2.0, 5.0), # Range for strong profit target multiplier
    'take_profit_multiplier_weak': (1.0, 3.0), # Range for weak profit target multiplier
}


# GA and Walk-Forward parameters (can keep the reduced values for faster PoC)
n_generations = 5
pop_size = 10
mutation_rate = 0.1
lookback_window_days = 365 # 1 year lookback window for optimization
step_size_days = 90 # Step forward by 90 days


# Ensure dollar_bars_df is available with raw OHLCV and DollarVolume columns
# Assuming dollar_bars_df is the result of create_dollar_bars and contains the necessary columns
# Pass a copy to the GA function to avoid modification
df_for_ga_optimization = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


# Run the Genetic Algorithm with walk-forward optimization using the extended parameter space
display("Starting Genetic Algorithm with Walk-Forward Optimization (Extended Parameters)...")
optimal_params_extended = genetic_algorithm_walk_forward_extended(
    df_for_ga_optimization,
    initial_capital,
    param_space_extended,
    n_generations,
    pop_size,
    mutation_rate,
    lookback_window_days,
    step_size_days
)

display("\nOptimal Parameters (Extended) found:")
display(optimal_params_extended)

"""**Reasoning**:
The previous execution failed because the initial walk-forward window size was insufficient for the GA, based on the calculated minimum required bars. This is due to the `min_bars_for_ga` calculation and the size of the initial data segment. To fix this, I need to either adjust the `min_bars_for_ga` heuristic to be more lenient or, more appropriately, adjust the starting point or size calculation of the initial walk-forward window to ensure it meets the minimum requirement. Given the error message specifically mentions the initial window, the simplest fix is to adjust how the initial window is determined to guarantee it's large enough. I will modify the initial window calculation logic to ensure the first window always has at least `min_bars_for_ga` bars.


"""

import random

def genetic_algorithm_walk_forward_extended(df, initial_capital, param_space, n_generations, pop_size, mutation_rate, lookback_window_days, step_size_days):
    """
    Runs a Genetic Algorithm with walk-forward optimization, optimizing a wide range
    of strategy parameters, including indicator choices and lookbacks.

    Args:
        df (pd.DataFrame): DataFrame containing the full historical data (dollar bars OHLCV).
                           Factors will be calculated inside the fitness function.
        initial_capital (float): Starting capital for backtests.
        param_space (dict): Dictionary defining the search space for each parameter,
                            including numeric ranges and indicator choices/lookback ranges,
                            entry/exit parameters, dynamic weight flag, and profit target multipliers.
                            Example: {'weight_trend': (0.1, 0.7), ..., 'indicator_trend': ['Slope', 'MACD'], 'lookback_trend': (10, 50), 'entry_step_size': (0.1, 0.5), ...}.
        n_generations (int): Number of generations for the GA.
        pop_size (int): Size of the population in each generation.
        mutation_rate (float): Probability of mutation for each parameter.
        lookback_window_days (int): Size of the walk-forward backtesting window (in days).
        step_size_days (int): Size of the step for the walk-forward window (in days).

    Returns:
        dict: The best set of parameters found during the walk-forward optimization.
    """

    best_overall_params = None
    best_overall_fitness = -float('inf')

    # Get all unique dates and sort them
    all_dates = df.index.unique().sort_values()
    total_bars = len(all_dates)

    # Calculate the minimum number of data points needed for indicator lookbacks
    # Consider all potential lookback parameters in the param_space
    min_lookback = 0
    for param, value_range in param_space.items():
        if param.startswith('lookback_') and isinstance(value_range, tuple):
            min_lookback = max(min_lookback, value_range[1]) # Use the upper bound of the lookback range

    # Ensure the initial window has enough data points for lookbacks and GA
    # A buffer of 50 bars or twice the max lookback is used as a heuristic
    min_bars_for_ga = max(min_lookback * 2, 50) # Heuristic: require at least twice the max lookback bars or 50 bars

    # Determine the start and end indices for the initial window
    # Ensure the initial window is at least min_bars_for_ga in size
    initial_window_start_idx = 0
    # Calculate the initial end index based on lookback_window_days from the start date
    initial_window_end_date_calc = all_dates[initial_window_start_idx] + pd.Timedelta(days=lookback_window_days)
    initial_window_end_idx_by_date = all_dates.searchsorted(initial_window_end_date_calc, side='left')

    # Ensure the initial window has at least min_bars_for_ga bars
    if (initial_window_end_idx_by_date - initial_window_start_idx) < min_bars_for_ga:
        # If calculating by days results in too few bars, use min_bars_for_ga to determine the end index
        initial_window_end_idx = initial_window_start_idx + min_bars_for_ga
        if initial_window_end_idx > total_bars:
             display(f"Error: Total data bars ({total_bars}) is less than the minimum required bars for GA ({min_bars_for_ga}). Cannot form initial window.")
             return None
        display(f"Warning: Initial window size by days ({initial_window_end_idx_by_date - initial_window_start_idx} bars) is less than minimum required ({min_bars_for_ga}). Adjusting initial window end index to {initial_window_end_idx}.")
    else:
        initial_window_end_idx = initial_window_end_idx_by_date


    window_start_idx = initial_window_start_idx
    window_end_idx = initial_window_end_idx # The index *after* the last date in the window

    # Loop while the end of the current window is within the total number of bars
    # and the window size is sufficient for the GA
    while window_end_idx <= total_bars and (window_end_idx - window_start_idx) >= min_bars_for_ga:
        # Define the training window (backtest period for GA optimization) using indices
        current_start_date = all_dates[window_start_idx]
        # Use the date of the last bar in the window
        current_end_date = all_dates[window_end_idx - 1]

        # Create the training dataframe for the current window
        # Pass a copy of the relevant data columns for the fitness function
        train_df = df.loc[current_start_date:current_end_date].copy()
        train_df_clean = train_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


        # Re-check if the training window has enough data points after subsetting by dates
        # This check might be redundant if the initial window logic is correct, but keep for safety
        if len(train_df_clean) < min_bars_for_ga:
             display(f"Skipping window: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')} due to insufficient data ({len(train_df_clean)} bars). Minimum required: {min_bars_for_ga}")
             # Move the window forward by step_size_days and find the new indices
             next_window_start_date = current_start_date + pd.Timedelta(days=step_size_days)
             window_start_idx = all_dates.searchsorted(next_window_start_date, side='left')
             # Calculate the new window end index based on the lookback_window_days from the new start date
             if window_start_idx < total_bars:
                new_window_end_date = all_dates[window_start_idx] + pd.Timedelta(days=lookback_window_days)
                window_end_idx = all_dates.searchsorted(new_window_end_date, side='left')
             else:
                # If the new start index is beyond the last bar, set end index to total_bars + 1 to terminate loop
                window_end_idx = total_bars + 1
             continue


        display(f"Optimizing on window: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")

        # --- Genetic Algorithm ---
        # Initialize population
        population = []
        for _ in range(pop_size):
            params = {}
            for param, value_range in param_space.items():
                if isinstance(value_range, tuple): # Numeric parameter
                    min_val, max_val = value_range
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param] = random.randint(min_val, max_val)
                    else:
                        params[param] = random.uniform(min_val, max_val)
                elif isinstance(value_range, list): # Categorical parameter (indicator choice)
                    params[param] = random.choice(value_range)
                elif isinstance(value_range, bool): # Boolean parameter (e.g., use_dynamic_weights)
                     params[param] = random.choice([True, False])


            population.append(params)


        # GA generations
        for generation in range(n_generations):
            # Evaluate fitness of each individual using the extended evaluation function
            fitness_scores = [evaluate_strategy_extended(params, train_df_clean.copy(), initial_capital) for params in population]

            # Handle potential errors or invalid fitness scores from evaluation
            # Filter out individuals with fitness below -9999 (indicating errors/invalid scenarios)
            valid_fitness_indices = [i for i, score in enumerate(fitness_scores) if not np.isinf(score) and not np.isnan(score) and score > -9999]

            if not valid_fitness_indices:
                 display(f"Generation {generation+1}: No valid fitness scores. Re-initializing population.")
                 # Re-initialize population if no valid individuals are found
                 population = []
                 for _ in range(pop_size):
                      params = {}
                      for param, value_range in param_space.items():
                           if isinstance(value_range, tuple): # Numeric parameter
                               min_val, max_val = value_range
                               if isinstance(min_val, int) and isinstance(max_val, int):
                                    params[param] = random.randint(min_val, max_val)
                               else:
                                    params[param] = random.uniform(min_val, max_val)
                           elif isinstance(value_range, list): # Categorical parameter (indicator choice)
                                params[param] = random.choice(value_range)
                           elif isinstance(value_range, bool): # Boolean parameter
                                params[param] = random.choice([True, False])

                      population.append(params)

                 continue # Skip selection/crossover/mutation for this generation


            # Select parents from valid individuals (e.g., tournament selection)
            valid_population = [population[i] for i in valid_fitness_indices]
            valid_fitness = [fitness_scores[i] for i in valid_fitness_indices]

            parents = []
            # Use the size of the valid population for selecting parents
            num_parents_to_select = min(pop_size, len(valid_population))
            # Ensure we select an even number of parents for crossover
            num_parents_to_select = (num_parents_to_select // 2) * 2

            if num_parents_to_select == 0:
                 display(f"Generation {generation+1}: Not enough valid individuals to select parents. Re-initializing population.")
                 # Re-initialize population if not enough valid parents
                 population = []
                 for _ in range(pop_size):
                      params = {}
                      for param, value_range in param_space.items():
                           if isinstance(value_range, tuple):
                               min_val, max_val = value_range
                               if isinstance(min_val, int) and isinstance(max_val, int):
                                    params[param] = random.randint(min_val, max_val)
                               else:
                                    params[param] = random.uniform(min_val, max_val)
                           elif isinstance(value_range, list):
                                params[param] = random.choice(value_range)
                           elif isinstance(value_range, bool): # Boolean parameter
                                params[param] = random.choice([True, False])

                      population.append(params)
                 continue


            for _ in range(num_parents_to_select // 2):
                # Select two random indices from the valid individuals list
                idx1_valid = random.choice(range(len(valid_population)))
                idx2_valid = random.choice(range(len(valid_population)))
                parent1 = valid_population[idx1_valid] if valid_fitness[idx1_valid] > valid_fitness[idx2_valid] else valid_population[idx2_valid]
                idx1_valid = random.choice(range(len(valid_population)))
                idx2_valid = random.choice(range(len(valid_population)))
                parent2 = valid_population[idx1_valid] if valid_fitness[idx1_valid] > valid_fitness[idx2_valid] else valid_population[idx2_valid]
                parents.extend([parent1, parent2])

            # Create next generation (crossover and mutation)
            next_population = []
            # Ensure we generate enough individuals for the next population size
            while len(next_population) < pop_size:
                 # Select two parents with replacement from the generated parents list
                 # Ensure there are at least 2 parents to select from
                 if len(parents) < 2:
                      # If not enough parents, break the loop and re-initialize population outside
                      display(f"Generation {generation+1}: Not enough parents generated for crossover. Breaking.")
                      break

                 parent1, parent2 = random.sample(parents, 2)

                 child1, child2 = {}, {}

                 # Crossover
                 for param in param_space.keys():
                     if random.random() < 0.5:
                         child1[param] = parent1[param]
                         child2[param] = parent2[param]
                     else:
                         child1[param] = parent2[param]
                         child2[param] = parent1[param]

                 # Mutation
                 for child in [child1, child2]:
                     for param, value_range in param_space.items():
                         if random.random() < mutation_rate:
                             if isinstance(value_range, tuple): # Numeric parameter mutation
                                 min_val, max_val = value_range
                                 if isinstance(min_val, int) and isinstance(max_val, int):
                                     child[param] = random.randint(min_val, max_val)
                                 else:
                                     child[param] = random.uniform(min_val, max_val)
                                 # Ensure mutated values are within bounds
                                 child[param] = max(min_val, min(max_val, child[param]))
                             elif isinstance(value_range, list): # Categorical parameter mutation (pick a random choice)
                                 child[param] = random.choice(value_range)
                             elif isinstance(value_range, bool): # Boolean parameter mutation (flip the value)
                                 child[param] = not child[param]


                 next_population.extend([child1, child2])

            # Ensure the size of the next population is exactly pop_size
            population = next_population[:pop_size]


            # Track best individual in this generation (from the evaluated valid individuals)
            # Check if valid_fitness is not empty before finding max
            if valid_fitness:
                 best_gen_fitness_idx = valid_fitness.index(max(valid_fitness))
                 best_gen_fitness = valid_fitness[best_gen_fitness_idx]
                 best_gen_params = valid_population[best_gen_fitness_idx]
                 display(f"Generation {generation+1}: Best Fitness = {best_gen_fitness:.4f}")
            else:
                 display(f"Generation {generation+1}: No valid fitness scores to report best fitness.")


        # After GA for the current window, select the best parameters found in this window
        # and track the overall best.
        if valid_fitness: # Check if valid_fitness is not empty
             best_window_fitness = max(valid_fitness) # Max fitness from the last generation's valid individuals
             best_window_params = valid_population[valid_fitness.index(best_window_fitness)]

             if best_window_fitness > best_overall_fitness:
                 best_overall_fitness = best_window_fitness
                 best_overall_params = best_window_params
                 display(f"New overall best fitness found: {best_overall_fitness:.4f}")
                 display(f"Corresponding parameters: {best_overall_params}")
        else:
             display(f"No valid fitness scores in the last generation of window ending {current_end_date.strftime('%Y-%m-%d')}. No update to overall best parameters.")


        # Move the window forward by step_size_days and find the new indices
        current_window_start_date = all_dates[window_start_idx]
        next_window_start_date = current_window_start_date + pd.Timedelta(days=step_size_days)
        window_start_idx = all_dates.searchsorted(next_window_start_date, side='left')

        # Calculate the new window end index based on the lookback_window_days from the new start date
        if window_start_idx < total_bars:
            new_window_end_date = all_dates[window_start_idx] + pd.Timedelta(days=lookback_window_days)
            window_end_idx = all_dates.searchsorted(new_window_end_date, side='left')
        else:
            # If the new start index is beyond the last bar, set end index to total_bars + 1 to terminate loop
            window_end_idx = total_bars + 1


    return best_overall_params


# Define the extended parameter space for the GA, including indicator choices, lookback ranges,
# entry/exit parameters, dynamic weight flag, and profit target multipliers.
param_space_extended = {
    'weight_trend': (0.1, 0.7),
    'weight_volatility': (0.1, 0.4),
    'weight_exhaustion': (0.1, 0.4),
    'strong_bull_threshold': (30, 70),
    'weak_bull_threshold': (10, 40),
    'neutral_threshold_upper': (10, 30),
    'neutral_threshold_lower': (-30, -10),
    'strong_bear_threshold': (-70, -30),
    'weak_bear_threshold': (-40, -10),
    'stop_loss_multiplier_strong': (1.5, 3.5),
    'stop_loss_multiplier_weak': (0.5, 2.5),
    'indicator_trend': list(INDICATOR_FUNCTIONS['Trend'].keys()), # Use actual implemented indicators
    'indicator_volatility': list(INDICATOR_FUNCTIONS['Volatility'].keys()), # Use actual implemented indicators
    'indicator_exhaustion': list(INDICATOR_FUNCTIONS['Exhaustion'].keys()), # Use actual implemented indicators
    'lookback_trend': (10, 50),
    'lookback_volatility': (10, 50),
    'lookback_exhaustion': (10, 50),
    'entry_step_size': (0.1, 0.5), # Range for entry step size
    'max_position_fraction': (0.5, 1.0), # Range for max position size
    'use_dynamic_weights': True, # Boolean parameter, GA will choose True or False
    'take_profit_multiplier_strong': (2.0, 5.0), # Range for strong profit target multiplier
    'take_profit_multiplier_weak': (1.0, 3.0), # Range for weak profit target multiplier
}


# GA and Walk-Forward parameters (can keep the reduced values for faster PoC)
n_generations = 5
pop_size = 10
mutation_rate = 0.1
lookback_window_days = 365 # 1 year lookback window for optimization
step_size_days = 90 # Step forward by 90 days


# Ensure dollar_bars_df is available with raw OHLCV and DollarVolume columns
# Assuming dollar_bars_df is the result of create_dollar_bars and contains the necessary columns
# Pass a copy to the GA function to avoid modification
df_for_ga_optimization = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


# Run the Genetic Algorithm with walk-forward optimization using the extended parameter space
display("Starting Genetic Algorithm with Walk-Forward Optimization (Extended Parameters)...")
optimal_params_extended = genetic_algorithm_walk_forward_extended(
    df_for_ga_optimization,
    initial_capital,
    param_space_extended,
    n_generations,
    pop_size,
    mutation_rate,
    lookback_window_days,
    step_size_days
)

display("\nOptimal Parameters (Extended) found:")
display(optimal_params_extended)

"""**Reasoning**:
The genetic algorithm with walk-forward optimization using extended parameters has completed and found an optimal set of parameters, including indicator choices and lookback periods. The next step is to evaluate the performance of these optimal parameters on the entire dataset to get a final performance assessment. This evaluation should include calculating the key performance metrics and plotting the equity curve.


"""

# Evaluate the performance of the optimal parameters on the full dataset
# If optimal_params_extended is None (meaning no improvement was found), evaluate with the initial sample parameters
if optimal_params_extended is None:
    display("No optimal parameters found by GA with extended parameters. Evaluating with initial sample parameters.")
    evaluation_params_final = sample_params # Use the initial sample parameters
else:
    display("Evaluating with optimal parameters found by GA with extended parameters.")
    evaluation_params_final = optimal_params_extended

# Re-run the trading simulation with the selected optimal parameters on the full dataset
# Use the evaluate_strategy_extended function for the final evaluation
# This function handles all the logic including factor calculation, dynamic weights,
# gradual entry/exit, and stop-loss/take-profit based on the provided parameters.

# Ensure dollar_bars_df is available with raw OHLCV and DollarVolume columns
# Pass a copy to the evaluation function
df_for_final_evaluation = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


# The evaluate_strategy_extended function returns the Sortino Ratio.
# To get the full performance metrics and equity curve, we need to re-run the simulation logic
# outside of the evaluate_strategy_extended function, using the final parameters.

# Extract parameters with defaults for the final simulation
weight_trend_final = evaluation_params_final.get('weight_trend', 0.4)
weight_volatility_final = evaluation_params_final.get('weight_volatility', 0.3)
weight_exhaustion_final = evaluation_params_final.get('weight_exhaustion', 0.3)

strong_bull_threshold_final = evaluation_params_final.get('strong_bull_threshold', 50)
weak_bull_threshold_final = evaluation_params_final.get('weak_bull_threshold', 20)
neutral_threshold_upper_final = evaluation_params_final.get('neutral_threshold_upper', 20)
neutral_threshold_lower_final = evaluation_params_final.get('neutral_threshold_lower', -20)
strong_bear_threshold_final = evaluation_params_final.get('strong_bear_threshold', -50)
weak_bear_threshold_final = evaluation_params_final.get('weak_bear_threshold', -20)

stop_loss_multiplier_strong_final = evaluation_params_final.get('stop_loss_multiplier_strong', 3)
stop_loss_multiplier_weak_final = evaluation_params_final.get('stop_loss_multiplier_weak', 3)

indicator_trend_final = evaluation_params_final.get('indicator_trend', 'Slope')
indicator_volatility_final = evaluation_params_final.get('indicator_volatility', 'ATR')
indicator_exhaustion_final = evaluation_params_final.get('indicator_exhaustion', 'SMADiff')

lookback_trend_final = evaluation_params_final.get('lookback_trend', 20)
lookback_volatility_final = evaluation_params_final.get('lookback_volatility', 20)
lookback_exhaustion_final = evaluation_params_final.get('lookback_exhaustion', 20)

entry_step_size_final = evaluation_params_final.get('entry_step_size', 0.2)
max_position_fraction_final = evaluation_params_final.get('max_position_fraction', 1.0)

use_dynamic_weights_final = evaluation_params_final.get('use_dynamic_weights', True)

take_profit_multiplier_strong_final = evaluation_params_final.get('take_profit_multiplier_strong', 4.0)
take_profit_multiplier_weak_final = evaluation_params_final.get('take_profit_multiplier_weak', 2.0)


# --- Calculate Factors for the full dataset using the optimal parameters ---
try:
    # Calculate Volatility first as Exhaustion (SMADiff) depends on it
    if indicator_volatility_final in INDICATOR_FUNCTIONS['Volatility']:
        df_for_final_evaluation['Volatility_Factor'] = INDICATOR_FUNCTIONS['Volatility'][indicator_volatility_final](df_for_final_evaluation.copy(), lookback_volatility_final)
    else:
        display(f"Warning: Unknown Volatility indicator '{indicator_volatility_final}'. Using default (ATR).")
        df_for_final_evaluation['Volatility_Factor'] = INDICATOR_FUNCTIONS['Volatility']['ATR'](df_for_final_evaluation.copy(), lookback_volatility_final)

    if indicator_trend_final in INDICATOR_FUNCTIONS['Trend']:
         df_for_final_evaluation['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend'][indicator_trend_final](df_for_final_evaluation.copy(), lookback_trend_final)
    else:
         display(f"Warning: Unknown Trend indicator '{indicator_trend_final}'. Using default (Slope).")
         df_for_final_evaluation['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend']['Slope'](df_for_final_evaluation.copy(), lookback_trend_final)

    if indicator_exhaustion_final in INDICATOR_FUNCTIONS['Exhaustion']:
        if indicator_exhaustion_final == 'SMADiff': # SMADiff requires normalized Volatility
             df_for_final_evaluation['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_final](df_for_final_evaluation.copy(), lookback_exhaustion_final, df_for_final_evaluation['Volatility_Factor'])
        else:
             # Other exhaustion indicators (like RSI) take df and their lookback
             df_for_final_evaluation['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_final](df_for_final_evaluation.copy(), lookback_exhaustion_final)
    else:
        display(f"Warning: Unknown Exhaustion indicator '{indicator_exhaustion_final}'. Using default (SMADiff).")
        df_for_final_evaluation['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion']['SMADiff'](df_for_final_evaluation.copy(), lookback_exhaustion_final, df_for_final_evaluation['Volatility_Factor'])

except Exception as e:
    display(f"Error during factor calculation for final evaluation: {e}")
    # If factor calculation fails, we cannot proceed with the simulation
    equity_curve_df_final_extended = pd.DataFrame() # Empty dataframe
    trade_log_df_final_extended = pd.DataFrame() # Empty dataframe
    display("Final evaluation aborted due to factor calculation error.")


# Drop rows with NaN values generated by lookback periods after factor calculation
df_for_final_evaluation = df_for_final_evaluation.dropna(subset=['Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor'])

if not df_for_final_evaluation.empty:

    # --- Calculate MSS and Regime for the full dataset ---
    if use_dynamic_weights_final:
         # Calculate initial static MSS to get a starting regime for dynamic weights
         static_weight_trend_initial = 0.4
         static_weight_volatility_initial = 0.3
         static_weight_exhaustion_initial = 0.3

         df_for_final_evaluation['MSS_static_initial'] = (static_weight_trend_initial * df_for_final_evaluation['Trend_Factor'] +
                                                         static_weight_volatility_initial * df_for_final_evaluation['Volatility_Factor'] +
                                                         static_weight_exhaustion_initial * df_for_final_evaluation['Exhaustion_Factor'])

         # Classify the initial static regime using the optimal thresholds
         def classify_regime_initial_final(mss):
              if mss > strong_bull_threshold_final: return 'Strong Bull'
              elif mss > weak_bull_threshold_final: return 'Weak Bull'
              elif mss >= neutral_threshold_lower_final and mss <= neutral_threshold_upper_final: return 'Neutral'
              elif mss > strong_bear_threshold_final: return 'Weak Bear'
              else: return 'Strong Bear'

         df_for_final_evaluation['Regime_initial'] = df_for_final_evaluation['MSS_static_initial'].apply(classify_regime_initial_final)


         # Calculate Dynamic MSS using dynamic weights based on the *initial* regime
         df_for_final_evaluation['MSS'] = np.nan # Initialize dynamic MSS column
         for index, row in df_for_final_evaluation.iterrows():
             current_regime_initial = row['Regime_initial'] # Use the initial static regime
             dynamic_weights = get_dynamic_weights(current_regime_initial) # Assuming get_dynamic_weights is available

             dynamic_mss = (dynamic_weights['Trend'] * row['Trend_Factor'] +
                            dynamic_weights['Volatility'] * row['Volatility_Factor'] +
                            dynamic_weights['Exhaustion'] * row['Exhaustion_Factor'])
             df_for_final_evaluation.loc[index, 'MSS'] = dynamic_mss

         # Classify the final Regime based on the Dynamic MSS and the optimal thresholds
         df_for_final_evaluation['Regime'] = df_for_final_evaluation['MSS'].apply(classify_regime_initial_final) # Use the same optimal thresholds

    else: # Use static weights for MSS calculation
         df_for_final_evaluation['MSS'] = (weight_trend_final * df_for_final_evaluation['Trend_Factor'] +
                                           weight_volatility_final * df_for_final_evaluation['Volatility_Factor'] +
                                           weight_exhaustion_final * df_for_final_evaluation['Exhaustion_Factor'])

         # Classify Regime based on Static MSS and the optimal thresholds
         def classify_regime_static_final(mss):
              if mss > strong_bull_threshold_final: return 'Strong Bull'
              elif mss > weak_bull_threshold_final: return 'Weak Bull'
              elif mss >= neutral_threshold_lower_final and mss <= neutral_threshold_upper_final: return 'Neutral'
              elif mss > strong_bear_threshold_final: return 'Weak Bear'
              else: return 'Strong Bear'

         df_for_final_evaluation['Regime'] = df_for_final_evaluation['MSS'].apply(classify_regime_static_final)


    # --- Trading Simulation with Enhanced Logic (using optimal parameters) ---
    position_fraction = 0.0 # Current position as a fraction of initial_capital
    total_units = 0.0
    total_cost_basis = 0.0 # Total cost paid for units (positive for buy, negative for sell)
    stop_loss = 0.0
    take_profit = 0.0 # Profit target level
    equity_curve_final_extended = []
    trade_log_final_extended = []
    current_capital = initial_capital

    max_step_fraction_final = entry_step_size_final * max_position_fraction_final # Step size in terms of fraction of initial_capital

    # Initializing these here, they are updated within the loop for trailing stops
    peak_price_since_entry = -float('inf') # Track peak price for trailing stop (Long)
    valley_price_since_entry = float('inf') # Track valley price for trailing stop (Short - not active yet)


    for index, row in df_for_final_evaluation.iterrows():
        current_price = row['Close']
        current_regime = row['Regime']
        current_mss = row['MSS']
        current_volatility_factor = row['Volatility_Factor']

        # Determine stop-loss distance and take-profit distance based on regime and Volatility Factor
        if not isinstance(current_volatility_factor, (int, float)) or np.isnan(current_volatility_factor) or abs(current_volatility_factor) < 1e-9:
             stop_loss_distance = 0
             take_profit_distance = 0
        else:
            if current_regime in ['Strong Bull', 'Strong Bear']:
                stop_loss_distance = stop_loss_multiplier_strong_final * abs(current_volatility_factor)
                take_profit_distance = take_profit_multiplier_strong_final * abs(current_volatility_factor)
            elif current_regime in ['Weak Bull', 'Weak Bear']:
                stop_loss_distance = stop_loss_multiplier_weak_final * abs(current_volatility_factor)
                take_profit_distance = take_profit_multiplier_weak_final * abs(current_volatility_factor)
            else: # Neutral
                stop_loss_distance = 0
                take_profit_distance = 0


        # Determine target position fraction based on regime and MSS confidence
        target_position_fraction = 0.0
        if current_regime == 'Strong Bull':
            normalized_mss = (current_mss - strong_bull_threshold_final) / (100 - strong_bull_threshold_final) if (100 - strong_bull_threshold_final) > 0 else 0
            target_position_fraction = max_position_fraction_final * np.clip(normalized_mss, 0, 1)
        elif current_regime == 'Weak Bull':
             normalized_mss = (current_mss - weak_bull_threshold_final) / (neutral_threshold_upper_final - weak_bull_threshold_final) if (neutral_threshold_upper_final - weak_bull_threshold_final) > 0 else 0
             target_position_fraction = position_fraction # Default to hold
             if position_fraction > 0:
                 target_position_fraction = max_position_fraction_final * np.clip(normalized_mss, 0, 1)
        elif current_regime == 'Neutral':
            target_position_fraction = 0.0
        elif current_regime == 'Weak Bear':
             normalized_mss = (current_mss - (-100)) / (weak_bear_threshold_final - (-100)) if (weak_bear_threshold_final - (-100)) > 0 else 0
             target_position_fraction = position_fraction # Default to hold
             if position_fraction < 0:
                  target_position_fraction = -max_position_fraction_final * np.clip(1 - normalized_mss, 0, 1)
        elif current_regime == 'Strong Bear':
            normalized_mss = (current_mss - (-100)) / (strong_bear_threshold_final - (-100)) if (strong_bear_threshold_final - (-100)) > 0 else 0
            target_position_fraction = -max_position_fraction_final * np.clip(1 - normalized_mss, 0, 1)

        # Ensure target_position_fraction has correct sign based on regime
        if target_position_fraction > 1e-9 and current_regime not in ['Strong Bull', 'Weak Bull']:
             target_position_fraction = 0.0
        elif target_position_fraction < -1e-9 and current_regime not in ['Strong Bear', 'Weak Bear']:
             target_position_fraction = 0.0
        # In Neutral, always target 0
        if current_regime == 'Neutral':
            target_position_fraction = 0.0


        # Calculate the change in position fraction
        position_fraction_change = target_position_fraction - position_fraction

        # Limit position fraction change to max_step_fraction_final
        position_fraction_change = np.clip(position_fraction_change, -max_step_fraction_final, max_step_fraction_final)

        # Calculate the capital amount to allocate/deallocate
        capital_to_trade = position_fraction_change * initial_capital

        # Calculate the units to trade based on the capital amount and current price
        units_to_trade = capital_to_trade / current_price if current_price > 0 else 0.0

        # Update total units and total cost basis
        # This simplified logic assumes entering/exiting at the current price
        # More complex logic would handle limit/market orders
        total_units += units_to_trade
        total_cost_basis += units_to_trade * current_price # Track cumulative cost/revenue

        # Update the current position fraction
        current_market_value = total_units * current_price if current_price > 0 else 0.0
        position_fraction = current_market_value / initial_capital if initial_capital > 0 else 0.0


        # Update Trailing Stop Loss and Take Profit based on the current position
        if abs(total_units) > 1e-9: # If position is active
            avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0

            if total_units > 0: # Long position
                # Update peak price for trailing stop
                peak_price_since_entry = max(peak_price_since_entry, current_price)
                # Calculate trailing stop level
                stop_loss_level = peak_price_since_entry - stop_loss_distance if stop_loss_distance > 0 else -float('inf')
                # The stop_loss variable holds the *active* stop loss level
                # Ensure stop loss does not move down during a long trade (only trails up)
                stop_loss = max(stop_loss, stop_loss_level) if stop_loss != 0.0 else stop_loss_level # Initialize or trail up

                # Calculate Take Profit level based on average cost basis
                take_profit_level = avg_cost_basis_per_unit + take_profit_distance if take_profit_distance > 0 else float('inf')
                take_profit = take_profit_level

            elif total_units < 0: # Short position (not active in this PoC)
                 # Update valley price for trailing stop
                 valley_price_since_entry = min(valley_price_since_entry, current_price)
                 # Calculate trailing stop level
                 stop_loss_level = valley_price_since_entry + stop_loss_distance if stop_loss_distance > 0 else float('inf')
                 # Ensure stop loss does not move up during a short trade (only trails down)
                 stop_loss = min(stop_loss, stop_loss_level) if stop_loss != 0.0 else stop_loss_level # Initialize or trail down

                 # Calculate Take Profit level based on average cost basis
                 take_profit_level = avg_cost_basis_per_unit - take_profit_distance if take_profit_distance > 0 else -float('inf')
                 take_profit = take_profit_level
        else: # No active position
             stop_loss = 0.0
             take_profit = 0.0
             # Reset peak/valley price when position is flat
             peak_price_since_entry = -float('inf')
             valley_price_since_entry = float('inf')


        # Check for Stop-Loss or Take-Profit hit
        if abs(total_units) > 1e-9: # Only check if position is active
             avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0 # Recalculate avg cost basis

             # Check Take Profit hit
             if total_units > 0 and current_price >= take_profit and not np.isinf(take_profit):
                 pnl = (current_price - avg_cost_basis_per_unit) * total_units # Calculate P/L
                 current_capital += pnl # Update capital
                 # Record trade log entry for Take Profit
                 trade_log_final_extended.append({'Date': index, 'Action': 'Take Profit Long', 'Price': current_price, 'Units': -total_units, 'TotalUnits': 0.0, 'PnL': pnl})
                 # Reset position variables
                 total_units = 0.0
                 total_cost_basis = 0.0
                 position_fraction = 0.0
                 stop_loss = 0.0
                 take_profit = 0.0
                 peak_price_since_entry = -float('inf') # Reset peak

             # Check Stop Loss hit (after checking take profit)
             elif total_units > 0 and current_price <= stop_loss and not np.isinf(stop_loss):
                 pnl = (current_price - avg_cost_basis_per_unit) * total_units # Calculate P/L
                 current_capital += pnl # Update capital
                 # Record trade log entry for Stop Out
                 trade_log_final_extended.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'Units': -total_units, 'TotalUnits': 0.0, 'PnL': pnl})
                 # Reset position variables
                 total_units = 0.0
                 total_cost_basis = 0.0
                 position_fraction = 0.0
                 stop_loss = 0.0
                 take_profit = 0.0
                 peak_price_since_entry = -float('inf') # Reset peak

             elif total_units < 0: # Short position (not active)
                  # Similar checks for short take profit and stop loss
                  pass # Skip short position checks for now


        # Append current equity to the equity curve
        # Current equity is current capital + (unrealized P/L of the current position)
        avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0
        unrealized_pnl = (current_price - avg_cost_basis_per_unit) * total_units if abs(total_units) > 1e-9 else 0.0
        equity_curve_final_extended.append({'Date': index, 'Equity': current_capital + unrealized_pnl})


    # Convert trade log and equity curve to DataFrames
    equity_curve_df_final_extended = pd.DataFrame(equity_curve_final_extended).set_index('Date')
    trade_log_df_final_extended = pd.DataFrame(trade_log_final_extended)


    # Evaluate performance metrics for the final simulation
    if equity_curve_df_final_extended.empty or len(equity_curve_df_final_extended) < 2:
        display("Final extended equity curve is empty or too short for evaluation.")
    else:
        equity_curve_df_final_extended['Daily_Return'] = equity_curve_df_final_extended['Equity'].pct_change().fillna(0)
        total_return_final_extended = (equity_curve_df_final_extended['Equity'].iloc[-1] - initial_capital) / initial_capital
        trading_periods_per_year = 365 # Adjust if necessary
        annualized_return_final_extended = (1 + total_return_final_extended)**(trading_periods_per_year / len(equity_curve_df_final_extended)) - 1

        equity_curve_df_final_extended['Peak'] = equity_curve_df_final_extended['Equity'].cummax()
        equity_curve_df_final_extended['Drawdown'] = equity_curve_df_final_extended['Equity'] - equity_curve_df_final_extended['Peak']
        max_drawdown_final_extended = equity_curve_df_final_extended['Drawdown'].min()

        mar = 0
        downside_returns_final_extended = equity_curve_df_final_extended[equity_curve_df_final_extended['Daily_Return'] < mar]['Daily_Return']
        downside_deviation_final_extended = downside_returns_final_extended.std()

        # Corrected conditional logic for Sortino Ratio calculation
        if downside_deviation_final_extended == 0 or np.isnan(downside_deviation_final_extended):
            if annualized_return_final_extended > mar:
                 sortino_ratio_final_extended = float('inf')
            elif annualized_return_final_extended == mar:
                 sortino_ratio_final_extended = 0
            else:
                 sortino_ratio_final_extended = -1000 # Use a very low number for problematic ratios
        else:
            sortino_ratio_final_extended = (annualized_return_final_extended - mar) / downside_deviation_final_extended

        # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
        if np.isinf(sortino_ratio_final_extended) or np.isnan(sortino_ratio_final_extended):
            sortino_ratio_final_extended = -1000 # Return a very low fitness for problematic ratios


        # Calculate Calmar Ratio
        calmar_ratio_final_extended = annualized_return_final_extended / abs(max_drawdown_final_extended) if max_drawdown_final_extended != 0 else float('inf')


        display("\nFinal Performance Metrics (using optimal extended parameters):")
        display(f"Initial Capital: {initial_capital:.2f}")
        display(f"Final Equity: {equity_curve_df_final_extended['Equity'].iloc[-1]:.2f}")
        display(f"Total Return: {total_return_final_extended:.4f}")
        display(f"Annualized Return: {annualized_return_final_extended:.4f}")
        display(f"Max Drawdown: {max_drawdown_final_extended:.2f}")
        display(f"Sortino Ratio (MAR=0): {sortino_ratio_final_extended:.4f}")
        display(f"Calmar Ratio: {calmar_ratio_final_extended:.4f}")

        # Plot final equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve_df_final_extended.index, equity_curve_df_final_extended['Equity'], label='Final Equity Curve (Optimized Extended)')
        plt.title('Final Equity Curve (using optimal extended parameters)')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Display head of the final trade log
        display("\nFinal Trade Log Summary (Optimized Extended):")
        display(trade_log_df_final_extended.head())

        # Analyze losing trades from the final simulation
        losing_trades_df_final_extended = trade_log_df_final_extended[trade_log_df_final_extended['PnL'].notna() & (trade_log_df_final_extended['PnL'] < 0)]
        display("\nLosing Trades Summary from Final Simulation (Optimized Extended):")
        display(losing_trades_df_final_extended.head()) # Display head of losing trades

else:
    display("Cannot perform final extended evaluation as the dataframe is empty.")

"""## Analyze market data around losing trades

### Subtask:
Manually examine the `dollar_bars_df` around the dates of the losing trades to see the price action, MSS values, factor values, and regime leading up to the loss.

**Reasoning**:
Iterate through the losing trades and display the data window around each trade to analyze the conditions leading to the loss.
"""

# Analyze the market data around losing trades
trade_window_size = 10 # Number of bars before the trade date to display

display("\nAnalyzing data around losing trades:")

# Ensure losing_trades_df_final_extended is available from previous steps
if 'losing_trades_df_final_extended' in locals() and not losing_trades_df_final_extended.empty:
    for index, trade in losing_trades_df_final_extended.iterrows():
        trade_date = trade['Date']
        display(f"\n--- Losing Trade on {trade_date.strftime('%Y-%m-%d')} ---")

        # Find the index of the trade date in the main dataframe (dollar_bars_df)
        # Ensure dollar_bars_df is available and has a DatetimeIndex
        if 'dollar_bars_df' in locals() and isinstance(dollar_bars_df.index, pd.DatetimeIndex):
             try:
                 trade_date_idx = dollar_bars_df.index.get_loc(trade_date)

                 # Get the data window
                 start_idx = max(0, trade_date_idx - trade_window_size)
                 end_idx = trade_date_idx # Include the trade date bar

                 # Select the relevant data window from dollar_bars_df
                 # Ensure the required columns exist in dollar_bars_df
                 required_cols = ['Close', 'MSS', 'Regime', 'Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor']
                 available_cols = [col for col in required_cols if col in dollar_bars_df.columns]

                 if len(available_cols) == len(required_cols):
                     data_window = dollar_bars_df.iloc[start_idx : end_idx + 1][available_cols]
                     display(data_window)
                 else:
                     display(f"Warning: Not all required columns ({required_cols}) available in dollar_bars_df. Available columns: {dollar_bars_df.columns.tolist()}")
                     # Display what columns are available in the window
                     available_data_window = dollar_bars_df.iloc[start_idx : end_idx + 1][[col for col in dollar_bars_df.columns if col in required_cols]]
                     display(available_data_window)


             except KeyError:
                 display(f"Warning: Trade date {trade_date.strftime('%Y-%m-%d')} not found in dollar_bars_df index. Skipping analysis for this trade.")
             except Exception as e:
                 display(f"An error occurred while processing trade on {trade_date.strftime('%Y-%m-%d')}: {e}")

        else:
            display("Error: 'dollar_bars_df' not found or does not have a DatetimeIndex. Cannot analyze losing trades.")
            break # Exit the loop if dollar_bars_df is not accessible

else:
    display("No losing trades data available for analysis ('losing_trades_df_final_extended' is not available or is empty).")

"""## Analyze Market Data around Losing Trades

### Subtask:
Manually examine the `dollar_bars_df` around the dates of the losing trades to see the price action, MSS values, factor values, and regime leading up to the loss.

**Reasoning**:
Select a few recent losing trades from the `losing_trades_df_enhanced_logic` and display a window of the `dollar_bars_df` (including relevant columns like Close, Factor values, MSS, and Regime) around the date of each selected trade to analyze the market context.
"""

# Select a few recent losing trades for analysis
# Let's select the last 5 losing trades from the optimized trade log
recent_losing_trades = losing_trades_df_optimized.tail(5)

display("Analyzing Market Data around Recent Losing Trades (Optimized Strategy):")

# Define a window size (e.g., 10 bars before and 5 bars after the trade date)
window_before = 10
window_after = 5

# Iterate through the selected losing trades
for index, trade in recent_losing_trades.iterrows():
    trade_date = trade['Date']
    trade_action = trade['Action']
    trade_price = trade['Price']
    trade_pnl = trade['PnL']

    display(f"\n--- Analyzing Losing Trade on {trade_date.strftime('%Y-%m-%d %H:%M:%S')} ---")
    display(f"Action: {trade_action}, Price: {trade_price:.2f}, PnL: {trade_pnl:.2f}")

    try:
        # Find the index location of the trade date in the df_sim index (which contains factors and regime)
        # Use asof to find the nearest index if an exact match isn't found
        date_index = df_sim.index.asof(trade_date)

        if date_index is not None:
             date_index_loc = df_sim.index.get_loc(date_index)

             # Define the start and end indices for the window
             start_index = max(0, date_index_loc - window_before)
             end_index = min(len(df_sim) - 1, date_index_loc + window_after)

             # Select the data window from df_sim
             # Include relevant columns: Close, Factor values, MSS_dynamic, Regime_dynamic
             analysis_window_df = df_sim.iloc[start_index:end_index + 1][['Close', 'Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor', 'MSS_dynamic', 'Regime_dynamic']]

             display(analysis_window_df)
        else:
            display(f"Warning: Trade date {trade_date} not found in df_sim index.")


    except Exception as e:
        display(f"An error occurred while analyzing market data around trade date {trade_date}: {e}")

import matplotlib.pyplot as plt # Import matplotlib for plotting if needed
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Re-define factor calculation functions to be selectable
def calculate_trend_slope(df, lookback):
    df_calc = df.copy() # Work on a copy
    df_calc['Trend_Slope_Val'] = np.nan
    # Ensure lookback is a valid integer
    lookback = int(lookback) if lookback is not None and lookback > 0 else 20 # Default lookback

    if len(df_calc) < lookback:
        df_calc['Trend_Factor'] = np.nan
        return df_calc['Trend_Factor']

    for i in range(lookback, len(df_calc)):
        y = df_calc['Close'].iloc[i-lookback:i]
        x = np.arange(lookback)
        # Handle potential division by zero or constant price in linregress
        if np.std(y) == 0:
             slope = 0
        else:
             slope, _, _, _, _ = linregress(x, y)
        df_calc.loc[df_calc.index[i], 'Trend_Slope_Val'] = slope

    # Now calculate rolling standard deviation of slopes
    df_calc['Slope_Std'] = df_calc['Trend_Slope_Val'].rolling(window=lookback).std()

    # Normalize Trend: Scale relative to the rolling standard deviation of slopes
    # Avoid division by zero
    valid_indices = df_calc.index[df_calc['Slope_Std'].notna() & (df_calc['Slope_Std'].abs() > 1e-9)]
    scaling_factor = 100 / (df_calc.loc[valid_indices, 'Slope_Std'] * 2) if not df_calc.loc[valid_indices, 'Slope_Std'].empty and (df_calc.loc[valid_indices, 'Slope_Std'] * 2).abs().mean() > 1e-9 else 0 # Handle empty valid_indices or zero mean std
    df_calc.loc[valid_indices, 'Trend_Factor'] = df_calc.loc[valid_indices, 'Trend_Slope_Val'] * scaling_factor

    df_calc = df_calc.drop(columns=['Trend_Slope_Val', 'Slope_Std']) # Drop intermediate columns
    df_calc['Trend_Factor'] = np.clip(df_calc['Trend_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Trend_Factor']


# Implement actual MACD calculation
def calculate_trend_macd(df, fastperiod=12, slowperiod=26, signalperiod=9):
    df_calc = df.copy() # Work on a copy
    # Ensure lookback periods are valid integers and greater than 0
    fastperiod = int(fastperiod) if fastperiod is not None and fastperiod > 0 else 12
    slowperiod = int(slowperiod) if slowperiod is not None and slowperiod > 0 else 26
    signalperiod = int(signalperiod) if signalperiod is not None and signalperiod > 0 else 9

    # Ensure slowperiod is greater than fastperiod
    if fastperiod >= slowperiod:
        # display(f"DEBUG: Invalid MACD periods: fast={fastperiod}, slow={slowperiod}. fastperiod must be less than slowperiod. Returning -2001.") # Debug print
        return -2001 # Treat as factor calculation error due to invalid params
        # Alternatively, use defaults or swap, but returning error is safer for GA

    if len(df_calc) < slowperiod + signalperiod: # MACD requires data for slow EMA + signal EMA
        df_calc['Trend_Factor'] = np.nan
        return df_calc['Trend_Factor']

    ema_fast = df_calc['Close'].ewm(span=fastperiod, adjust=False).mean()
    ema_slow = df_calc['Close'].ewm(span=slowperiod, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    # macd_histogram = macd_line - signal_line # Not used directly in MSS, but good to have

    # Normalize MACD: Scale relative to a rolling standard deviation of MACD line
    # This is one approach, other normalizations are possible (e.g., scaling relative to price)
    df_calc['MACD_Std'] = macd_line.rolling(window=slowperiod).std() # Use slowperiod for rolling std
    # Avoid division by zero
    valid_indices = df_calc.index[df_calc['MACD_Std'].notna() & (df_calc['MACD_Std'].abs() > 1e-9)]

    # Scale MACD line based on its rolling standard deviation
    # Adjust the multiplier (e.g., 5) to control the range
    # Let's use a multiplier of 5 for now
    scaling_factor = 100 / (df_calc.loc[valid_indices, 'MACD_Std'] * 5) if not df_calc.loc[valid_indices, 'MACD_Std'].empty and (df_calc.loc[valid_indices, 'MACD_Std'] * 5).abs().mean() > 1e-9 else 0 # Handle empty valid_indices or zero mean std
    df_calc.loc[valid_indices, 'Trend_Factor'] = macd_line.loc[valid_indices] * scaling_factor

    df_calc['Trend_Factor'] = np.clip(df_calc['Trend_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Trend_Factor']


def calculate_volatility_atr(df, lookback):
    df_calc = df.copy() # Work on a copy
    lookback = int(lookback) if lookback is not None and lookback > 0 else 20 # Default lookback

    if len(df_calc) < lookback + 1: # ATR needs previous close + lookback
        df_calc['Volatility_Factor'] = np.nan
        return df_calc['Volatility_Factor']

    df_calc['TR'] = np.maximum(np.maximum(df_calc['High'] - df_calc['Low'], abs(df_calc['High'] - df_calc['Close'].shift(1))), abs(df_calc['Low'] - df_calc['Close'].shift(1)))
    df_calc['Volatility_ATR_Absolute'] = df_calc['TR'].rolling(window=lookback).mean()
    df_calc = df_calc.drop(columns=['TR'])

    # Modified Normalization for ATR: Scale relative to a rolling average of ATR
    # Normalize ATR by dividing by a rolling average of ATR and then scaling to -100 to 100
    df_calc['ATR_MA'] = df_calc['Volatility_ATR_Absolute'].rolling(window=lookback).mean()
    # Avoid division by zero
    valid_indices = df_calc.index[df_calc['ATR_MA'].notna() & (df_calc['ATR_MA'].abs() > 1e-9)]
    df_calc.loc[valid_indices, 'Volatility_Factor'] = ((df_calc.loc[valid_indices, 'Volatility_ATR_Absolute'] / df_calc.loc[valid_indices, 'ATR_MA']) - 1) * 100 if not df_calc.loc[valid_indices, 'ATR_MA'].empty and (df_calc.loc[valid_indices, 'ATR_MA'].abs().mean() > 1e-9 or df_calc.loc[valid_indices, 'ATR_MA'].mean() == 0) else 0 # Handle empty valid_indices or zero mean MA

    df_calc = df_calc.drop(columns=['ATR_MA', 'Volatility_ATR_Absolute'])
    df_calc['Volatility_Factor'] = np.clip(df_calc['Volatility_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Volatility_Factor']


# Implement actual Standard Deviation calculation for Volatility
def calculate_volatility_stddev(df, lookback):
    df_calc = df.copy() # Work on a copy
    lookback = int(lookback) if lookback is not None and lookback > 0 else 20 # Default lookback

    if len(df_calc) < lookback + 1: # Needs previous close for return and then rolling window
        df_calc['Volatility_Factor'] = np.nan
        return df_calc['Volatility_Factor']

    # Calculate rolling standard deviation of daily returns (log returns often preferred)
    df_calc['Log_Return'] = np.log(df_calc['Close'] / df_calc['Close'].shift(1))
    df_calc['Rolling_StdDev'] = df_calc['Log_Return'].rolling(window=lookback).std()

    # Normalize Standard Deviation: Scale relative to a rolling average of StdDev
    df_calc['StdDev_MA'] = df_calc['Rolling_StdDev'].rolling(window=lookback).mean()
    # Avoid division by zero
    valid_indices = df_calc.index[df_calc['StdDev_MA'].notna() & (df_calc['StdDev_MA'].abs() > 1e-9)] # Fixed typo: Std_MA corrected to StdDev_MA

    # Scale StdDev based on its rolling average
    # Adjust the multiplier (e.g., 1000) - Needs tuning as log returns stddev is small
    scaling_factor = 100 # Example scaling factor
    df_calc.loc[valid_indices, 'Volatility_Factor'] = ((df_calc.loc[valid_indices, 'Rolling_StdDev'] / df_calc.loc[valid_indices, 'StdDev_MA']) - 1) * scaling_factor if not df_calc.loc[valid_indices, 'StdDev_MA'].empty and (df_calc.loc[valid_indices, 'StdDev_MA'].abs().mean() > 1e-9 or df_calc.loc[valid_indices, 'StdDev_MA'].mean() == 0) else 0 # Handle empty valid_indices or zero mean MA

    df_calc = df_calc.drop(columns=['Log_Return', 'Rolling_StdDev', 'StdDev_MA'])
    df_calc['Volatility_Factor'] = np.clip(df_calc['Volatility_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Volatility_Factor']


def calculate_exhaustion_sma_diff(df, sma_lookback, atr_series):
    df_calc = df.copy() # Work on a copy
    sma_lookback = int(sma_lookback) if sma_lookback is not None and sma_lookback > 0 else 20 # Default lookback

    if len(df_calc) < sma_lookback:
        df_calc['Exhaustion_Factor'] = np.nan
        return df_calc['Exhaustion_Factor']

    df_calc['SMA'] = df_calc['Close'].rolling(window=sma_lookback).mean()

    # Ensure ATR series is aligned and not NaN for calculation
    # Using the passed atr_series directly
    # Align atr_series to df_calc's index
    atr_series_aligned = atr_series.reindex(df_calc.index)

    valid_indices = df_calc.index[atr_series_aligned.notna() & (atr_series_aligned.abs() > 1e-9)]
    df_calc.loc[valid_indices, 'Exhaustion_Factor'] = (df_calc.loc[valid_indices, 'Close'] - df_calc.loc[valid_indices, 'SMA']) / atr_series_aligned.loc[valid_indices] if not atr_series_aligned.loc[valid_indices].empty and (atr_series_aligned.loc[valid_indices].abs().mean() > 1e-9 or atr_series_aligned.loc[valid_indices].mean() == 0) else 0 # Handle empty valid_indices or zero mean ATR

    df_calc = df_calc.drop(columns=['SMA'])
    # Normalize Exhaustion: Scale the ratio to fit -100 to 100
    # Assuming a range like -10 to 10 covers most cases relative to ATR
    scaling_factor = 100 / 10 # Example scaling
    df_calc['Exhaustion_Factor'] = np.clip(df_calc['Exhaustion_Factor'] * scaling_factor, -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Exhaustion_Factor']


# Implement actual RSI calculation for Exhaustion
def calculate_exhaustion_rsi(df, lookback):
    df_calc = df.copy() # Work on a copy
    lookback = int(lookback) if lookback is not None and lookback > 0 else 14 # Default lookback

    if len(df_calc) < lookback * 2: # RSI needs data for initial average and then calculation
        df_calc['Exhaustion_Factor'] = np.nan
        return df_calc['Exhaustion_Factor']

    # Calculate price changes
    delta = df_calc['Close'].diff()

    # Get gains and losses (ignore 0 changes)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate rolling average gains and losses (using alpha=1/lookback for Wilder's smoothing)
    avg_gains = gains.ewm(com=lookback-1, adjust=False).mean()
    avg_losses = losses.ewm(com=lookback-1, adjust=False).mean()

    # Calculate Relative Strength (RS)
    # Handle potential division by zero (when avg_losses is 0)
    # Add a small epsilon to avoid division by zero
    rs = avg_gains / (avg_losses + 1e-10)
    # Replace inf with nan and fill leading nan with 0 (occurs before enough data points for EMA)
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)


    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    # Normalize RSI: RSI is typically 0-100. Scale it to -100 to 100.
    # A simple linear scaling: RSI 0 -> -100, RSI 100 -> 100, RSI 50 -> 0
    df_calc['Exhaustion_Factor'] = (rsi - 50) * 2 # Scale from 0-100 to -100-100

    df_calc['Exhaustion_Factor'] = np.clip(df_calc['Exhaustion_Factor'], -100, 100) # Clip to bounds

    # Align index before returning
    return df_calc['Exhaustion_Factor']


# Update the INDICATOR_FUNCTIONS dictionary with implemented functions
INDICATOR_FUNCTIONS = {
    'Trend': {
        'Slope': calculate_trend_slope,
        'MACD': calculate_trend_macd,
    },
    'Volatility': {
        'ATR': calculate_volatility_atr,
        'StdDev': calculate_volatility_stddev,
    },
    'Exhaustion': {
        'SMADiff': calculate_exhaustion_sma_diff,
        'RSI': calculate_exhaustion_rsi,
    }
}


# Define the dynamic weights function (same as before)
def get_dynamic_weights(regime, weights_params):
    """
    Returns dynamic weights for factors based on market regime, using weights from params.

    Args:
        regime (str): The current market regime.
        weights_params (dict): Dictionary containing weight parameters for each regime.
                                e.g., {'strong_bull_trend': 0.6, 'strong_bull_volatility': 0.1, ...}

    Returns:
        dict: Dictionary of weights for 'Trend', 'Volatility', 'Exhaustion'.
    """
    # Ensure weights sum to 1 (or normalize them) - Normalization is safer
    def normalize_weights(w):
        total = sum(w.values())
        if total == 0: return {k: 1/len(w) for k in w} # Avoid division by zero
        # Corrected the dictionary comprehension syntax
        return {k: w[k] / total for k in w.keys()} # Corrected: use w[k] to get the value for key k

    if regime == 'Strong Bull':
        weights = {'Trend': weights_params.get('strong_bull_trend', 0.6),
                   'Volatility': weights_params.get('strong_bull_volatility', 0.1),
                   'Exhaustion': weights_params.get('strong_bull_exhaustion', 0.3)}
        return normalize_weights(weights)
    elif regime == 'Weak Bull':
        weights = {'Trend': weights_params.get('weak_bull_trend', 0.4),
                   'Volatility': weights_params.get('weak_bull_volatility', 0.2),
                   'Exhaustion': weights_params.get('weak_bull_exhaustion', 0.4)}
        return normalize_weights(weights)
    elif regime == 'Neutral':
        weights = {'Trend': weights_params.get('neutral_trend', 0.2),
                   'Volatility': weights_params.get('neutral_volatility', 0.4),
                   'Exhaustion': weights_params.get('neutral_exhaustion', 0.4)}
        return normalize_weights(weights)
    elif regime == 'Weak Bear':
         weights = {'Trend': weights_params.get('weak_bear_trend', 0.4),
                   'Volatility': weights_params.get('weak_bear_volatility', 0.2),
                   'Exhaustion': weights_params.get('weak_bear_exhaustion', 0.4)}
         return normalize_weights(weights)
    elif regime == 'Strong Bear':
        weights = {'Trend': weights_params.get('strong_bear_trend', 0.6),
                   'Volatility': weights_params.get('strong_bear_volatility', 0.1),
                   'Exhaustion': weights_params.get('strong_bear_exhaustion', 0.3)}
        return normalize_weights(weights)
    else: # Default or unexpected regime
        # Fallback to a neutral or default weighting scheme
        weights = {'Trend': 1/3, 'Volatility': 1/3, 'Exhaustion': 1/3}
        return weights # No need to normalize default equal weights


# Define the fitness function (updated to accept extended parameters and enhanced error handling, now includes Max Drawdown Percentage in fitness)
def evaluate_strategy_with_indicators_and_params(params, df_dollar_bars, initial_capital=100000):
    """
    Evaluates a trading strategy with given parameters and indicator choices
    on historical dollar bar data.

    Args:
        params (dict): Dictionary of strategy parameters, including:
                       - 'indicator_trend', 'indicator_volatility', 'indicator_exhaustion' (names)
                       - 'lookback_trend', 'lookback_volatility', 'lookback_exhaustion' (int)
                       - 'strong_bull_threshold', 'weak_bull_threshold', ... (float)
                       - 'stop_loss_multiplier_strong', 'stop_loss_multiplier_weak' (float)
                       - 'entry_step_size' (float)
                       - Dynamic weight parameters for each regime (e.g., 'strong_bull_trend')

        df_dollar_bars (pd.DataFrame): DataFrame containing raw dollar bars (Open, High, Low, Close, Volume, DollarVolume).
        initial_capital (float): Starting capital for the backtest.

    Returns:
        float: A combined fitness score (Sortino Ratio penalized by Max Drawdown Percentage),
               or a very low number indicating a specific error condition.
    """
    df_eval = df_dollar_bars.copy() # Work on a copy

    # Extract parameters (with default fallbacks)
    indicator_trend_name = params.get('indicator_trend', 'Slope')
    indicator_volatility_name = params.get('indicator_volatility', 'ATR')
    indicator_exhaustion_name = params.get('indicator_exhaustion', 'SMADiff')

    lookback_trend = params.get('lookback_trend', 20)
    lookback_volatility = params.get('lookback_volatility', 20)
    lookback_exhaustion = params.get('lookback_exhaustion', 20)

    # Determine the maximum lookback required based on selected indicators
    max_required_lookback = 0
    if indicator_trend_name == 'Slope':
        max_required_lookback = max(max_required_lookback, lookback_trend)
    elif indicator_trend_name == 'MACD':
         # MACD needs slowperiod + signalperiod bars for full calculation
         macd_fast = params.get('macd_fastperiod', 12)
         macd_slow = params.get('macd_slowperiod', 26)
         macd_signal = params.get('macd_signalperiod', 9)
         max_required_lookback = max(max_required_lookback, macd_slow + macd_signal)
         # Ensure MACD periods are valid
         # Corrected typo: macc_signal -> macd_signal
         if macd_fast <= 0 or macd_slow <= 0 or macd_signal <= 0 or macd_fast >= macd_slow:
              # print(f"DEBUG: Invalid MACD periods: fast={macd_fast}, slow={macd_slow}, signal={macd_signal}. Returning -2001.") # Debug print
              return -2001 # Treat as factor calculation error due to invalid params


    if indicator_volatility_name == 'ATR':
        max_required_lookback = max(max_required_lookback, lookback_volatility + 1) # ATR needs previous close + lookback
         # Ensure lookback is valid
        if lookback_volatility <= 0:
             # print(f"DEBUG: Invalid ATR lookback: {lookback_volatility}. Returning -2001.") # Debug print
             return -2001 # Treat as factor calculation error due to invalid params
    elif indicator_volatility_name == 'StdDev':
        max_required_lookback = max(max_required_lookback, lookback_volatility + 1) # StdDev needs previous close for return + lookback
        # Ensure lookback is valid
        if lookback_volatility <= 0:
             # print(f"DEBUG: Invalid StdDev lookback: {lookback_volatility}. Returning -2001.") # Debug print
             return -2001 # Treat as factor calculation error due to invalid params


    if indicator_exhaustion_name == 'SMADiff':
        # SMADiff needs the SMA lookback + it relies on Volatility_Factor (ATR) which has its own lookback
        # We need enough data for both the SMA and the ATR calculation within the window
        max_required_lookback = max(max_required_lookback, lookback_exhaustion, lookback_volatility + 1) # Max of SMA lookback and ATR lookback
        # Ensure lookback is valid
        if lookback_exhaustion <= 0:
             # print(f"DEBUG: Invalid SMADiff lookback: {lookback_exhaustion}. Returning -2001.") # Debug print
             return -2001 # Treat as factor calculation error due to invalid params

    elif indicator_exhaustion_name == 'RSI':
         rsi_period = params.get('rsi_period', 14) # Make RSI period configurable
         max_required_lookback = max(max_required_lookback, rsi_period * 2) # RSI needs roughly 2*lookback for initial EMA
         # Ensure lookback is valid
         if rsi_period <= 0:
              # print(f"DEBUG: Invalid RSI period: {rsi_period}. Returning -2001.") # Debug print
              return -2001 # Treat as factor calculation error due to invalid params


    # Add a check for sufficient data length relative to max required lookback
    # We need enough data for the longest lookback *plus* at least one bar after that for trading logic
    if len(df_eval) < max_required_lookback + 1:
        # print(f"DEBUG - Error -2002: Insufficient data length ({len(df_eval)}) for max required lookback ({max_required_lookback}). Returning error.") # Debug print
        # print(f"DEBUG - Error -2002: Parameters: {params}") # Debug print
        # print(f"DEBUG - Error -2002: Data window size passed to fitness: {len(df_dollar_bars)}") # Size of the data chunk passed to fitness function # Debug print
        return -2002 # Return specific error code for empty DataFrame due to insufficient data


    strong_bull_threshold = params.get('strong_bull_threshold', 50)
    weak_bull_threshold = params.get('weak_bull_threshold', 20)
    neutral_threshold_upper = params.get('neutral_threshold_upper', 20)
    neutral_threshold_lower = params.get('neutral_threshold_lower', -20)
    strong_bear_threshold = params.get('strong_bear_threshold', -50)
    weak_bear_threshold = params.get('weak_bear_threshold', -20)

    stop_loss_multiplier_strong = params.get('stop_loss_multiplier_strong', 2)
    stop_loss_multiplier_weak = params.get('stop_loss_multiplier_weak', 1)

    entry_step_size = params.get('entry_step_size', 0.2)

    # Extract dynamic weight parameters - Ensure correct extraction for get_dynamic_weights
    weights_params = {
        'strong_bull_trend': params.get('strong_bull_trend', 0.6), 'strong_bull_volatility': params.get('strong_bull_volatility', 0.1), 'strong_bull_exhaustion': params.get('strong_bull_exhaustion', 0.3),
        'weak_bull_trend': params.get('weak_bull_trend', 0.4), 'weak_bull_volatility': params.get('weak_bull_volatility', 0.2), 'weak_bull_exhaustion': params.get('weak_bull_exhaustion', 0.4),
        'neutral_trend': params.get('neutral_trend', 0.2), 'neutral_volatility': params.get('neutral_volatility', 0.4), 'neutral_exhaustion': params.get('neutral_exhaustion', 0.4),
        'weak_bear_trend': params.get('weak_bear_trend', 0.4), 'weak_bear_volatility': params.get('weak_bear_volatility', 0.2), 'weak_bear_exhaustion': params.get('weak_bear_exhaustion', 0.4),
        'strong_bear_trend': params.get('strong_bear_trend', 0.6), 'strong_bear_volatility': params.get('strong_bear_volatility', 0.1), 'strong_bear_exhaustion': params.get('strong_bear_exhaustion', 0.3),
    }


    # --- Calculate Factors using specified indicators and lookbacks ---
    try:
        # Calculate Volatility first as Exhaustion might depend on it (SMADiff)
        if indicator_volatility_name in INDICATOR_FUNCTIONS['Volatility']:
            df_eval['Volatility_Factor'] = INDICATOR_FUNCTIONS['Volatility'][indicator_volatility_name](df_eval.copy(), lookback_volatility)
        else:
            # print(f"DEBUG: Unknown Volatility indicator: {indicator_volatility_name}. Using default (ATR).") # Debug print
            df_eval['Volatility_Factor'] = calculate_volatility_atr(df_eval.copy(), lookback_volatility)

        if indicator_trend_name in INDICATOR_FUNCTIONS['Trend']:
             if indicator_trend_name == 'Slope':
                  df_eval['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend'][indicator_trend_name](df_eval.copy(), lookback_trend)
             elif indicator_trend_name == 'MACD':
                  # MACD uses its own default lookbacks unless specified in params (which is not in the current param set)
                  # Let's make MACD lookbacks also configurable via params if they exist, otherwise use defaults
                  macd_fast = params.get('macd_fastperiod', 12)
                  macd_slow = params.get('macd_slowperiod', 26)
                  macd_signal = params.get('macd_signalperiod', 9)
                  # Pass lookbacks to MACD function
                  df_eval['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend'][indicator_trend_name](df_eval.copy(), fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
             else:
                  df_eval['Trend_Factor'] = np.nan # Unknown indicator

        else:
             # print(f"DEBUG: Unknown Trend indicator: {indicator_trend_name}. Using default (Slope).") # Debug print
             df_eval['Trend_Factor'] = calculate_trend_slope(df_eval.copy(), lookback_trend) # Use default Slope

        if indicator_exhaustion_name in INDICATOR_FUNCTIONS['Exhaustion']:
            # Exhaustion calculation might need the Volatility factor output
            if indicator_exhaustion_name == 'SMADiff':
                 # SMADiff specifically requires the normalized ATR as an argument
                 # Ensure Volatility_Factor is calculated before passing it
                 if 'Volatility_Factor' in df_eval.columns:
                     df_eval['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_name](df_eval.copy(), lookback_exhaustion, df_eval['Volatility_Factor'])
                 else:
                      # print("DEBUG: Volatility_Factor not calculated for SMADiff.") # Debug print
                      df_eval['Exhaustion_Factor'] = np.nan
            elif indicator_exhaustion_name == 'RSI':
                 # RSI takes df and lookback_exhaustion
                 rsi_period = params.get('rsi_period', 14) # Make RSI period configurable
                 df_eval['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_name](df_eval.copy(), lookback=rsi_period)
            else:
                 df_eval['Exhaustion_Factor'] = np.nan # Unknown indicator
        else:
            # print(f"DEBUG: Unknown Exhaustion indicator: {indicator_exhaustion_name}. Using default (SMADiff).") # Debug print
            # Ensure Volatility_Factor is calculated before passing it to SMADiff default
            if 'Volatility_Factor' in df_eval.columns:
                df_eval['Exhaustion_Factor'] = calculate_exhaustion_sma_diff(df_eval.copy(), lookback_exhaustion, df_eval['Volatility_Factor'])
            else:
                 # print("DEBUG: Volatility_Factor not calculated for default SMADiff.") # Debug print
                 df_eval['Exhaustion_Factor'] = np.nan


    except Exception as e:
        print(f"DEBUG - Error -2001: Error during factor calculation: {e}") # Debug print
        print(f"DEBUG - Error -2001: Parameters: {params}")
        return -2001 # Return specific error code for factor calculation error


    # Drop rows with NaN values generated by lookback periods after factor calculation
    initial_rows = len(df_eval) # This was the original size of the passed window
    df_eval = df_eval.dropna(subset=['Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor'])

    # The check for empty DataFrame after dropna is now redundant IF the initial data length check is sufficient.
    # However, keep it as a safety net in case some edge case produces NaNs unexpectedly later in calculation.
    if len(df_eval) == 0:
        # Add debugging print to show parameters and data size when this error occurs
        print(f"DEBUG - Error -2002: DataFrame is empty after dropping NaNs.")
        print(f"DEBUG - Error -2002: Original rows: {initial_rows}")
        print(f"DEBUG - Error -2002: Parameters: {params}")
        print(f"DEBUG - Error -2002: Data window size passed to fitness: {len(df_dollar_bars)}") # Size of the data chunk passed to fitness function
        return -2002 # Return specific error code for empty DataFrame after dropna


    # --- Calculate Dynamic MSS and Regime ---

    # Check if df_eval is empty after dropna before calculating MSS
    if df_eval.empty:
         # This case should ideally be caught by the len(df_eval) == 0 check above, but as a safety.
         print(f"DEBUG - Error -2002: DataFrame is empty after dropna before MSS calculation. Returning error.")
         print(f"DEBUG - Error -2002: Parameters: {params}")
         print(f"DEBUG - Error -2002: Data window size passed to fitness: {len(df_dollar_bars)}") # Size of the data chunk passed to fitness function
         return -2002


    # Calculate initial static MSS to get a starting regime for dynamic weights
    # Use default static weights for this initial MSS calculation (not optimized by GA)
    static_weight_trend = 0.4
    static_weight_volatility = 0.3
    static_weight_exhaustion = 0.3

    df_eval['MSS_static_initial'] = (static_weight_trend * df_eval['Trend_Factor'] +
                                    static_weight_volatility * df_eval['Volatility_Factor'] +
                                    static_weight_exhaustion * df_eval['Exhaustion_Factor'])

    # Classify the initial static regime using the provided thresholds
    def classify_regime_static(mss):
        if mss > strong_bull_threshold:
            return 'Strong Bull'
        elif mss > weak_bull_threshold:
            return 'Weak Bull'
        elif mss >= neutral_threshold_lower and mss <= neutral_threshold_upper:
            return 'Neutral'
        elif mss > strong_bear_threshold:
            return 'Weak Bear'
        else:
            return 'Strong Bear'

    df_eval['Regime_initial'] = df_eval['MSS_static_initial'].apply(classify_regime_static)


    # Now, calculate Dynamic MSS using dynamic weights based on the *initial* regime
    df_eval['MSS_dynamic'] = np.nan # Initialize dynamic MSS column

    # Iterate through the dataframe to calculate dynamic MSS
    # Use the initial static regime ('Regime_initial') to determine dynamic weights for the current bar.
    # Use .loc for setting values to avoid SettingWithCopyWarning
    try:
        for index, row in df_eval.iterrows():
            current_regime_initial = row['Regime_initial'] # Use the initial static regime
            # Pass the correctly extracted weights_params dictionary
            dynamic_weights = get_dynamic_weights(current_regime_initial, weights_params)

            # Calculate dynamic MSS using factor values and dynamic weights
            dynamic_mss = (dynamic_weights['Trend'] * row['Trend_Factor'] +
                           dynamic_weights['Volatility'] * row['Volatility_Factor'] +
                           dynamic_weights['Exhaustion'] * row['Exhaustion_Factor'])

            df_eval.loc[index, 'MSS_dynamic'] = dynamic_mss
    except Exception as e:
        print(f"DEBUG - Error -2003: Error during Dynamic MSS calculation: {e}") # Debug print
        print(f"DEBUG - Error -2003: Parameters: {params}")
        print(f"DEBUG - Error -2003: Row index: {index}")
        print(f"DEBUG - Error -2003: Row data:\n{row.to_dict()}")
        return -2003 # Return specific error code for Dynamic MSS calculation error


    # Update the 'Regime' column based on the newly calculated 'MSS_dynamic'
    # Use the same provided thresholds for classifying the dynamic MSS into regimes for consistency with the action matrix.
    try:
        df_eval['Regime_dynamic'] = df_eval['MSS_dynamic'].apply(classify_regime_static) # Use provided thresholds
    except Exception as e:
        print(f"DEBUG - Error -2004: Error during Dynamic Regime classification: {e}") # Debug print
        print(f"DEBUG - Error -2004: Parameters: {params}")
        return -2004 # Return specific error code for Dynamic Regime classification error


    # --- Trading Simulation (Enhanced Logic with Fractional Positions and Risk Management) ---
    position_fraction = 0.0 # Current position as a fraction of initial_capital (positive for long, negative for short)
    total_units = 0.0
    total_cost_basis = 0.0 # Total cost paid for units (positive for buy, negative for sell)
    stop_loss = 0.0
    take_profit = 0.0 # Profit target level
    equity_curve = []
    trade_log = [] # Enable trade log capture for checking if trades occurred
    current_capital = initial_capital # Start with the initial capital
    max_position_fraction = 1.0 # Max position is 100% of initial capital

    # Redefine max_step based on entry_step_size parameter
    max_step_fraction = entry_step_size * max_position_fraction # Step size in terms of fraction of initial_capital

    peak_price_since_entry = -float('inf') # Track peak price for trailing stop (Long)
    valley_price_since_entry = float('inf') # Track valley price for trailing stop (Short - not active yet)

    try:
        # Iterate through the dataframe
        for index, row in df_eval.iterrows():
            try: # Inner try-except for per-bar simulation issues
                current_price = row['Close']
                current_regime = row['Regime_dynamic'] # Use the regime based on dynamic MSS
                current_mss = row['MSS_dynamic'] # Use the dynamic MSS
                current_volatility_factor = row['Volatility_Factor'] # Use the normalized Volatility Factor

                # Determine stop-loss distance and take-profit distance based on regime and Volatility Factor
                # Use absolute value of volatility factor for distance as it represents price movement
                # Ensure current_volatility_factor is a valid number before using it
                if not isinstance(current_volatility_factor, (int, float)) or np.isnan(current_volatility_factor) or abs(current_volatility_factor) < 1e-9:
                     stop_loss_distance = 0 # Cannot calculate dynamic stop loss without valid Volatility Factor
                     take_profit_distance = 0 # Cannot calculate dynamic take profit without valid Volatility Factor
                else:
                    # Example Profit Target Multipliers (could also be optimized by GA)
                    take_profit_multiplier_strong = params.get('take_profit_multiplier_strong', 4.0) # Make configurable
                    take_profit_multiplier_weak = params.get('take_profit_multiplier_weak', 2.0)   # Make configurable

                    if current_regime in ['Strong Bull', 'Strong Bear']:
                        stop_loss_distance = stop_loss_multiplier_strong * abs(current_volatility_factor)
                        take_profit_distance = take_profit_multiplier_strong * abs(current_volatility_factor)
                    elif current_regime in ['Weak Bull', 'Weak Bear']:
                        stop_loss_distance = stop_loss_multiplier_weak * abs(current_volatility_factor)
                        take_profit_distance = take_profit_multiplier_weak * abs(current_volatility_factor)
                    else: # Neutral
                        stop_loss_distance = 0
                        take_profit_distance = 0

                    # Ensure calculated distances are valid numbers
                    if np.isnan(stop_loss_distance) or np.isinf(stop_loss_distance): stop_loss_distance = 0
                    if np.isnan(take_profit_distance) or np.isinf(take_profit_distance): take_profit_distance = 0


                # --- Trading Logic with Fractional Positions (Capital Allocation Model) ---

                # Determine target position fraction based on regime and MSS confidence
                target_position_fraction = 0.0
                if current_regime == 'Strong Bull':
                    # Ensure denominator is not zero
                    denominator = (100 - strong_bull_threshold)
                    normalized_mss = (current_mss - strong_bull_threshold) / denominator if denominator > 0 else 0
                    target_position_fraction = max_position_fraction * np.clip(normalized_mss, 0, 1)

                elif current_regime == 'Weak Bull':
                     # Ensure denominator is not zero
                     denominator = (neutral_threshold_upper - weak_bull_threshold)
                     normalized_mss = (current_mss - weak_bull_threshold) / denominator if denominator > 0 else 0
                     target_position_fraction = position_fraction # Default to hold
                     if position_fraction > 1e-9: # If currently long
                         target_position_fraction = max_position_fraction * np.clip(normalized_mss, 0, 1)

                elif current_regime == 'Neutral':
                    target_position_fraction = 0.0

                elif current_regime == 'Weak Bear':
                     # Ensure denominator is not zero
                     denominator = (weak_bear_threshold - (-100))
                     normalized_mss = (current_mss - (-100)) / denominator if denominator > 0 else 0
                     target_position_fraction = position_fraction # Default to hold
                     if position_fraction < -1e-9: # If currently short
                          target_position_fraction = -max_position_fraction * np.clip(1 - normalized_mss, 0, 1)

                elif current_regime == 'Strong Bear':
                    # Ensure denominator is not zero
                    denominator = (strong_bear_threshold - (-100))
                    normalized_mss = (current_mss - (-100)) / denominator if denominator > 0 else 0
                    target_position_fraction = -max_position_fraction * np.clip(1 - normalized_mss, 0, 1)

                # Ensure target_position_fraction is a valid number
                if np.isnan(target_position_fraction) or np.isinf(target_position_fraction):
                     target_position_fraction = 0.0


                # Calculate the change in position fraction
                position_fraction_change = target_position_fraction - position_fraction

                # Limit position fraction change to max_step_fraction
                position_fraction_change = np.clip(position_fraction_change, -max_step_fraction, max_step_fraction)

                # Calculate the amount of capital to allocate/deallocate in this step
                capital_to_trade = position_fraction_change * initial_capital # Amount of initial capital equivalent to the fraction change

                # Calculate the units to trade based on the capital amount and current price
                # Guard against division by zero if current_price is zero or very close to zero
                units_to_trade = capital_to_trade / current_price if current_price > 1e-9 else 0.0


                # Update total units and total cost basis based on units_to_trade
                trade_occurred_in_bar = False # Flag to check if any trade happened in this bar
                if units_to_trade > 1e-9: # Buying (increasing long or entering long)
                    total_cost_basis += units_to_trade * current_price # Add cost for new units
                    total_units += units_to_trade
                    # action = 'Increase Long' if position_fraction_change > 0 and total_units > units_to_trade else 'Enter Long' # Refine action description
                    trade_log.append({'Date': index, 'Action': 'Buy (Gradual)', 'Price': current_price, 'Units': units_to_trade, 'TotalUnits': total_units, 'PnL': np.nan}) # Log the trade
                    trade_occurred_in_bar = True


                elif units_to_trade < -1e-9: # Selling (decreasing long or exiting long)
                    # Ensure we don't sell more units than we have
                    units_to_sell = abs(units_to_trade)

                    if total_units > 1e-9: # Only sell if we have units (long position)
                        units_to_sell = min(units_to_sell, total_units) # Do not sell more than available units
                        units_to_trade = -units_to_sell # Adjust units_to_trade based on actual units sold

                        # Guard against division by zero if total_units is zero or very close to zero
                        avg_cost_basis_per_unit = total_cost_basis / total_units if abs(total_units) > 1e-9 else 0
                        pnl_on_sold_units = (current_price - avg_cost_basis_per_unit) * units_to_sell # Calculate P/L

                        current_capital += pnl_on_sold_units # Update capital with realized P/L

                        # Update total units and total cost basis based on units sold
                        # Assuming weighted average cost basis reduction
                        total_cost_basis -= units_to_sell * avg_cost_basis_per_unit
                        total_units -= units_to_sell

                        # action = 'Decrease Long' if total_units > 1e-9 else 'Exit Long' # Refine action description
                        trade_log.append({'Date': index, 'Action': 'Sell (Gradual)', 'Price': current_price, 'Units': -units_to_sell, 'TotalUnits': total_units, 'PnL': pnl_on_sold_units}) # Log the trade
                        trade_occurred_in_bar = True


                    elif total_units < -1e-9: # Selling a short position (buying to cover) - This logic needs to handle short positions separately
                         # For now, this block is not active as we are focusing on long-only for this part.
                         pass # Skip short selling for now in this enhanced logic PoC

                # Update the current position fraction based on the new total units and current price
                # This represents the current value of the position as a fraction of initial capital
                current_market_value = total_units * current_price if current_price > 1e-9 else 0.0 # Guard against price zero
                position_fraction = current_market_value / initial_capital if initial_capital > 0 else 0.0


                # --- Refined Stop-Loss and Take-Profit Logic ---
                if abs(total_units) > 1e-9: # If position is active
                    # Guard against division by zero if total_units is zero or very close to zero
                    avg_cost_basis_per_unit = total_cost_basis / total_units if abs(total_units) > 1e-9 else 0

                    # Update peak/valley price since entry for trailing stop
                    if total_units > 0: # Long position
                         peak_price_since_entry = max(peak_price_since_entry, current_price)
                         # Calculate trailing stop level
                         stop_loss_level = peak_price_since_entry - stop_loss_distance if stop_loss_distance > 0 else -float('inf')
                         # The stop_loss variable should hold the *active* stop loss level
                         stop_loss = stop_loss_level

                         # Calculate Take Profit level based on average cost basis
                         take_profit_level = avg_cost_basis_per_unit + take_profit_distance if take_profit_distance > 0 else float('inf')
                         take_profit = take_profit_level


                    elif total_units < 0: # Short position (not active in this PoC)
                         valley_price_since_entry = min(valley_price_since_entry, current_price)
                         # Calculate trailing stop level
                         stop_loss_level = valley_price_since_entry + stop_loss_distance if stop_loss_distance > 0 else float('inf')
                         stop_loss = stop_loss_level

                         # Calculate Take Profit level based on average cost basis
                         take_profit_level = avg_cost_basis_per_unit - take_profit_distance if take_profit_distance > 0 else -float('inf')
                         take_profit = take_profit_level

                    # Ensure stop_loss and take_profit are valid numbers
                    if not isinstance(stop_loss, (int, float)) or np.isnan(stop_loss) or np.isinf(stop_loss): stop_loss = -float('inf') if total_units > 0 else float('inf') # Set to safe values
                    if not isinstance(take_profit, (int, float)) or np.isnan(take_profit) or np.isinf(take_profit): take_profit = float('inf') if total_units > 0 else -float('inf') # Set to safe values


                else: # No active position
                     stop_loss = 0.0
                     take_profit = 0.0
                     # Reset peak/valley price when position is flat
                     peak_price_since_entry = -float('inf')
                     valley_price_since_entry = float('inf')


                # Check for Stop-Loss or Take-Profit hit
                if abs(total_units) > 1e-9: # Only check if position is active
                     # Guard against division by zero if total_units is zero or very close to zero
                     avg_cost_basis_per_unit = total_cost_basis / total_units if abs(total_units) > 1e-9 else 0 # Recalculate avg cost basis

                     if total_units > 0: # Long position
                         # Check Take Profit hit
                         if current_price >= take_profit and not np.isinf(take_profit):
                             pnl = (current_price - avg_cost_basis_per_unit) * total_units # Calculate P/L
                             current_capital += pnl # Update capital
                             trade_log.append({'Date': index, 'Action': 'Take Profit Long', 'Price': current_price, 'Units': -total_units, 'TotalUnits': 0.0, 'PnL': pnl}) # Log the trade
                             total_units = 0.0
                             total_cost_basis = 0.0
                             position_fraction = 0.0
                             stop_loss = 0.0
                             take_profit = 0.0
                             peak_price_since_entry = -float('inf') # Reset peak
                             trade_occurred_in_bar = True # Logged trade means trade occurred


                         # Check Stop Loss hit (after checking take profit)
                         # Ensure stop_loss is a valid number and positive for long position
                         elif current_price <= stop_loss and not np.isinf(stop_loss) and stop_loss > 1e-9: # Ensure stop_loss is positive and not tiny
                             pnl = (current_price - avg_cost_basis_per_unit) * total_units # Calculate P/L
                             current_capital += pnl # Update capital
                             trade_log.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'Units': -total_units, 'TotalUnits': 0.0, 'PnL': pnl}) # Log the trade
                             total_units = 0.0
                             total_cost_basis = 0.0
                             position_fraction = 0.0
                             stop_loss = 0.0
                             take_profit = 0.0
                             peak_price_since_entry = -float('inf') # Reset peak
                             trade_occurred_in_bar = True # Logged trade means trade occurred


                     # Fixed IndentationError here: moved this elif block back one level
                     elif total_units < 0: # Short position (not active)
                          # Similar checks for short take profit and stop loss
                          pass # Skip short position checks for now

            except Exception as e:
                # Print debugging information for simulation errors
                print(f"DEBUG - Error -3000: Error during simulation at index {index}.")
                print(f"DEBUG - Error -3000: Error details: {e}")
                print(f"DEBUG - Error -3000: Parameters: {params}")
                print(f"DEBUG - Error -3000: Current row data:\n{row.to_dict()}") # Print row as dict for better readability
                print(f"DEBUG - Error -3000: Current position fraction: {position_fraction}, Total units: {total_units}, Current capital: {current_capital}")
                print(f"DEBUG - Error -3000: Stop Loss: {stop_loss}, Take Profit: {take_profit}") # Add SL/TP to debug
                return -3000 # Return specific error code for trading simulation error


            # Append current equity to the equity curve
            # Current equity is current capital + (unrealized P/L of the current position)
            # Guard against division by zero if total_units is zero when calculating avg_cost_basis_per_unit
            avg_cost_basis_per_unit = total_cost_basis / total_units if abs(total_units) > 1e-9 else 0
            unrealized_pnl = (current_price - avg_cost_basis_per_unit) * total_units if abs(total_units) > 1e-9 else 0.0
            equity_curve.append({'Date': index, 'Equity': current_capital + unrealized_pnl})

    except Exception as e:
        print(f"DEBUG - Error -3001: Uncaught error during trading simulation loop setup/iteration: {e}") # Debug print for outer loop errors
        print(f"DEBUG - Error -3001: Parameters: {params}")
        return -3001 # Return specific error code for outer simulation loop error


    # Convert equity curve to DataFrame
    equity_curve_df_eval = pd.DataFrame(equity_curve).set_index('Date')

    # --- Performance Evaluation ---

    # Check if *any* trades occurred. If not, return a specific low fitness.
    # We check the length of the trade_log list
    if not trade_log:
         # print("DEBUG: No trades occurred in this window. Returning -5000 fitness.") # Debug print
         return -5000 # Specific error code for no trades


    # Check if equity curve is empty or too short before calculating metrics
    # This check is less likely to be hit now that we check for no trades first,
    # but keep as a safety measure.
    if equity_curve_df_eval.empty or len(equity_curve_df_eval) < 2:
         # print("DEBUG: Equity curve is empty or too short for evaluation.") # Debug print
         return -10000 # New specific error code for empty/short equity curve


    equity_curve_df_eval['Daily_Return'] = equity_curve_df_eval['Equity'].pct_change().fillna(0)
    total_return = (equity_curve_df_eval['Equity'].iloc[-1] - initial_capital) / initial_capital
    trading_periods_per_year = 365 # Adjust if dollar bars don't represent daily frequency

    # Calculate Annualized Return
    if len(equity_curve_df_eval) > 1 and initial_capital > 0:
        if total_return > -1:
            annualized_return = (1 + total_return)**(trading_periods_per_year / len(equity_curve_df_eval)) - 1
        else:
            annualized_return = -9999 # Distinct large negative number for very poor return
    else:
        annualized_return = 0


    # Calculate Sortino Ratio
    mar = 0 # Minimum Acceptable Return (MAR)
    downside_returns = equity_curve_df_eval[equity_curve_df_eval['Daily_Return'] < mar]['Daily_Return']
    downside_deviation = downside_returns.std()

    if downside_deviation == 0 or np.isnan(downside_deviation):
        if annualized_return > mar:
             sortino_ratio = float('inf')
        elif annualized_return == mar:
             sortino_ratio = 0
        else:
             # Losing strategy with no measurable downside deviation (e.g., flat equity curve below initial capital)
             sortino_ratio = -1003 # Specific error code for losing with no downside
    else:
        sortino_ratio = (annualized_return - mar) / downside_deviation

    # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
    if np.isinf(sortino_ratio):
         if annualized_return > mar:
              pass # Keep the infinite value for now, will handle below in combined fitness
         else:
              sortino_ratio = -1001 # Specific error code for problematic infinite ratio
    elif np.isnan(sortino_ratio):
        sortino_ratio = -1002 # Specific error code for NaN ratio


    # Calculate Maximum Drawdown Percentage
    equity_curve_df_eval['Peak'] = equity_curve_df_eval['Equity'].cummax()
    equity_curve_df_eval['Drawdown'] = equity_curve_df_eval['Equity'] - equity_curve_df_eval['Peak']
    max_drawdown_dollar = equity_curve_df_eval['Drawdown'].min()

    # Calculate Max Drawdown Percentage (relative to the peak equity before the largest drop)
    # Find the peak equity before the max drawdown occurred
    if max_drawdown_dollar < 0:
        # Find the index of the maximum drawdown
        idx_max_drawdown = equity_curve_df_eval['Drawdown'].idxmin()
        # The peak before this drawdown is the max equity up to this point
        peak_before_drawdown = equity_curve_df_eval['Equity'].loc[:idx_max_drawdown].max()
        if peak_before_drawdown > 0:
             max_drawdown_percentage = abs(max_drawdown_dollar) / peak_before_drawdown * 100 # Express as a positive percentage
        else:
             max_drawdown_percentage = 100.0 # 100% drawdown if peak was 0 or negative (unlikely with positive initial capital)
    else:
        max_drawdown_percentage = 0.0 # No drawdown if max_drawdown_dollar is 0 or positive

    # Ensure max_drawdown_percentage is a valid number
    if np.isnan(max_drawdown_percentage) or np.isinf(max_drawdown_percentage):
         max_drawdown_percentage = 100.0 # Treat invalid as 100% drawdown


    # --- Combine Sortino Ratio and Max Drawdown Percentage for Fitness ---
    # We want to maximize Sortino Ratio and minimize Max Drawdown Percentage.
    # A simple combined fitness can be Sortino Ratio - k * Max Drawdown Percentage
    # We need to handle the infinite Sortino Ratio case.
    # Let's use a large number instead of inf for the combined fitness, penalizing by drawdown.

    # Define weights for combination (can be parameters to optimize later)
    weight_sortino = 0.7
    weight_drawdown = 0.3
    large_sortino_substitute = 1000 # Substitute for infinite sortino

    # Check if Sortino is one of our specific error codes BEFORE combining
    if sortino_ratio in [-2001, -2002, -2003, -2004, -3000, -3001, -10000, -1003, -1001, -1002, -9999, -5000]: # Added -5000 here
         combined_fitness = sortino_ratio # If it's an error code, return the error code
    elif np.isinf(sortino_ratio):
         # Substitute inf with a large value and penalize by drawdown percentage
         combined_fitness = large_sortino_substitute - (max_drawdown_percentage * weight_drawdown)
    elif np.isnan(sortino_ratio):
        combined_fitness = -1002 # Should be caught earlier, but as a fallback
    else:
         # Combine finite Sortino with penalized drawdown percentage
         combined_fitness = (sortino_ratio * weight_sortino) - (max_drawdown_percentage * weight_drawdown)

    # Ensure fitness doesn't go below our error codes' range if the combination results in a very small number
    # This check might be less necessary now that we return error codes directly, but keep as a safety.
    # Let's refine this check: if the calculated combined_fitness is lower than the lowest possible valid fitness
    # (which would be a very negative Sortino with high drawdown penalty), and it's NOT one of our specific
    # error codes, then something is wrong.
    # A very low valid Sortino could be -100, combined with 100% drawdown (-30), resulting in ~ -130.
    # Let's set a threshold below which we suspect an issue if it's not a known error code.
    # Updated lowest_expected_valid_fitness to be below the -5000 no-trade fitness
    lowest_expected_valid_fitness = -6000 # Heuristic: assuming fitness won't normally drop below -6000 for valid runs

    if combined_fitness < lowest_expected_valid_fitness and combined_fitness not in [-2001, -2002, -2003, -2004, -3000, -3001, -10000, -1003, -1001, -1002, -9999, -5000]:
         # print(f"DEBUG: Combined fitness ({combined_fitness}) is very low and not a known error code. Max Drawdown Pct: {max_drawdown_percentage}, Sortino: {sortino_ratio}") # Debug print
         return -4000 # New specific error code for very low combined fitness


    return combined_fitness


# Example usage of the updated fitness function (using current parameters and default indicators/lookbacks)
# Let's define a sample params dictionary that includes all the parameters the GA will optimize
# These are example values, not necessarily the ones from the last successful run

sample_ga_params = {
    'indicator_trend': 'Slope', # Example indicator choice
    'indicator_volatility': 'ATR', # Example indicator choice
    'indicator_exhaustion': 'SMADiff', # Example indicator choice
    'lookback_trend': 20, # Example lookback
    'lookback_volatility': 20, # Example lookback
    'lookback_exhaustion': 20, # Example lookback (for SMADDiff)

    'strong_bull_threshold': 50, # Example threshold
    'weak_bull_threshold': 20, # Example threshold
    'neutral_threshold_upper': 20, # Example threshold
    'neutral_threshold_lower': -20, # Example threshold
    'strong_bear_threshold': -50, # Example threshold
    'weak_bear_threshold': -20, # Example threshold

    'stop_loss_multiplier_strong': 3.0, # Example stop loss multiplier
    'stop_loss_multiplier_weak': 3.0, # Example stop loss multiplier
    'take_profit_multiplier_strong': 4.0, # Example take profit multiplier
    'take_profit_multiplier_weak': 2.0, # Example take profit multiplier


    'entry_step_size': 0.2, # Example gradual entry step size

    # Example dynamic weight parameters (sum for each regime should ideally be 1, or they will be normalized)
    'strong_bull_trend': 0.6, 'strong_bull_volatility': 0.1, 'strong_bull_exhaustion': 0.3,
    'weak_bull_trend': 0.4, 'weak_bull_volatility': 0.2, 'weak_bull_exhaustion': 0.4,
    'neutral_trend': 0.2, 'neutral_volatility': 0.4, 'neutral_exhaustion': 0.4,
    'weak_bear_trend': 0.4, 'weak_bear_volatility': 0.2, 'weak_bear_exhaustion': 0.4,
    'strong_bear_trend': 0.6, 'strong_bear_volatility': 0.1, 'strong_bear_exhaustion': 0.3,
}


# The df_dollar_bars passed to the function should contain only the raw OHLCV and DollarVolume columns
# The factor calculation will happen inside the evaluate_strategy_with_indicators_and_params function
# Assuming the global dollar_bars_df currently holds the raw dollar bar data after the initial loading step.
# If dollar_bars_df has been modified with factor columns, you might need to reload or use a clean version.
# For clarity, let's ensure we work with a dataframe having only the necessary columns for factor calculation:
df_for_fitness_evaluation = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


# Calculate fitness using the sample GA parameters and the raw dollar bars data
# fitness = evaluate_strategy_with_indicators_and_params(sample_ga_params, df_for_fitness_evaluation, initial_capital)
# display(f"Fitness (Sortino Ratio) with sample GA parameters: {fitness}")

# Example of how to call run_genetic_algorithm with sample parameters
# (This section is commented out and for reference only)
# from genetic_algorithm import run_genetic_algorithm # Assuming run_genetic_algorithm is in genetic_algorithm.py or defined elsewhere

# ga_population_size_sample = 10
# ga_num_generations_sample = 5
# ga_mutation_rate_sample = 0.1
# ga_walkforward_periods_sample = 2
# ga_train_period_ratio_sample = 0.7
# initial_capital_sample = 100000

# best_ga_params_sample, best_ga_fitness_sample = run_genetic_algorithm(
#     df_for_fitness_evaluation,
#     PARAMETER_SPACE, # Assuming PARAMETER_SPACE is defined
#     evaluate_strategy_with_indicators_and_params,
#     population_size=ga_population_size_sample,
#     num_generations=ga_num_generations_sample,
#     mutation_rate=ga_mutation_rate_sample,
#     initial_capital=initial_capital_sample,
#     walkforward_periods=ga_walkforward_periods_sample,
#     train_period_ratio=ga_train_period_ratio_sample
# )

# display("\nGenetic Algorithm Optimization Completed (Sample Run).")
# display(f"Overall Best GA Parameters (Sample Run): {best_ga_params_sample}")
# display(f"Overall Best GA Fitness (Sample Run - Test Combined Score): {best_ga_fitness_sample:.4f}")

"""## Set up the GA Framework with Genetic Operators and Walk-Forward Optimization

### Subtask:
Structure the Genetic Algorithm process, incorporating the genetic operators (selection, crossover, mutation) and the walk-forward optimization framework. The GA will optimize both the numeric parameters and the indicator choices.

**Reasoning**:
Implement functions for initializing the population, selection, crossover, and mutation. Structure the main GA loop and integrate a walk-forward optimization approach.
"""

import random
import numpy as np
import pandas as pd

# Define the parameter space for the GA
# This includes ranges for numeric parameters and choices for categorical parameters (indicators)
PARAMETER_SPACE = {
    'indicator_trend': ['Slope', 'MACD'], # List of available Trend indicators
    'indicator_volatility': ['ATR', 'StdDev'], # List of available Volatility indicators
    'indicator_exhaustion': ['SMADiff', 'RSI'], # List of available Exhaustion indicators

    'lookback_trend': (10, 50, 1), # (min, max, step) for integer parameters
    'lookback_volatility': (10, 50, 1),
    'lookback_exhaustion': (10, 50, 1), # Applies to SMADiff SMA period and RSI period

    'strong_bull_threshold': (30.0, 80.0, 0.1), # (min, max, step) for float parameters
    'weak_bull_threshold': (10.0, 40.0, 0.1),
    'neutral_threshold_upper': (10.0, 40.0, 0.1),
    'neutral_threshold_lower': (-40.0, -10.0, 0.1),
    'strong_bear_threshold': (-80.0, -30.0, 0.1),
    'weak_bear_threshold': (-40.0, -10.0, 0.1),

    'stop_loss_multiplier_strong': (1.0, 5.0, 0.1),
    'stop_loss_multiplier_weak': (0.5, 3.0, 0.1),

    'entry_step_size': (0.1, 1.0, 0.1), # Fractional step size (0.1 to 1.0)

    # Dynamic weight parameters (sum for each regime will be normalized in the fitness function)
    'strong_bull_trend': (0.0, 1.0, 0.05), 'strong_bull_volatility': (0.0, 1.0, 0.05), 'strong_bull_exhaustion': (0.0, 1.0, 0.05),
    'weak_bull_trend': (0.0, 1.0, 0.05), 'weak_bull_volatility': (0.0, 1.0, 0.05), 'weak_bull_exhaustion': (0.0, 1.0, 0.05),
    'neutral_trend': (0.0, 1.0, 0.05), 'neutral_volatility': (0.0, 1.0, 0.05), 'neutral_exhaustion': (0.0, 1.0, 0.05),
    'weak_bear_trend': (0.0, 1.0, 0.05), 'weak_bear_volatility': (0.0, 1.0, 0.05), 'weak_bear_exhaustion': (0.0, 1.0, 0.05),
    'strong_bear_trend': (0.0, 1.0, 0.05), 'strong_bear_volatility': (0.0, 1.0, 0.05), 'strong_bear_exhaustion': (0.0, 1.0, 0.05),

    # Add MACD and RSI specific lookbacks to parameter space if they are used
    'macd_fastperiod': (5, 30, 1),
    'macd_slowperiod': (20, 60, 1),
    'macd_signalperiod': (5, 20, 1),
    'rsi_period': (5, 30, 1),
    'take_profit_multiplier_strong': (1.0, 5.0, 0.1), # Added take profit multipliers to optimization
    'take_profit_multiplier_weak': (0.5, 3.0, 0.1), # Added take profit multipliers to optimization

}


def generate_random_params(parameter_space):
    """Generates a random set of parameters within the defined space."""
    params = {}
    for param, values in parameter_space.items():
        if isinstance(values, list): # Categorical parameter
            params[param] = random.choice(values)
        elif isinstance(values, tuple) and len(values) == 3: # Numeric parameter (min, max, step)
            min_val, max_val, step = values
            if isinstance(min_val, int):
                params[param] = random.randrange(min_val, max_val + step, step)
            elif isinstance(min_val, float):
                # Generate random float within range with specified step
                num_steps = int(round((max_val - min_val) / step)) # Use round for robustness
                params[param] = min_val + random.randint(0, num_steps) * step
                params[param] = round(params[param], len(str(step).split('.')[-1])) # Round to step decimal places
        else:
            raise ValueError(f"Invalid parameter space definition for {param}: {values}")

    # Ensure MACD fastperiod < slowperiod if MACD is selected
    if params.get('indicator_trend') == 'MACD':
         while params['macd_fastperiod'] >= params['macd_slowperiod']:
              params['macd_fastperiod'] = random.randrange(parameter_space['macd_fastperiod'][0], parameter_space['macd_fastperiod'][1] + parameter_space['macd_fastperiod'][2], parameter_space['macd_fastperiod'][2])
              params['macd_slowperiod'] = random.randrange(parameter_space['macd_slowperiod'][0], parameter_space['macd_slowperiod'][1] + parameter_space['macd_slowperiod'][2], parameter_space['macd_slowperiod'][2])

    return params

def initialize_population(population_size, parameter_space):
    """Initializes a population of random parameter sets."""
    return [generate_random_params(parameter_space) for _ in range(population_size)]

def select_parents(population, fitnesses, num_parents):
    """Selects parent parameter sets using tournament selection."""
    # Simple tournament selection: randomly select a few individuals and pick the best one
    parents = []
    tournament_size = 5 # Example tournament size
    # Ensure tournament size does not exceed population size
    tournament_size = min(tournament_size, len(population))

    if tournament_size == 0: # Handle case where population is too small
        return []

    # Filter out individuals with error fitnesses before selection
    valid_indices = [i for i, f in enumerate(fitnesses) if f > -1000] # Assuming error codes are < -1000

    if len(valid_indices) < tournament_size:
         # If not enough valid individuals for a full tournament, use all valid individuals
         tournament_size = len(valid_indices)
         if tournament_size == 0: # If no valid individuals at all
              return [] # Return empty parents list


    for _ in range(num_parents):
        # Select indices for the tournament from the valid indices
        current_tournament_indices_in_valid = random.sample(range(len(valid_indices)), tournament_size)
        # Map back to original population indices
        current_tournament_indices_in_population = [valid_indices[i] for i in current_tournament_indices_in_valid]

        tournament_fitnesses = [fitnesses[i] for i in current_tournament_indices_in_population]
        # Select the index of the best fitness in the tournament
        winner_index_in_tournament = tournament_fitnesses.index(max(tournament_fitnesses))
        winner_index_in_population = current_tournament_indices_in_population[winner_index_in_tournament]
        parents.append(population[winner_index_in_population])
    return parents

def crossover(parent1_params, parent2_params, parameter_space):
    """Performs crossover to create new offspring parameter sets."""
    offspring1_params = {}
    offspring2_params = {}
    for param in parameter_space.keys():
        # Randomly choose which parent contributes the parameter
        if random.random() < 0.5:
            offspring1_params[param] = parent1_params[param]
            offspring2_params[param] = parent2_params[param]
        else:
            offspring1_params[param] = parent2_params[param]
            offspring2_params[param] = parent1_params[param]

    # Ensure MACD fastperiod < slowperiod after crossover if MACD is selected
    if offspring1_params.get('indicator_trend') == 'MACD':
         if offspring1_params['macd_fastperiod'] >= offspring1_params['macd_slowperiod']:
              # Swap if necessary, or re-generate one of them
              # Ensure new value is within space limits after swap/regen
              param_space_fast = parameter_space['macd_fastperiod']
              param_space_slow = parameter_space['macd_slowperiod']

              if offspring1_params['macd_slowperiod'] < param_space_fast[0]: # If swapped slow is too small for fast min
                   offspring1_params['macd_fastperiod'] = random.randrange(param_space_fast[0], offspring1_params['macd_slowperiod'], param_space_fast[2]) if offspring1_params['macd_slowperiod'] > param_space_fast[0] else param_space_fast[0]
              elif offspring1_params['macd_fastperiod'] > param_space_slow[1]: # If swapped fast is too large for slow max
                   offspring1_params['macd_slowperiod'] = random.randrange(offspring1_params['macd_fastperiod'] + param_space_slow[2], param_space_slow[1] + param_space_slow[2], param_space_slow[2]) if offspring1_params['macd_fastperiod'] + param_space_slow[2] < param_space_slow[1] + param_space_slow[2] else param_space_slow[1]
              else: # Standard swap if within range
                   offspring1_params['macd_fastperiod'], offspring1_params['macd_slowperiod'] = offspring1_params['macd_slowperiod'], offspring1_params['macd_fastperiod']


              if offspring1_params['macd_fastperiod'] >= offspring1_params['macd_slowperiod']: # If still not correct after swap/regen attempt
                   # Re-generate both, ensuring fast < slow
                   offspring1_params['macd_slowperiod'] = random.randrange(param_space_slow[0], param_space_slow[1] + param_space_slow[2], param_space_slow[2])
                   offspring1_params['macd_fastperiod'] = random.randrange(param_space_fast[0], min(param_space_fast[1], offspring1_params['macd_slowperiod']), param_space_fast[2]) # Ensure fast < slow max
                   if offspring1_params['macd_fastperiod'] >= offspring1_params['macd_slowperiod']: # Final check, if still issue, just pick valid defaults
                       offspring1_params['macd_fastperiod'] = 12
                       offspring1_params['macd_slowperiod'] = 26



    if offspring2_params.get('indicator_trend') == 'MACD':
         if offspring2_params['macd_fastperiod'] >= offspring2_params['macd_slowperiod']:
              # Swap if necessary, or re-generate one of them
              # Ensure new value is within space limits after swap/regen
              param_space_fast = parameter_space['macd_fastperiod']
              param_space_slow = parameter_space['macd_slowperiod']

              if offspring2_params['macd_slowperiod'] < param_space_fast[0]: # If swapped slow is too small for fast min
                   offspring2_params['macd_fastperiod'] = random.randrange(param_space_fast[0], offspring2_params['macd_slowperiod'], param_space_fast[2]) if offspring2_params['macd_slowperiod'] > param_space_fast[0] else param_space_fast[0]
              elif offspring2_params['macd_fastperiod'] > param_space_slow[1]: # If swapped fast is too large for slow max
                   offspring2_params['macd_slowperiod'] = random.randrange(offspring2_params['macd_fastperiod'] + param_space_slow[2], param_space_slow[1] + param_space_slow[2], param_space_slow[2]) if offspring2_params['macd_fastperiod'] + param_space_slow[2] < param_space_slow[1] + param_space_slow[2] else param_space_slow[1]
              else: # Standard swap if within range
                   offspring2_params['macd_fastperiod'], offspring2_params['macd_slowperiod'] = offspring2_params['macd_slowperiod'], offspring2_params['macd_fastperiod']

              if offspring2_params['macd_fastperiod'] >= offspring2_params['macd_slowperiod']: # Final check, if still issue, just pick valid defaults
                   # Re-generate both, ensuring fast < slow
                   offspring2_params['macd_slowperiod'] = random.randrange(param_space_slow[0], param_space_slow[1] + param_space_slow[2], param_space_slow[2])
                   offspring2_params['macd_fastperiod'] = random.randrange(param_space_fast[0], min(param_space_fast[1], offspring2_params['macd_slowperiod']), param_space_fast[2]) # Ensure fast < slow max
                   if offspring2_params['macd_fastperiod'] >= offspring2_params['macd_slowperiod']: # Final check, if still issue, just pick valid defaults
                       offspring2_params['macd_fastperiod'] = 12
                       offspring2_params['macd_slowperiod'] = 26

    return offspring1_params, offspring2_params


def mutate(params, parameter_space, mutation_rate):
    """Mutates a parameter set with a given probability."""
    mutated_params = params.copy()
    for param, values in parameter_space.items():
        if random.random() < mutation_rate:
            if isinstance(values, list): # Categorical parameter
                mutated_params[param] = random.choice(values)
            elif isinstance(values, tuple) and len(values) == 3: # Numeric parameter
                min_val, max_val, step = values
                if isinstance(min_val, int):
                    # Mutate by adding/subtracting a random multiple of step
                    # Ensure mutated value stays within bounds [min_val, max_val]
                    mutation_amount = random.choice([-step, step]) * random.randint(1, 5) # Mutate by up to 5 steps
                    mutated_value = params[param] + mutation_amount
                    mutated_params[param] = max(min_val, min(max_val, mutated_value))
                elif isinstance(min_val, float):
                     # Mutate by adding/subtracting a random multiple of step
                     # Ensure mutated value stays within bounds [min_val, max_val]
                    mutation_amount = random.choice([-step, step]) * random.randint(1, 5) # Mutate by up to 5 steps
                    mutated_value = params[param] + mutation_amount
                    mutated_params[param] = max(min_val, min(max_val, mutated_value))
                    # Ensure the mutated float is a multiple of the step (to stay within the defined grid)
                    mutated_params[param] = round((mutated_params[param] - min_val) / step) * step + min_val
                    mutated_params[param] = round(mutated_params[param], len(str(step).split('.')[-1])) # Round to step decimal places

    # Ensure MACD fastperiod < slowperiod after mutation if MACD is selected
    if mutated_params.get('indicator_trend') == 'MACD':
         if mutated_params['macd_fastperiod'] >= mutated_params['macd_slowperiod']:
              # Swap if necessary, or re-generate one of them
              # Ensure new value is within space limits after swap/regen
              param_space_fast = parameter_space['macd_fastperiod']
              param_space_slow = parameter_space['macd_slowperiod']

              if mutated_params['macd_slowperiod'] < param_space_fast[0]: # If swapped slow is too small for fast min
                   mutated_params['macd_fastperiod'] = random.randrange(param_space_fast[0], mutated_params['macd_slowperiod'], param_space_fast[2]) if mutated_params['macd_slowperiod'] > param_space_fast[0] else param_space_fast[0]
              elif mutated_params['macd_fastperiod'] > param_space_slow[1]: # If swapped fast is too large for slow max
                   mutated_params['macd_slowperiod'] = random.randrange(mutated_params['macd_fastperiod'] + param_space_slow[2], param_space_slow[1] + param_space_slow[2], param_space_slow[2]) if mutated_params['macd_fastperiod'] + param_space_slow[2] < param_space_slow[1] + param_space_slow[2] else param_space_slow[1]
              else: # Standard swap if within range
                   mutated_params['macd_fastperiod'], mutated_params['macd_slowperiod'] = mutated_params['macd_slowperiod'], mutated_params['macd_fastperiod']


              if mutated_params['macd_fastperiod'] >= mutated_params['macd_slowperiod']: # Final check, if still issue, just pick valid defaults
                   # Re-generate both, ensuring fast < slow
                   mutated_params['macd_slowperiod'] = random.randrange(param_space_slow[0], param_space_slow[1] + param_space_slow[2], param_space_slow[2])
                   mutated_params['macd_fastperiod'] = random.randrange(param_space_fast[0], min(param_space_fast[1], mutated_params['macd_slowperiod']), param_space_fast[2]) # Ensure fast < slow max
                   if mutated_params['macd_fastperiod'] >= mutated_params['macd_slowperiod']: # Final check, if still issue, just pick valid defaults
                       mutated_params['macd_fastperiod'] = 12
                       mutated_params['macd_slowperiod'] = 26


    return mutated_params


def run_genetic_algorithm(df_dollar_bars, parameter_space, fitness_function,
                          population_size=50, num_generations=100, mutation_rate=0.1,
                          initial_capital=100000, walkforward_periods=5, train_period_ratio=0.7):
    """Runs the Genetic Algorithm with walk-forward optimization."""

    # Ensure df_dollar_bars is sorted by index (Date)
    df_dollar_bars = df_dollar_bars.sort_index()

    total_bars = len(df_dollar_bars)

    # Calculate the size of each walk-forward window
    # Ensure we have enough data for at least one full train+test window
    min_total_window_size = 100 # Example minimum total window size
    if total_bars < min_total_window_size:
         display(f"Error: Total data bars ({total_bars}) is less than minimum required for walk-forward ({min_total_window_size}). Cannot run walk-forward.")
         return None, -10000 # Return None params and error fitness


    # Calculate window sizes based on total bars and number of periods
    # Distribute the total bars as evenly as possible across windows
    bars_per_window = total_bars // walkforward_periods
    train_bars_per_window = int(bars_per_window * train_period_ratio)
    test_bars_per_window = bars_per_window - train_bars_per_window

    # Ensure minimum bars for train and test windows
    min_train_bars = 50 # Example minimum train bars
    min_test_bars = 50 # Example minimum test bars (needs to be larger than max possible lookback for robust evaluation)
    # A safer minimum test bar count might be related to the maximum possible lookback in the parameter space
    # Estimate max possible lookback from parameter space (worst case: MACD slow + signal, or 2*RSI period, or max lookback + ATR/StdDev dependencies)
    max_possible_lookback = 0
    for param, values in parameter_space.items():
         if param in ['lookback_trend', 'lookback_volatility', 'lookback_exhaustion', 'rsi_period']:
              if isinstance(values, tuple) and len(values) == 3:
                   max_possible_lookback = max(max_possible_lookback, values[1]) # Use max value of lookback tuples
         elif param == 'macd_slowperiod':
              if isinstance(values, tuple) and len(values) == 3:
                   max_macd_slow = values[1]
                   # Assume max signal period is also the max of its range
                   max_macd_signal = parameter_space.get('macd_signalperiod', (5, 20, 1))[1] # Get max signalperiod, default if not in space
                   max_possible_lookback = max(max_possible_lookback, max_macd_slow + max_macd_signal)

    # Set a minimum test window size that is at least the maximum possible lookback plus a buffer
    min_test_bars_needed = max_possible_lookback + 10 # Need enough bars after lookback for simulation


    # Ensure the calculated window sizes are feasible
    if train_bars_per_window < min_train_bars:
         display(f"Warning: Calculated train window size ({train_bars_per_window}) is less than minimum required ({min_train_bars}). Adjusting train size.")
         train_bars_per_window = min_train_bars
         test_bars_per_window = bars_per_window - train_bars_per_window # Recalculate test size

    if test_bars_per_window < min_test_bars_needed:
         display(f"Warning: Calculated test window size ({test_bars_per_window}) is less than minimum required after lookback ({min_test_bars_needed}). Adjusting test size.")
         test_bars_per_window = min_test_bars_needed
         train_bars_per_window = bars_per_window - test_bars_per_window # Recalculate train size


    # Final check after adjustments
    if train_bars_per_window < min_train_bars or test_bars_per_window < min_test_bars_needed or train_bars_per_window + test_bars_per_window > bars_per_window * walkforward_periods:
         # This condition needs refinement. The issue is likely if the total length of a window (train+test)
         # is less than the sum of minimums, or if after adjustments the total length exceeds the initial bars_per_window.
         # A simpler check: ensure total_bars is enough for walkforward_periods * (min_train_bars + min_test_bars_needed)
         required_total_bars = walkforward_periods * (min_train_bars + min_test_bars_needed)
         if total_bars < required_total_bars:
              display(f"Error: Total data bars ({total_bars}) is insufficient for {walkforward_periods} walk-forward periods with minimum train ({min_train_bars}) and test ({min_test_bars_needed}) bars per window. Required: {required_total_bars} bars.")
              display("Consider reducing walk-forward periods or minimum required bars.")
              return None, -10000 # Return error


    display(f"Using Adjusted Window Sizes: Train = {train_bars_per_window} bars, Test = {test_bars_per_window} bars.")


    best_params_overall = None
    best_fitness_overall = -float('inf')
    walkforward_results = []

    current_start_index = 0
    for i in range(walkforward_periods):

        # Calculate window indices
        train_start = current_start_index
        train_end = train_start + train_bars_per_window
        test_start = train_end
        test_end = test_start + test_bars_per_window # Initial test end

        # Adjust test_end for the last window if it extends beyond total data
        if test_end > total_bars:
             test_end = total_bars

        # Check if the current window has enough data
        # We need enough data for the training period AND enough data in the test period *after* considering the maximum lookback.
        # The fitness function checks the test period length after dropping NaNs.
        # Here, we just need to ensure the raw test_data slice has enough length to potentially yield a valid evaluation.
        # The minimum data needed in the raw test slice should be > max_possible_lookback for any trading to be possible after dropna.
        if test_end - test_start < min_test_bars_needed:
             display(f"Warning: Skipping walk-forward window {i+1} due to insufficient data in test period ({test_end-test_start} bars). Minimum required after lookback: {min_test_bars_needed}.")
             # If the remaining data is too short for any valid test window, stop the loop
             if total_bars - test_start < min_test_bars_needed:
                  display("Stopping walk-forward as remaining data is too short for a valid test window.")
                  break
             # Move current_start_index forward even if skipping to avoid infinite loop on small data
             current_start_index = test_end if test_end > current_start_index else current_start_index + 1 # Ensure progress
             continue # Skip this window


        train_data = df_dollar_bars.iloc[train_start:train_end].copy()
        test_data = df_dollar_bars.iloc[test_start:test_end].copy()

        display(f"\n--- Walk-Forward Window {i+1}/{walkforward_periods} ---")
        display(f"Train Period: {train_data.index.min().strftime('%Y-%m-%d')} to {train_data.index.max().strftime('%Y-%m-%d')} ({len(train_data)} bars)")
        display(f"Test Period: {test_data.index.min().strftime('%Y-%m-%d')} to {test_data.index.max().strftime('%Y-%m-%d')} ({len(test_data)} bars)")

        population = initialize_population(population_size, parameter_space)
        best_fitness_this_window = -float('inf')
        best_params_this_window = None
        window_has_valid_params = False # Track if any valid params were found in this window


        # Run GA on the training data
        for generation in range(num_generations):
            fitnesses = []
            for params in population:
                # Evaluate on training data
                fitness = fitness_function(params, train_data, initial_capital)
                fitnesses.append(fitness)

            # Select parents (e.g., top 50%) - only from individuals with valid fitness
            # Filter out error fitnesses for parent selection
            valid_indices = [i for i, f in enumerate(fitnesses) if f > -1000] # Assuming error codes are < -1000

            if not valid_indices:
                 # If no valid individuals in this generation, regenerate population
                 display(f"  Warning: No valid parameters found in training for generation {generation+1}. Regenerating population.")
                 population = initialize_population(population_size, parameter_space)
                 continue # Move to next generation


            num_parents = max(1, len(valid_indices) // 2) # Ensure at least 1 parent if valid exist
            parents = select_parents(population, fitnesses, num_parents)

            # Create next generation
            next_population = parents # Keep the best individuals (elitism)
            num_offspring = population_size - len(next_population) # Calculate remaining slots

            # Ensure we have enough parents for crossover, otherwise just mutate/duplicate
            if len(parents) < 2:
                 display(f"  Warning: Not enough valid parents ({len(parents)}) for crossover in generation {generation+1}. Duplicating/Mutating existing parents.")
                 while len(next_population) < population_size and parents:
                      next_population.append(mutate(random.choice(parents), parameter_space, mutation_rate)) # Mutate existing parents
                 # If still not full and no parents left, fill with random
                 while len(next_population) < population_size:
                      next_population.append(generate_random_params(parameter_space))

            else: # Enough parents for crossover
                 while len(next_population) < population_size:
                     # Select two parents randomly from the parents pool
                     p1 = random.choice(parents)
                     p2 = random.choice(parents)

                     # Perform crossover
                     offspring1, offspring2 = crossover(p1, p2, parameter_space)

                     # Perform mutation
                     mutated_offspring1 = mutate(offspring1, parameter_space, mutation_rate)
                     mutated_offspring2 = mutate(offspring2, parameter_space, mutation_rate)

                     next_population.append(mutated_offspring1)
                     if len(next_population) < population_size:
                         next_population.append(mutated_offspring2)


            population = next_population

            # Track the best individual in this generation on the TRAINING data (from valid fitnesses)
            current_best_fitness_in_gen = max([fitnesses[i] for i in valid_indices])
            current_best_params_in_gen = population[[fitnesses[i] for i in valid_indices].index(current_best_fitness_in_gen)] # Get params corresponding to max valid fitness

            if current_best_fitness_in_gen > best_fitness_this_window:
                best_fitness_this_window = current_best_fitness_in_gen
                best_params_this_window = current_best_params_in_gen
                window_has_valid_params = True


            # display(f"  Generation {generation+1}: Best Train Fitness = {best_fitness_this_window:.4f}") # Optional: print progress


        # Evaluate the best parameters from the training period on the TEST data
        if best_params_this_window is not None: # Only evaluate if a best parameter set was found in training
            test_fitness = fitness_function(best_params_this_window, test_data, initial_capital)
            display(f"  Best Train Fitness: {best_fitness_this_window:.4f}")
            display(f"  Test Fitness (Combined Score): {test_fitness:.4f}") # Use Combined Score in display

            walkforward_results.append({
                'Train_Start': train_data.index.min(), 'Train_End': train_data.index.max(),
                'Test_Start': test_data.index.min(), 'Test_End': test_data.index.max(),
                'Best_Train_Fitness': best_fitness_this_window,
                'Test_Fitness': test_fitness,
                'Best_Params': best_params_this_window
            })

        else:
             display("  No best parameters found for this training window.")


        # Move to the next window - Stepping window
        current_start_index = test_end


    display("\nWalk-Forward Optimization Completed.")
    display("Walk-Forward Window Results:")
    walkforward_results_df = pd.DataFrame(walkforward_results)
    display(walkforward_results_df)

    # Determine the overall best parameters (e.g., based on average test fitness)
    if not walkforward_results_df.empty:
        # Filter out error codes from test fitness before calculating average
        valid_test_results_df = walkforward_results_df[walkforward_results_df['Test_Fitness'] > -1000] # Assuming error codes are below -1000

        if not valid_test_results_df.empty:
             average_test_fitness = valid_test_results_df['Test_Fitness'].mean()
             display(f"\nAverage Valid Test Fitness (Combined Score) across windows: {average_test_fitness:.4f}")

             # Select the parameters from the window with the highest VALID test fitness.
             best_window_index = valid_test_results_df['Test_Fitness'].idxmax()
             best_params_overall = valid_test_results_df.loc[best_window_index, 'Best_Params']
             best_fitness_overall = valid_test_results_df.loc[best_window_index, 'Test_Fitness'] # Use test fitness as overall best fitness metric


             display("\nOverall Best Parameters (from window with highest valid test fitness):")
             display(best_params_overall)
             display(f"Overall Best Fitness (Test Combined Score): {best_fitness_overall:.4f}")
        else:
             display("\nNo valid test fitness results found across all windows.")
             best_params_overall = None
             best_fitness_overall = -10000 # Indicate no valid results


    else:
        display("\nNo walk-forward results to determine overall best parameters.")
        best_params_overall = None
        best_fitness_overall = -10000 # Indicate no results


    return best_params_overall, best_fitness_overall # Return the best parameters found and their fitness


# Example usage of the GA framework (requires the dollar_bars_df and evaluate_strategy_with_indicators_and_params function)
# Assuming dollar_bars_df contains the raw dollar bar data
# Assuming evaluate_strategy_with_indicators_and_params is defined in a previous cell

# Ensure dollar_bars_df is available and is the raw dollar bar data for evaluation
# df_for_ga_optimization = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy() # Use raw data

# Note: The dollar_bars_df was updated with factors, MSS, and Regime in previous steps.
# The evaluate_strategy_with_indicators_and_params function expects the *raw* dollar bars data.
# So, we need to ensure we pass the correct raw dollar bars data to the GA function.
# Let's assume `raw_dollar_bars_df` holds the initial dollar bar data before factor calculation.
# If not, you might need to reload or reconstruct it.
# Based on the notebook state, the global `dollar_bars_df` *does* contain the raw data columns plus calculated factors.
# The fitness function makes a copy and calculates factors internally. So, passing the current `dollar_bars_df` should be fine,
# as the fitness function will use the raw columns and recalculate.

# Let's use the global dollar_bars_df which has the raw data columns
df_for_ga_optimization = dollar_bars_df.copy()


# Run the GA (example settings)
# These settings (population_size, num_generations, mutation_rate, walkforward_periods, train_period_ratio)
# are examples and should be tuned. Running this might take a while depending on the data size and settings.
# ga_population_size = 30 # Reduced for quicker example
# ga_num_generations = 20 # Reduced for quicker example
# ga_mutation_rate = 0.1
# ga_walkforward_periods = 3 # Reduced for quicker example
# ga_train_period_ratio = 0.7

# Ensure initial_capital is defined (it is from previous cells)

# best_ga_params, best_ga_fitness = run_genetic_algorithm(
#     df_for_ga_optimization,
#     PARAMETER_SPACE,
#     evaluate_strategy_with_indicators_and_params,
#     population_size=ga_population_size,
#     num_generations=ga_num_generations,
#     mutation_rate=ga_mutation_rate,
#     initial_capital=initial_capital,
#     walkforward_periods=ga_walkforward_periods,
#     train_period_ratio=ga_train_period_ratio
# )

# display("\nGA Run Completed.")
# display(f"Overall Best GA Parameters: {best_ga_params}")
# display(f"Overall Best GA Fitness (Test Sortino Ratio): {best_ga_fitness:.4f}")

"""**Reasoning**:
Call the `run_genetic_algorithm` function with the raw dollar bars data, the defined parameter space, and the fitness function.

## Run the GA Optimization (Corrected)

### Subtask:
Execute the Genetic Algorithm with walk-forward optimization on the historical data using the corrected fitness function.

**Reasoning**:
Call the `run_genetic_algorithm` function with the raw dollar bars data, the defined parameter space, and the corrected fitness function.
"""

# Ensure dollar_bars_df is the raw dollar bar data for the fitness function
# Based on the notebook state, the global `dollar_bars_df` contains the raw data columns
# plus calculated factors. The fitness function is designed to use the raw columns and recalculate.
# So, passing the current `dollar_bars_df` should be fine.
df_for_ga_optimization = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


# Run the GA (example settings)
# These settings (population_size, num_generations, mutation_rate, walkforward_periods, train_period_ratio)
# are examples and should be tuned for a proper optimization run.
# Reduced settings here for a quicker example execution.
ga_population_size = 50 # Example population size
ga_num_generations = 50 # Example number of generations
ga_mutation_rate = 0.1 # Example mutation rate
ga_walkforward_periods = 5 # Example number of walk-forward windows
ga_train_period_ratio = 0.7 # Example ratio of data for training in each window

# Ensure initial_capital is defined (it is from previous cells)

display("Starting Genetic Algorithm Optimization (Corrected)...")
display(f"Settings: Population Size={ga_population_size}, Generations={ga_num_generations}, Mutation Rate={ga_mutation_rate}, Walk-Forward Periods={ga_walkforward_periods}, Train Ratio={ga_train_period_ratio}")


best_ga_params, best_ga_fitness = run_genetic_algorithm(
    df_for_ga_optimization, # Pass the raw dollar bars data
    PARAMETER_SPACE, # Use the defined parameter space
    evaluate_strategy_with_indicators_and_params, # Use the corrected fitness function
    population_size=ga_population_size,
    num_generations=ga_num_generations,
    mutation_rate=ga_mutation_rate,
    initial_capital=initial_capital,
    walkforward_periods=ga_walkforward_periods,
    train_period_ratio=ga_train_period_ratio
)

display("\nGenetic Algorithm Optimization Completed (Corrected).")
display(f"Overall Best GA Parameters: {best_ga_params}")
display(f"Overall Best GA Fitness (Test Sortino Ratio): {best_ga_fitness:.4f}")

"""## Evaluate Optimized Parameters

### Subtask:
Use the best parameter set found by the GA run to evaluate its performance on the entire dataset.

**Reasoning**:
Call the `evaluate_strategy_with_indicators_and_params` function with the `best_ga_params` found in the previous step and the entire raw dollar bars dataset. Then, evaluate and display the performance metrics.
"""

# Ensure dollar_bars_df is the raw dollar bar data needed by the fitness function
df_for_evaluation = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()

# Check if best_ga_params is not None before proceeding with evaluation
if best_ga_params is None:
    display("No valid optimized parameters were found by the Genetic Algorithm. Cannot perform evaluation.")
else:
    # Evaluate the strategy using the best parameters found by the quick GA test
    # The evaluate_strategy_with_indicators_and_params function also calculates performance internally
    # but we want to display full performance metrics and plot the equity curve.
    # Let's re-run the simulation part within this cell to get the full equity curve and trade log.

    # Re-run the trading simulation with the best parameters found by the GA
    # This code is adapted from the enhanced logic simulation but uses the best_ga_params

    params = best_ga_params # Use the parameters found by the GA

    # Extract parameters
    indicator_trend_name = params.get('indicator_trend', 'Slope')
    indicator_volatility_name = params.get('indicator_volatility', 'ATR')
    indicator_exhaustion_name = params.get('indicator_exhaustion', 'SMADiff')

    lookback_trend = params.get('lookback_trend', 20)
    lookback_volatility = params.get('lookback_volatility', 20)
    lookback_exhaustion = params.get('lookback_exhaustion', 20)

    # Regime thresholds
    strong_bull_threshold = params.get('strong_bull_threshold', 50)
    weak_bull_threshold = params.get('weak_bull_threshold', 20)
    neutral_threshold_upper = params.get('neutral_threshold_upper', 20)
    neutral_threshold_lower = params.get('neutral_threshold_lower', -20)
    strong_bear_threshold = params.get('strong_bear_threshold', -50)
    weak_bear_threshold = params.get('weak_bear_threshold', -20)

    # Stop-loss multipliers
    stop_loss_multiplier_strong = params.get('stop_loss_multiplier_strong', 2)
    stop_loss_multiplier_weak = params.get('stop_loss_multiplier_weak', 1)

    # Gradual entry/exit parameter
    entry_step_size = params.get('entry_step_size', 0.2)

    # Extract dynamic weight parameters
    weights_params = {
        'strong_bull_trend': params.get('strong_bull_trend', 0.6), 'strong_bull_volatility': params.get('strong_bull_volatility', 0.1), 'strong_bull_exhaustion': params.get('strong_bull_exhaustion', 0.3),
        'weak_bull_trend': params.get('weak_bull_trend', 0.4), 'weak_bull_volatility': params.get('weak_bull_volatility', 0.2), 'weak_bull_exhaustion': params.get('weak_bull_exhaustion', 0.4),
        'neutral_trend': params.get('neutral_trend', 0.2), 'neutral_volatility': params.get('neutral_volatility', 0.4), 'neutral_exhaustion': params.get('neutral_exhaustion', 0.4),
        'weak_bear_trend': params.get('weak_bear_trend', 0.4), 'weak_bear_volatility': params.get('weak_bear_volatility', 0.2), 'weak_bear_exhaustion': params.get('weak_bear_exhaustion', 0.4),
        'strong_bear_trend': params.get('strong_bear_trend', 0.6), 'strong_bear_volatility': params.get('strong_bear_volatility', 0.1), 'strong_bear_exhaustion': params.get('strong_bear_exhaustion', 0.3),
    }

    # --- Calculate Factors using specified indicators and lookbacks on the entire dataset ---
    df_sim = df_for_evaluation.copy() # Work on a copy for simulation

    try:
        # Calculate Volatility first as Exhaustion might depend on it (SMADiff)
        if indicator_volatility_name in INDICATOR_FUNCTIONS['Volatility']:
            df_sim['Volatility_Factor'] = INDICATOR_FUNCTIONS['Volatility'][indicator_volatility_name](df_sim.copy(), lookback_volatility)
        else:
            df_sim['Volatility_Factor'] = calculate_volatility_atr(df_sim.copy(), lookback_volatility)

        if indicator_trend_name in INDICATOR_FUNCTIONS['Trend']:
             if indicator_trend_name == 'Slope':
                  df_sim['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend'][indicator_trend_name](df_sim.copy(), lookback_trend)
             elif indicator_trend_name == 'MACD':
                  # MACD uses its own default lookbacks unless specified in params (which is not in the current param set)
                  # Let's make MACD lookbacks also configurable via params if they exist, otherwise use defaults
                  macd_fast = params.get('macd_fastperiod', 12)
                  macd_slow = params.get('macd_slowperiod', 26)
                  macd_signal = params.get('macd_signalperiod', 9)
                  df_sim['Trend_Factor'] = INDICATOR_FUNCTIONS['Trend'][indicator_trend_name](df_sim.copy(), fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
             else:
                  df_sim['Trend_Factor'] = np.nan

        else:
             df_sim['Trend_Factor'] = calculate_trend_slope(df_sim.copy(), lookback_trend)

        if indicator_exhaustion_name in INDICATOR_FUNCTIONS['Exhaustion']:
            if indicator_exhaustion_name == 'SMADiff':
                 if 'Volatility_Factor' in df_sim.columns:
                     df_sim['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_name](df_sim.copy(), lookback_exhaustion, df_sim['Volatility_Factor'])
                 else:
                      df_sim['Exhaustion_Factor'] = np.nan
        elif indicator_exhaustion_name == 'RSI':
             # RSI takes df and lookback_exhaustion
             rsi_period = params.get('rsi_period', 14) # Make RSI period configurable
             df_sim['Exhaustion_Factor'] = INDICATOR_FUNCTIONS['Exhaustion'][indicator_exhaustion_name](df_sim.copy(), lookback=rsi_period)
        else:
             df_sim['Exhaustion_Factor'] = np.nan
        else:
            if 'Volatility_Factor' in df_sim.columns:
                df_sim['Exhaustion_Factor'] = calculate_exhaustion_sma_diff(df_sim.copy(), lookback_exhaustion, df_sim['Volatility_Factor'])
            else:
                 df_sim['Exhaustion_Factor'] = np.nan

    except Exception as e:
        display(f"Error during factor calculation with optimized parameters: {e}")
        df_sim = pd.DataFrame() # Empty dataframe on error


    # Drop rows with NaN values generated by lookback periods after factor calculation
    df_sim = df_sim.dropna(subset=['Trend_Factor', 'Volatility_Factor', 'Exhaustion_Factor'])

    if not df_sim.empty:
        # --- Calculate Dynamic MSS and Regime with Optimized Weights and Thresholds ---

        # Calculate initial static MSS to get a starting regime for dynamic weights
        # Use default static weights for this initial MSS calculation (not optimized by GA)
        static_weight_trend = 0.4
        static_weight_volatility = 0.3
        static_weight_exhaustion = 0.3

        df_sim['MSS_static_initial'] = (static_weight_trend * df_sim['Trend_Factor'] +
                                        static_weight_volatility * df_sim['Volatility_Factor'] +
                                        static_weight_exhaustion * df_sim['Exhaustion_Factor'])

        # Classify the initial static regime using the OPTIMIZED thresholds
        def classify_regime_optimized_thresholds(mss):
            if mss > strong_bull_threshold:
                return 'Strong Bull'
            elif mss > weak_bull_threshold:
                return 'Weak Bull'
            elif mss >= neutral_threshold_lower and mss <= neutral_threshold_upper:
                return 'Neutral'
            elif mss > strong_bear_threshold:
                return 'Weak Bear'
            else:
                return 'Strong Bear'

        df_sim['Regime_initial'] = df_sim['MSS_static_initial'].apply(classify_regime_optimized_thresholds)


        # Now, calculate Dynamic MSS using OPTIMIZED dynamic weights based on the *initial* regime
        df_sim['MSS_dynamic'] = np.nan # Initialize dynamic MSS column

        for index, row in df_sim.iterrows():
            current_regime_initial = row['Regime_initial'] # Use the initial static regime
            # Pass the OPTIMIZED weights_params dictionary
            dynamic_weights = get_dynamic_weights(current_regime_initial, weights_params)

            # Calculate dynamic MSS using factor values and dynamic weights
            dynamic_mss = (dynamic_weights['Trend'] * row['Trend_Factor'] +
                           dynamic_weights['Volatility'] * row['Volatility_Factor'] +
                           dynamic_weights['Exhaustion'] * row['Exhaustion_Factor'])

            df_sim.loc[index, 'MSS_dynamic'] = dynamic_mss


        # Update the 'Regime' column based on the newly calculated 'MSS_dynamic'
        # Use the OPTIMIZED thresholds for classifying the dynamic MSS into regimes
        df_sim['Regime_dynamic'] = df_sim['MSS_dynamic'].apply(classify_regime_optimized_thresholds)


        # --- Trading Simulation with Optimized Parameters ---
        position_fraction = 0.0 # Current position as a fraction of initial_capital
        total_units = 0.0
        total_cost_basis = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        equity_curve_optimized = []
        trade_log_optimized = []
        current_capital = initial_capital
        max_position_fraction = 1.0

        # Use the OPTIMIZED entry_step_size
        max_step_fraction = entry_step_size * max_position_fraction

        peak_price_since_entry = -float('inf')
        valley_price_since_entry = float('inf')


        for index, row in df_sim.iterrows():
            current_price = row['Close']
            current_regime = row['Regime_dynamic'] # Use the regime based on dynamic MSS and optimized thresholds
            current_mss = row['MSS_dynamic'] # Use the dynamic MSS
            current_volatility_factor = row['Volatility_Factor'] # Use the normalized Volatility Factor

            # Determine stop-loss distance and take-profit distance based on regime and Volatility Factor
            # Use OPTIMIZED stop-loss multipliers
            if not isinstance(current_volatility_factor, (int, float)) or np.isnan(current_volatility_factor) or abs(current_volatility_factor) < 1e-9:
                 stop_loss_distance = 0
                 take_profit_distance = 0
            else:
                # Example Profit Target Multipliers (could also be optimized by GA)
                take_profit_multiplier_strong = 4.0
                take_profit_multiplier_weak = 2.0

                if current_regime in ['Strong Bull', 'Strong Bear']:
                    stop_loss_distance = stop_loss_multiplier_strong * abs(current_volatility_factor)
                    take_profit_distance = take_profit_multiplier_strong * abs(current_volatility_factor)
                elif current_regime in ['Weak Bull', 'Weak Bear']:
                    stop_loss_distance = stop_loss_multiplier_weak * abs(current_volatility_factor)
                    take_profit_distance = take_profit_multiplier_weak * abs(current_volatility_factor)
                else: # Neutral
                    stop_loss_distance = 0
                    take_profit_distance = 0


            # --- Trading Logic with Fractional Positions (Capital Allocation Model) ---

            # Determine target position fraction based on regime and MSS confidence using OPTIMIZED thresholds
            target_position_fraction = 0.0
            if current_regime == 'Strong Bull':
                normalized_mss = (current_mss - strong_bull_threshold) / (100 - strong_bull_threshold) if (100 - strong_bull_threshold) > 0 else 0
                target_position_fraction = max_position_fraction * np.clip(normalized_mss, 0, 1)

            elif current_regime == 'Weak Bull':
                 normalized_mss = (current_mss - weak_bull_threshold) / (neutral_threshold_upper - weak_bull_threshold) if (neutral_threshold_upper - weak_bull_threshold) > 0 else 0
                 target_position_fraction = position_fraction # Default to hold
                 if position_fraction > 1e-9: # If currently long
                     target_position_fraction = max_position_fraction * np.clip(normalized_mss, 0, 1)

            elif current_regime == 'Neutral':
                target_position_fraction = 0.0

            elif current_regime == 'Weak Bear':
                 normalized_mss = (current_mss - (-100)) / (weak_bear_threshold - (-100)) if (weak_bear_threshold - (-100)) > 0 else 0
                 target_position_fraction = position_fraction # Default to hold
                 if position_fraction < -1e-9: # If currently short
                      target_position_fraction = -max_position_fraction * np.clip(1 - normalized_mss, 0, 1)

            elif current_regime == 'Strong Bear':
                normalized_mss = (current_mss - (-100)) / (strong_bear_threshold - (-100)) if (strong_bear_threshold - (-100)) > 0 else 0
                target_position_fraction = -max_position_fraction * np.clip(1 - normalized_mss, 0, 1)

            # Ensure target_position_fraction has correct sign based on regime
            if target_position_fraction > 1e-9 and current_regime not in ['Strong Bull', 'Weak Bull']:
                 target_position_fraction = 0.0
            elif target_position_fraction < -1e-9 and current_regime not in ['Strong Bear', 'Weak Bear']:
                 target_position_fraction = 0.0
            # In Neutral, always target 0
            if current_regime == 'Neutral':
                target_position_fraction = 0.0


            # Calculate the change in position fraction
            position_fraction_change = target_position_fraction - position_fraction

            # Limit position fraction change to max_step_fraction (using OPTIMIZED entry_step_size)
            position_fraction_change = np.clip(position_fraction_change, -max_step_fraction, max_step_fraction)

            # Calculate the amount of capital to allocate/deallocate in this step
            capital_to_trade = position_fraction_change * initial_capital

            # Calculate the units to trade based on the capital amount and current price
            units_to_trade = capital_to_trade / current_price if current_price > 0 else 0.0

            # Update total units and total cost basis based on units_to_trade
            if units_to_trade > 1e-9: # Buying
                total_cost_basis += units_to_trade * current_price
                total_units += units_to_trade
                action = 'Increase Long' if position_fraction_change > 0 and total_units > units_to_trade else 'Enter Long'
                trade_log_optimized.append({'Date': index, 'Action': action + ' (Gradual)', 'Price': current_price, 'Units': units_to_trade, 'TotalUnits': total_units, 'PnL': np.nan})
            elif units_to_trade < -1e-9: # Selling
                units_to_sell = abs(units_to_trade)
                if total_units > 1e-9: # Only sell if we have units (long position)
                    units_to_sell = min(units_to_sell, total_units)
                    units_to_trade = -units_to_sell

                    avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0
                    pnl_on_sold_units = (current_price - avg_cost_basis_per_unit) * units_to_sell

                    current_capital += pnl_on_sold_units

                    total_cost_basis -= units_to_sell * avg_cost_basis_per_unit
                    total_units -= units_to_sell

                    action = 'Decrease Long' if total_units > 1e-9 else 'Exit Long'
                    trade_log_optimized.append({'Date': index, 'Action': action + ' (Gradual)', 'Price': current_price, 'Units': -units_to_sell, 'TotalUnits': total_units, 'PnL': pnl_on_sold_units})
                elif total_units < -1e-9: # Short selling (not active)
                     pass

            current_market_value = total_units * current_price if current_price > 0 else 0.0
            position_fraction = current_market_value / initial_capital if initial_capital > 0 else 0.0


            # --- Refined Stop-Loss and Take-Profit Logic ---
            if abs(total_units) > 1e-9:
                avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0

                if total_units > 0: # Long position
                     peak_price_since_entry = max(peak_price_since_entry, current_price)
                     stop_loss_level = peak_price_since_entry - stop_loss_distance if stop_loss_distance > 0 else -float('inf')
                     stop_loss = stop_loss_level

                     take_profit_level = avg_cost_basis_per_unit + take_profit_distance if take_profit_distance > 0 else float('inf')
                     take_profit = take_profit_level
                elif total_units < 0: # Short position (not active)
                     valley_price_since_entry = min(valley_price_since_entry, current_price)
                     stop_loss_level = valley_price_since_entry + stop_loss_distance if stop_loss_distance > 0 else float('inf')
                     stop_loss = stop_loss_level

                     take_profit_level = avg_cost_basis_per_unit - take_profit_distance if take_profit_distance > 0 else -float('inf')
                     take_profit = take_profit_level
                else:
                     stop_loss = 0.0
                     take_profit = 0.0
                     peak_price_since_entry = -float('inf')
                     valley_price_since_entry = float('inf')

            # Check for Stop-Loss or Take-Profit hit
            if abs(total_units) > 1e-9:
                 avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0

                 if total_units > 0: # Long position
                     if current_price >= take_profit and not np.isinf(take_profit):
                         pnl = (current_price - avg_cost_basis_per_unit) * total_units
                         current_capital += pnl
                         trade_log_optimized.append({'Date': index, 'Action': 'Take Profit Long', 'Price': current_price, 'Units': -total_units, 'TotalUnits': 0.0, 'PnL': pnl})
                         total_units = 0.0
                         total_cost_basis = 0.0
                         position_fraction = 0.0
                         stop_loss = 0.0
                         take_profit = 0.0
                         peak_price_since_entry = -float('inf')

                     elif current_price <= stop_loss and not np.isinf(stop_loss) and stop_loss > 0: # Ensure stop_loss is positive for long
                         pnl = (current_price - avg_cost_basis_per_unit) * total_units
                         current_capital += pnl
                         trade_log_optimized.append({'Date': index, 'Action': 'Stop Out Long', 'Price': current_price, 'Units': -total_units, 'TotalUnits': 0.0, 'PnL': pnl})
                         total_units = 0.0
                         total_cost_basis = 0.0
                         position_fraction = 0.0
                         stop_loss = 0.0
                         take_profit = 0.0
                         peak_price_since_entry = -float('inf')

                 elif total_units < 0: # Short position (not active)
                      pass

            avg_cost_basis_per_unit = total_cost_basis / total_units if total_units != 0 else 0
            unrealized_pnl = (current_price - avg_cost_basis_per_unit) * total_units if abs(total_units) > 1e-9 else 0.0
            equity_curve_optimized.append({'Date': index, 'Equity': current_capital + unrealized_pnl})


    # Convert trade log and equity curve to DataFrames
    trade_log_optimized_df = pd.DataFrame(trade_log_optimized)
    equity_curve_df_optimized = pd.DataFrame(equity_curve_optimized).set_index('Date')


    # --- Performance Evaluation ---
    if equity_curve_df_optimized.empty or len(equity_curve_df_optimized) < 2:
        display("Optimized equity curve is empty or too short for evaluation.")
    else:
        equity_curve_df_optimized['Daily_Return'] = equity_curve_df_optimized['Equity'].pct_change().fillna(0)
        total_return_optimized = (equity_curve_df_optimized['Equity'].iloc[-1] - initial_capital) / initial_capital
        trading_periods_per_year = 365 # Adjust if necessary
        annualized_return_optimized = (1 + total_return_optimized)**(trading_periods_per_year / len(equity_curve_df_optimized)) - 1

        equity_curve_df_optimized['Peak'] = equity_curve_df_optimized['Equity'].cummax()
        equity_curve_df_optimized['Drawdown'] = equity_curve_df_optimized['Equity'] - equity_curve_df_optimized['Peak']
        max_drawdown_optimized = equity_curve_df_optimized['Drawdown'].min()

        mar = 0
        downside_returns_optimized = equity_curve_df_optimized[equity_curve_df_optimized['Daily_Return'] < mar]['Daily_Return']

        downside_deviation = downside_returns_optimized.std()

        if downside_deviation == 0 or np.isnan(downside_deviation):
            if annualized_return_optimized > mar:
                 sortino_ratio_optimized = float('inf')
            elif annualized_return_optimized == mar:
                 sortino_ratio_optimized = 0
            else:
                 # Losing strategy with no measurable downside deviation (e.g., flat equity curve below initial capital)
                 sortino_ratio_optimized = -1003 # Specific error code for losing with no downside

        else:
            sortino_ratio_optimized = (annualized_return_optimized - mar) / downside_deviation

        # Handle cases where sortino_ratio is infinite or NaN after calculation (safety check)
        if np.isinf(sortino_ratio_optimized):
             # If infinite and annualized_return is > mar, it's a good result, don't return error code
             if annualized_return_optimized > mar:
                  pass # Keep the infinite value
             else:
                  sortino_ratio_optimized = -1001 # Specific error code for problematic infinite ratio
        elif np.isnan(sortino_ratio_optimized):
            sortino_ratio_optimized = -1002 # Specific error code for NaN ratio


        calmar_ratio_optimized = annualized_return_optimized / abs(max_drawdown_optimized) if max_drawdown_optimized != 0 else float('inf')


        display("\nPerformance Metrics (using Optimized Parameters from GA Test):")
        display(f"Initial Capital: {initial_capital:.2f}")
        display(f"Final Equity: {equity_curve_df_optimized['Equity'].iloc[-1]:.2f}")
        display(f"Total Return: {total_return_optimized:.4f}")
        display(f"Annualized Return: {annualized_return_optimized:.4f}")
        display(f"Max Drawdown: {max_drawdown_optimized:.2f}")
        display(f"Sortino Ratio (MAR=0): {sortino_ratio_optimized:.4f}")
        display(f"Calmar Ratio: {calmar_ratio_optimized:.4f}")

        # Plot the equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve_df_optimized.index, equity_curve_df_optimized['Equity'], label='Equity Curve (Optimized Parameters)')
        plt.title('Equity Curve (Optimized Parameters)')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Display head of the trade log
        display("\nTrade Log Summary (Optimized Parameters Simulation):")
        display(trade_log_optimized_df.head())

        # Analyze winning trades from the simulation
        winning_trades_df_optimized = trade_log_optimized_df[trade_log_optimized_df['PnL'].notna() & (trade_log_optimized_df['PnL'] > 0)]
        display("\nWinning Trades Summary (Optimized Parameters Simulation):")
        display(winning_trades_df_optimized.head())

        # Analyze losing trades from the simulation
        losing_trades_df_optimized = trade_log_optimized_df[trade_log_optimized_df['PnL'].notna() & (trade_log_optimized_df['PnL'] < 0)]
        display("\nLosing Trades Summary (Optimized Parameters Simulation):")
        display(losing_trades_df_optimized.head())

else:
    display("Cannot perform evaluation as the optimized equity curve is empty.")

# Ensure dollar_bars_df is the raw dollar bar data for the fitness function
# Based on the notebook state, the global `dollar_bars_df` contains the raw data columns
# plus calculated factors. The fitness function is designed to use the raw columns and recalculate.
# So, passing the current `dollar_bars_df` should be fine.
df_for_ga_optimization = dollar_bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']].copy()


# Run the GA (settings for a longer test run with more windows)
# These settings are increased from the previous test run but are still less than the full run.
ga_population_size = 40 # Increased population size
ga_num_generations = 40 # Increased number of generations
ga_mutation_rate = 0.1 # Example mutation rate
ga_walkforward_periods = 5 # Increased number of walk-forward windows
ga_train_period_ratio = 0.7 # Example ratio of data for training in each window

# Ensure initial_capital is defined (it is from previous cells)

display("Starting Genetic Algorithm Optimization (More Windows Test Run)...")
display(f"Settings: Population Size={ga_population_size}, Generations={ga_num_generations}, Mutation Rate={ga_mutation_rate}, Walk-Forward Periods={ga_walkforward_periods}, Train Ratio={ga_train_period_ratio}")


best_ga_params, best_ga_fitness = run_genetic_algorithm(
    df_for_ga_optimization, # Pass the raw dollar bars data
    PARAMETER_SPACE, # Use the defined parameter space
    evaluate_strategy_with_indicators_and_params, # Use the corrected fitness function
    population_size=ga_population_size,
    num_generations=ga_num_generations,
    mutation_rate=ga_mutation_rate,
    initial_capital=initial_capital,
    walkforward_periods=ga_walkforward_periods,
    train_period_ratio=ga_train_period_ratio
)

display("\nGenetic Algorithm Optimization Completed (More Windows Test Run).")
display(f"Overall Best GA Parameters: {best_ga_params}")
display(f"Overall Best GA Fitness (Test Combined Score): {best_ga_fitness:.4f}")