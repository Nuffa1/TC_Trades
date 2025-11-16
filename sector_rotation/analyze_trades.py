import pandas as pd

# Load trade log
df = pd.read_csv('trade_log.csv', index_col=0)

print("\n" + "="*80)
print("BACKTESTER TRADE SUMMARY (2007-2025)")
print("="*80)
print(f"\nTotal Rebalances: {len(df)}")
print(f"Backtest Period: {df.index[0]} to {df.index[-1]}")

# Group by regime
regime_trades = df.groupby('regime').size()
print(f"\nTrades by Regime:")
for regime, count in regime_trades.sort_values(ascending=False).items():
    print(f"  {regime:20s}: {count:3d} trades")

# Average turnover
avg_turnover = df['turnover'].mean()
print(f"\nAverage Turnover per Rebalance: {avg_turnover:.4f} (as % of portfolio)")

# Show sample trades from each regime
print("\n" + "="*80)
print("SAMPLE TRADES FROM EACH REGIME")
print("="*80)

for regime in ['RECESSION', 'EARLY_EXPANSION', 'MID_EXPANSION', 'LATE_EXPANSION', 'RECOVERY']:
    sample = df[df['regime'] == regime].iloc[0] if regime in df['regime'].values else None
    if sample is not None:
        print(f"\n{regime}:")
        print(f"  Date: {sample.name}")
        print(f"  Sectors Available: {sample['sectors_available']}")
        print(f"  Tickers Selected: {sample['tickers_selected']}")
        print(f"  Weights: {sample['weights']}")
        print(f"  Turnover: {sample['turnover']:.4f}")
        print(f"  Portfolio Value: ${sample['portfolio_value']:,.0f}")

# Show trades with highest turnover (most sector switching)
print("\n" + "="*80)
print("BIGGEST SECTOR ROTATIONS (by turnover)")
print("="*80)
top_turns = df.nlargest(5, 'turnover')[['regime', 'tickers_selected', 'turnover', 'portfolio_value']]
for idx, (date, row) in enumerate(top_turns.iterrows(), 1):
    print(f"\n{idx}. {date} - Turnover: {row['turnover']:.4f}")
    print(f"   Regime: {row['regime']}")
    print(f"   Selected: {row['tickers_selected']}")
    print(f"   Portfolio Value: ${row['portfolio_value']:,.0f}")

# Final statistics
print("\n" + "="*80)
print("PORTFOLIO PERFORMANCE")
print("="*80)
equity_df = pd.read_csv('equity_curve.csv', index_col=0)
initial = equity_df['equity'].iloc[0]
final = equity_df['equity'].iloc[-1]
total_return = (final / initial - 1) * 100
cagr = ((final / initial) ** (1 / 18) - 1) * 100  # ~18 years of backtest

print(f"Initial Capital: ${initial:,.0f}")
print(f"Final Capital: ${final:,.0f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Approx CAGR: {cagr:.2f}%")
print(f"Peak Value: ${equity_df['equity'].max():,.0f}")
print(f"Lowest Value: ${equity_df['equity'].min():,.0f}")
