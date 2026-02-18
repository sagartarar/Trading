import pandas as pd
import argparse
import os

def simulate_portfolio(trade_file, initial_capital=100000, max_positions=5):
    if not os.path.exists(trade_file):
        print(f"File {trade_file} not found.")
        return

    # Load trades
    trades = pd.read_csv(trade_file)
    trades['Entry Date'] = pd.to_datetime(trades['Entry Date'])
    trades['Exit Date'] = pd.to_datetime(trades['Exit Date'])
    
    # Sort by Entry Date
    trades = trades.sort_values('Entry Date')
    
    # Simulation State
    cash = initial_capital
    equity = initial_capital
    open_positions = [] # List of dicts: {'Symbol': str, 'Cost': float, 'Value': float, 'Exit Date': datetime}
    position_size_limit = initial_capital / max_positions # Fixed size alloc (e.g. 20k)
    # Alternatively, dynamic sizing: current_equity / max_positions. 
    # Let's use Dynamic sizing based on Current Equity at time of entry? 
    # Or simple fixed fraction. Let's start with Dynamic: Allocation = 20% of Equity.
    
    equity_curve = []
    
    # We need to iterate through time day by day to manage concurrency correctly.
    # Get all unique dates from Entry and Exit
    all_dates = sorted(list(set(trades['Entry Date'].unique()) | set(trades['Exit Date'].unique())))
    
    # Create a map of trades starting on each date
    entries_by_date = trades.groupby('Entry Date')
    
    current_date_idx = 0
    
    for current_date in all_dates:
        # 1. Process Exits (Sell first, then Buy)
        # Check if any open position exits today
        # We assume exit happens at Open/Close? Backtester implies Close or Stop hit intraday.
        # We process exits first to free up cash/slots.
        
        remaining_positions = []
        for pos in open_positions:
            if pos['Exit Date'] <= current_date:
                # Close Position
                # PnL was pre-calculated in trade log, but applied to the allocated amount
                # We need to find the specific trade record to get PnL? 
                # Simpler: The trade log has PnL %. 
                # Final Value = Allocated * (1 + PnL)
                final_value = pos['Allocation'] * (1 + pos['PnL'])
                cash += final_value
                # print(f"{current_date.date()}: Sold {pos['Symbol']} for {final_value:.2f} (Alloc: {pos['Allocation']:.2f})")
            else:
                remaining_positions.append(pos)
        
        open_positions = remaining_positions
        
        # 2. Process Entries
        if current_date in entries_by_date.groups:
            todays_entries = entries_by_date.get_group(current_date)
            
            for _, trade in todays_entries.iterrows():
                # Check constraints
                if len(open_positions) < max_positions:
                    # Determine Position Size
                    # allocation = equity / max_positions # Rebalance on every trade?
                    # Safer: allocation = cash / (max_positions - len(open_positions))? No.
                    # Standard: 20% of Current Total Equity. 
                    current_total_equity = cash + sum([p['Allocation'] for p in open_positions]) # Approximated (positions don't mark to market daily here)
                    target_allocation = current_total_equity / max_positions
                    
                    if cash >= target_allocation:
                        # Enter Trade
                        cash -= target_allocation
                        open_positions.append({
                            'Symbol': trade['Symbol'],
                            'Allocation': target_allocation,
                            'Exit Date': trade['Exit Date'],
                            'PnL': trade['PnL']
                        })
                        # print(f"{current_date.date()}: Bought {trade['Symbol']} with {target_allocation:.2f}")
                    else:
                        pass # Not enough cash to take full position
                else:
                    pass # Max slots full
        
        # 3. Mark to Market (Daily Equity)
        # Since we don't have daily usage data for open positions, we just assume their value = allocation (flat) 
        # until they close. This makes the curve step-like but accurate at realized points.
        # To make it smooth, we'd need daily prices for all open stocks. Too complex for this script.
        # We will record Realized Equity.
        
        current_invested = sum([p['Allocation'] for p in open_positions])
        total_equity = cash + current_invested
        equity_curve.append({'Date': current_date, 'Equity': total_equity, 'Cash': cash, 'Positions': len(open_positions)})

    # Save Curve
    df_curve = pd.DataFrame(equity_curve)
    output_file = trade_file.replace(".csv", "_portfolio.csv")
    df_curve.to_csv(output_file, index=False)
    print(f"Portfolio simulation saved to {output_file}")
    
    # Metrics
    final_equity = df_curve.iloc[-1]['Equity']
    return_pct = (final_equity - initial_capital) / initial_capital
    print(f"Initial: {initial_capital}, Final: {final_equity:.2f}, Return: {return_pct:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trade_file", help="Path to trade log CSV")
    parser.add_argument("--capital", type=float, default=100000)
    args = parser.parse_args()
    
    simulate_portfolio(args.trade_file, args.capital)
