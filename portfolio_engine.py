import pandas as pd
import numpy as np
import yfinance as yf
from indicators import IndicatorLibrary
from scoring import ScoreParser
from pathlib import Path
from datetime import timedelta

class DataCache:
    """Efficient Parquet-based cache for stock data."""

    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, ticker):
        """Generate cache file path."""
        filename = f"{ticker}.parquet"
        return self.cache_dir / filename

    def get(self, ticker):
        """Retrieve cached data if available."""
        cache_path = self._get_cache_path(ticker)

        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            # Fix MultiIndex columns from old cache format
            if isinstance(df.columns, pd.MultiIndex):
                print(f"[CACHE FIX] {ticker}: Converting MultiIndex columns")
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            # Also check for tuple-like strings in column names
            elif len(df.columns) > 0 and isinstance(df.columns[0], tuple):
                print(f"[CACHE FIX] {ticker}: Converting tuple columns")
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            return df
        except Exception as e:
            print(f"Cache read error for {ticker}: {e}")
            return None

    def set(self, ticker, data):
        """Store data in cache as Parquet."""
        cache_path = self._get_cache_path(ticker)
        try:
            data.to_parquet(cache_path, compression='snappy')
        except Exception as e:
            print(f"Cache save error for {ticker}: {e}")

    def exists(self, ticker):
        """Check if ticker data exists in cache."""
        return self._get_cache_path(ticker).exists()

    def get_cache_info(self):
        """Get cache statistics."""
        files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            'total_files': len(files),
            'total_size_mb': total_size / (1024 * 1024),
            'tickers': [f.stem for f in files]
        }

    def clear(self):
        """Clear all cached data."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()

    def delete_ticker(self, ticker):
        """Delete cache for specific ticker."""
        cache_path = self._get_cache_path(ticker)
        if cache_path.exists():
            cache_path.unlink()


class PortfolioEngine:
    def __init__(self, universe, start_date, end_date, initial_capital=100000, use_cache=True, data_source="yahoo"):
        """Initialize portfolio engine.
        
        Args:
            universe: List of stock symbols
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital amount
            use_cache: Whether to use local cache for Yahoo data
            data_source: "yahoo" for Yahoo Finance or "dhan" for Dhan/Hugging Face data
        """
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data_source = data_source  # "yahoo" or "dhan"
        self.data = {}
        self.portfolio_value = []
        self.trades = []
        self.holdings_history = []
        self.parser = ScoreParser()
        self.cache = DataCache() if use_cache else None
        self.regime_index_data = None
        # EQUITY regime filter tracking
        self.equity_regime_analysis = None  # Stores theoretical vs actual comparison data
        self.regime_trigger_events = []  # List of {date, type: 'trigger'/'recovery', drawdown, peak}

    @staticmethod
    def _get_scalar(value):
        """Safely extract scalar from potential Series or DataFrame."""
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return value.iloc[0] if len(value) > 0 else 0
        return value

    def download_and_cache_universe(self, universe_tickers, progress_callback=None, stop_flag=None):
        """Fast batch download with fallback to single ticker download."""
        import time

        # Filter already cached
        tickers_to_download = []
        for ticker in universe_tickers:
            if self.cache and self.cache.exists(ticker):
                continue
            tickers_to_download.append(ticker)

        if not tickers_to_download:
            return len(universe_tickers)

        success_count = 0
        start_time = time.time()
        
        # Convert to .NS format for yfinance
        tickers_ns = [t if t.endswith(('.NS', '.BO')) else f"{t}.NS" for t in tickers_to_download]
        ticker_map = {ns: orig for ns, orig in zip(tickers_ns, tickers_to_download)}
        
        # Download in chunks of 200 (optimal batch size)
        CHUNK_SIZE = 200
        chunks = [tickers_ns[i:i + CHUNK_SIZE] for i in range(0, len(tickers_ns), CHUNK_SIZE)]
        
        completed = 0
        for chunk_idx, chunk in enumerate(chunks):
            if stop_flag and stop_flag[0]:
                print(f"Stopped at chunk {chunk_idx}/{len(chunks)}")
                break
            
            # Batch download with threads
            batch_result = self._download_batch(chunk, ticker_map)
            
            # Count successes - no slow fallback
            for ticker_ns in chunk:
                if batch_result.get(ticker_ns, False):
                    success_count += 1
                completed += 1
            
            # Update progress after each chunk
            if progress_callback:
                elapsed = time.time() - start_time
                avg = elapsed / completed if completed > 0 else 0
                remaining_time = avg * (len(tickers_to_download) - completed)
                try:
                    progress_callback(completed, len(tickers_to_download), f"Batch {chunk_idx + 1}/{len(chunks)}", remaining_time)
                except TypeError:
                    progress_callback(completed, len(tickers_to_download), f"Batch {chunk_idx + 1}")

        if progress_callback:
            try:
                progress_callback(len(tickers_to_download), len(tickers_to_download), "Done", 0)
            except TypeError:
                progress_callback(len(tickers_to_download), len(tickers_to_download), "Done")

        elapsed = time.time() - start_time
        print(f"Downloaded {success_count}/{len(tickers_to_download)} stocks in {elapsed:.1f}s ({elapsed/max(len(tickers_to_download),1):.2f}s/stock)")
        return len(universe_tickers)
    
    def _download_batch(self, tickers, ticker_map):
        """Batch download multiple tickers at once using threads."""
        try:
            data = yf.download(
                tickers,
                period="max",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"Batch download failed: {e}")
            return {}
        
        saved = {}
        
        if isinstance(data.columns, pd.MultiIndex):
            for ticker_ns in tickers:
                try:
                    ticker = ticker_map[ticker_ns]
                    df = data[ticker_ns].dropna(how="all")
                    if not df.empty and len(df) >= 100:
                        df = df.reset_index()
                        # Save without indicators for speed - indicators calculated on fetch
                        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                        df = df[[col for col in expected_cols if col in df.columns]]
                        if self.cache:
                            self.cache.set(ticker, df)
                        saved[ticker_ns] = True
                    else:
                        saved[ticker_ns] = False
                except Exception:
                    saved[ticker_ns] = False
        else:
            # Single ticker result (different format)
            if len(tickers) == 1 and not data.empty:
                ticker_ns = tickers[0]
                ticker = ticker_map[ticker_ns]
                df = data.dropna(how="all").reset_index()
                if len(df) >= 100:
                    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df = df[[col for col in expected_cols if col in df.columns]]
                    if self.cache:
                        self.cache.set(ticker, df)
                    saved[ticker_ns] = True
        
        return saved
    
    def _download_single(self, ticker_ns, ticker, retries=2, backoff=1):
        """Download single ticker with minimal retry."""
        import time
        for attempt in range(1, retries + 1):
            try:
                df = yf.download(ticker_ns, period="max", interval="1d", progress=False)
                if not df.empty and len(df) >= 100:
                    df = df.reset_index()
                    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df = df[[col for col in expected_cols if col in df.columns]]
                    if self.cache:
                        self.cache.set(ticker, df)
                    return True
            except Exception:
                pass
            time.sleep(backoff)
        return False
    
    def _process_and_cache_df(self, ticker, df):
        """Process dataframe and save to cache with indicators."""
        try:
            expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in expected_cols if col in df.columns]]
            
            # Calculate indicators
            try:
                df_with_date_index = df.set_index('Date')
                df_with_date_index = IndicatorLibrary.add_momentum_volatility_metrics(df_with_date_index)
                df_with_date_index = IndicatorLibrary.add_regime_filters(df_with_date_index)
                df = df_with_date_index.reset_index()
            except:
                pass  # Use raw data if indicators fail
            
            if self.cache:
                self.cache.set(ticker, df)
        except Exception as e:
            print(f"Error caching {ticker}: {e}")

    def fetch_data(self, progress_callback=None):
        """Fetch data based on selected data source.
        
        Uses Yahoo Finance (with local cache) or Dhan data from Hugging Face.
        """
        if self.data_source == "dhan":
            return self._fetch_from_huggingface(progress_callback)
        else:
            return self._fetch_from_yahoo(progress_callback)
    
    def _fetch_from_huggingface(self, progress_callback=None):
        """Fetch data from Hugging Face (Dhan API source)."""
        print(f"Loading data from Hugging Face for {len(self.universe)} stocks...")
        
        try:
            from huggingface_manager import HuggingFaceManager, is_hf_configured
            
            if not is_hf_configured():
                print("Hugging Face not configured. Falling back to Yahoo Finance.")
                return self._fetch_from_yahoo(progress_callback)
            
            hf = HuggingFaceManager()
            
        except Exception as e:
            print(f"Could not connect to Hugging Face: {e}. Falling back to Yahoo Finance.")
            return self._fetch_from_yahoo(progress_callback)
        
        def clean_dataframe(df, ticker_name="unknown"):
            """Remove duplicates and ensure clean data."""
            try:
                if df is None or df.empty:
                    return df
                if hasattr(df.index, 'duplicated') and df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
                return df
            except Exception:
                return df
        
        loaded_count = 0
        for i, ticker in enumerate(self.universe):
            if progress_callback:
                progress_callback(i + 1, len(self.universe), ticker)
            
            try:
                df = hf.download_symbol_data(ticker)
                
                if df is not None and not df.empty:
                    # Ensure Date is the index
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    df = clean_dataframe(df, ticker)
                    
                    # Include 300 days BEFORE start_date for indicator lookback
                    extended_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=300)
                    mask = (df.index >= extended_start) & (df.index <= pd.Timestamp(self.end_date))
                    df_filtered = df[mask].copy()
                    
                    if not df_filtered.empty and len(df_filtered) >= 100:
                        self.data[ticker] = df_filtered
                        loaded_count += 1
                        
            except Exception as e:
                print(f"Error loading {ticker} from HF: {e}")
        
        print(f"Successfully loaded {loaded_count} stocks from Hugging Face")
        
        # If very few stocks loaded, suggest downloading data first
        if loaded_count < len(self.universe) * 0.1:
            print("⚠️ Very few stocks available in HF. Please download Broker API data first (Data Download tab).")
        
        return len(self.data) > 0
    
    def _fetch_from_yahoo(self, progress_callback=None):
        """Fetch data from Yahoo Finance with local cache."""
        print(f"Loading data for {len(self.universe)} stocks...")
        tickers_to_download = []

        def clean_dataframe(df, ticker_name="unknown"):
            """Remove duplicates, detect anomalies, and ensure clean data."""
            try:
                if df is None or df.empty:
                    return df
                # Remove duplicate indices (dates)
                if hasattr(df.index, 'duplicated') and df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='last')]
                # Sort by date
                df = df.sort_index()
                
                # DATA QUALITY: Detect extreme single-day price changes (>50%)
                # Just warn, don't fix
                if 'Close' in df.columns and len(df) > 3:
                    close = df['Close']
                    pct_change = close.pct_change().abs()
                    for i in range(1, len(pct_change) - 1):
                        if pct_change.iloc[i] > 0.5:  # >50% change
                            next_change = pct_change.iloc[i + 1] if i + 1 < len(pct_change) else 0
                            if next_change > 0.3:  # If it reverses next day
                                print(f"⚠️ DATA ANOMALY: {ticker_name} on {df.index[i].date()} - {pct_change.iloc[i]*100:.1f}% drop, then {next_change*100:.1f}% recovery")
                
                return df
            except Exception:
                return df  # Return unchanged if cleaning fails

        # First, try to load from cache
        for i, ticker in enumerate(self.universe):
            if progress_callback:
                progress_callback(i + 1, len(self.universe), ticker)

            if self.cache:
                cached_data = self.cache.get(ticker)
                if cached_data is not None:
                    try:
                        # Fix index - Date should be the index
                        if 'Date' in cached_data.columns:
                            cached_data['Date'] = pd.to_datetime(cached_data['Date'])
                            cached_data.set_index('Date', inplace=True)

                        # Ensure index is datetime
                        if not isinstance(cached_data.index, pd.DatetimeIndex):
                            cached_data.index = pd.to_datetime(cached_data.index)

                        # Clean data - remove duplicate indices and fix anomalies
                        cached_data = clean_dataframe(cached_data, ticker)

                        # Include 300 days BEFORE start_date for indicator lookback
                        # 6-month performance needs ~130 days, 1-year needs ~260 days
                        extended_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=300)
                        mask = (cached_data.index >= extended_start) & \
                               (cached_data.index <= pd.Timestamp(self.end_date))
                        df_filtered = cached_data[mask].copy()
                        
                        # Flatten MultiIndex columns if present
                        if isinstance(df_filtered.columns, pd.MultiIndex):
                            df_filtered.columns = [col[0] if isinstance(col, tuple) else col for col in df_filtered.columns]

                        if not df_filtered.empty and len(df_filtered) >= 100:
                            self.data[ticker] = df_filtered
                            continue
                    except Exception as e:
                        print(f"Error loading {ticker}: {e}")

            # If not in cache or insufficient data, mark for download
            tickers_to_download.append(ticker)

        # Download missing tickers (with indicators calculated automatically)
        if tickers_to_download:
            print(f"Downloading {len(tickers_to_download)} missing stocks...")
            self.download_and_cache_universe(tickers_to_download, progress_callback)

            # Retry loading after download
            for ticker in tickers_to_download:
                if self.cache:
                    cached_data = self.cache.get(ticker)
                    if cached_data is not None:
                        try:
                            # Fix index - Date should be the index
                            if 'Date' in cached_data.columns:
                                cached_data['Date'] = pd.to_datetime(cached_data['Date'])
                                cached_data.set_index('Date', inplace=True)

                            # Ensure index is datetime
                            if not isinstance(cached_data.index, pd.DatetimeIndex):
                                cached_data.index = pd.to_datetime(cached_data.index)

                            # Clean data
                            cached_data = clean_dataframe(cached_data, ticker)

                            # Include 300 days BEFORE start_date for indicator lookback
                            extended_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=300)
                            mask = (cached_data.index >= extended_start) & \
                                   (cached_data.index <= pd.Timestamp(self.end_date))
                            df_filtered = cached_data[mask].copy()
                            
                            # Flatten MultiIndex columns if present
                            if isinstance(df_filtered.columns, pd.MultiIndex):
                                df_filtered.columns = [col[0] if isinstance(col, tuple) else col for col in df_filtered.columns]

                            if not df_filtered.empty:
                                self.data[ticker] = df_filtered
                        except Exception as e:
                            print(f"Error loading {ticker} after download: {e}")

        print(f"Successfully loaded {len(self.data)} stocks")
        return len(self.data) > 0
    
    def calculate_indicators_for_formula(self, formula, regime_config=None):
        """Calculate only the indicators needed for the formula and regime filter."""
        # Determine which indicator types are needed
        needs_momentum = any(x in formula.upper() for x in ['PERFORMANCE', 'SHARPE', 'SORTINO', 'CALMAR', 'VOLATILITY', 'DRAWDOWN'])
        needs_regime = regime_config is not None and regime_config.get('type') != 'EQUITY'
        
        if not needs_momentum and not needs_regime:
            return  # No indicators needed
        
        # Extract required periods from formula (e.g., "15 Month Performance" -> (15, 'Performance'))
        required_periods = None
        if needs_momentum:
            from scoring import ScoreParser
            parser = ScoreParser()
            required_periods = parser.extract_required_periods(formula)
            print(f"Calculating indicators (momentum={needs_momentum}, regime={needs_regime}, periods={required_periods})...")
        else:
            print(f"Calculating indicators (momentum={needs_momentum}, regime={needs_regime})...")
        
        for ticker in self.data:
            try:
                df = self.data[ticker]
                
                # Flatten columns first if needed (fix for cloud cache issues)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                elif len(df.columns) > 0 and isinstance(df.columns[0], tuple):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                # Only calculate if not already calculated
                if needs_momentum and '6 Month Performance' not in df.columns:
                    df = IndicatorLibrary.add_momentum_volatility_metrics(df, required_periods)
                elif needs_momentum and required_periods:
                    # Check if any required periods are missing and add them
                    df = IndicatorLibrary.add_momentum_volatility_metrics(df, required_periods)
                    
                if needs_regime and 'EMA_200' not in df.columns:
                    df = IndicatorLibrary.add_regime_filters(df)
                
                # Flatten again after indicators (some libraries create MultiIndex)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                self.data[ticker] = df
            except Exception as e:
                print(f"Error calculating indicators for {ticker}: {e}")

    def _get_rebalance_dates(self, all_dates, rebal_config):
        """Generate rebalance dates based on config. Ensures every period has a rebalance."""
        freq = rebal_config['frequency']
        all_dates_set = set(all_dates)
        
        if freq == 'Weekly':
            # Get day of week (0=Monday, 4=Friday)
            day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
            target_day = day_map[rebal_config['day']]
            
            rebalance_dates = [d for d in all_dates if d.weekday() == target_day]
        
        elif freq == 'Every 2 Weeks':
            # Every 2 weeks on specified day
            day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
            target_day = day_map[rebal_config['day']]
            
            # Get all dates on target day
            matching_dates = [d for d in all_dates if d.weekday() == target_day]
            # Take every other one
            rebalance_dates = matching_dates[::2]
        
        else:  # Monthly, Bi-Monthly, Quarterly, Half-Yearly, Annually
            target_date = rebal_config['date']
            alt_option = rebal_config.get('alt_day', 'Next Day')
            
            # Determine the skip interval (how many months between rebalances)
            freq_to_skip = {
                'Monthly': 1,
                'Bi-Monthly': 2,
                'Quarterly': 3,
                'Half-Yearly': 6,
                'Annually': 12
            }
            skip_months = freq_to_skip.get(freq, 1)
            
            rebalance_dates = []
            
            # Group dates by (year, month)
            month_groups = {}
            for date in all_dates:
                key = (date.year, date.month)
                if key not in month_groups:
                    month_groups[key] = []
                month_groups[key].append(date)
            
            # Get sorted month keys
            sorted_months = sorted(month_groups.keys())
            
            # Select months at the specified interval (starting from first available)
            selected_months = sorted_months[::skip_months]
            
            # For each selected month, find the best rebalance date
            for (year, month) in selected_months:
                month_dates_sorted = sorted(month_groups[(year, month)])
                rebalance_date = None
                
                # First, try to find exact target date
                for d in month_dates_sorted:
                    if d.day == target_date:
                        rebalance_date = d
                        break
                
                # If not found, use alternative
                if rebalance_date is None:
                    if alt_option == 'Previous Day':
                        # Find the closest trading day BEFORE target date
                        for d in reversed(month_dates_sorted):
                            if d.day < target_date:
                                rebalance_date = d
                                break
                        # If no day before, take the first available day
                        if rebalance_date is None and month_dates_sorted:
                            rebalance_date = month_dates_sorted[0]
                    else:  # Next Day
                        # Find the closest trading day AFTER target date
                        for d in month_dates_sorted:
                            if d.day > target_date:
                                rebalance_date = d
                                break
                        # If no day after, take the last available day
                        if rebalance_date is None and month_dates_sorted:
                            rebalance_date = month_dates_sorted[-1]
                
                if rebalance_date:
                    rebalance_dates.append(rebalance_date)
        
        
        print(f"Generated {len(rebalance_dates)} rebalance dates from {len(all_dates)} trading days ({freq})")
        return sorted(rebalance_dates)

    def _check_stock_regime_filter(self, ticker, date, regime_config):
        """Check if regime filter is triggered for a specific stock.
        
        Used when regime_config['index'] == 'Stock' to apply per-stock filtering.
        
        Returns: (triggered: bool)
        """
        if ticker not in self.data:
            return False
        
        df = self.data[ticker]
        if date not in df.index:
            return False
        
        row = df.loc[date]
        regime_type = regime_config['type']
        
        # Helper to extract scalar from potential Series
        def get_scalar(val):
            if hasattr(val, 'iloc'):
                return float(val.iloc[0])
            return float(val) if val is not None else 0.0
        
        if regime_type == 'EMA':
            ema_period = regime_config['value']
            ema_col = f'EMA_{ema_period}'
            close_price = get_scalar(row.get('Close', 0))
            ema_value = get_scalar(row.get(ema_col, 0))
            
            # Triggered when stock closes BELOW its EMA
            if ema_col in row.index and ema_value > 0 and close_price < ema_value:
                return True
        
        elif regime_type == 'MACD':
            macd_val = get_scalar(row.get('MACD', 0))
            signal_val = get_scalar(row.get('MACD_Signal', 0))
            # Triggered when MACD is below signal line
            if macd_val < signal_val:
                return True
        
        elif regime_type == 'SUPERTREND':
            st_direction = row.get('Supertrend_Direction', 'BUY')
            if hasattr(st_direction, 'iloc'):
                st_direction = st_direction.iloc[0]
            # Triggered when SuperTrend says SELL
            if st_direction == 'SELL':
                return True
        
        return False

    def _check_risk_management(self, date, holdings, entry_prices, risk_config):
        """
        Check portfolio and trade level risk triggers using daily Low as intraday proxy.
        
        Args:
            date: Current date
            holdings: {ticker: shares}
            entry_prices: {ticker: cost_basis per share}
            risk_config: {'portfolio': {...}, 'trade': {...}}
        
        Returns:
            (triggered: bool, tickers_to_exit: list, trigger_reason: str)
        """
        if not risk_config:
            return False, [], ""
        
        portfolio_config = risk_config.get('portfolio', {})
        trade_config = risk_config.get('trade', {})
        
        if not portfolio_config.get('enabled') and not trade_config.get('enabled'):
            return False, [], ""
        
        if not holdings:
            return False, [], ""
        
        # Calculate current values using daily LOW as worst-case intraday proxy
        position_losses = {}  # {ticker: (loss_amount, current_value, is_losing)}
        total_loss = 0
        total_invested = 0
        losing_tickers = []
        
        for ticker, shares in holdings.items():
            if ticker not in self.data or shares <= 0:
                continue
            
            df = self.data[ticker]
            if date not in df.index:
                continue
            
            entry_price = entry_prices.get(ticker, 0)
            if entry_price <= 0:
                continue
            
            # Use daily LOW as worst-case intraday price
            row = df.loc[date]
            low_price = row['Low'] if 'Low' in row.index else row['Close']
            if hasattr(low_price, 'iloc'):
                low_price = low_price.iloc[0]
            
            position_value = shares * low_price
            cost_basis = shares * entry_price
            loss = cost_basis - position_value  # Positive = loss, Negative = profit
            
            position_losses[ticker] = {
                'loss': loss,
                'cost': cost_basis,
                'current': position_value,
                'is_losing': loss > 0
            }
            
            if loss > 0:
                total_loss += loss
                losing_tickers.append(ticker)
            
            total_invested += cost_basis
        
        tickers_to_exit = []
        trigger_reason = ""
        
        # Check Portfolio-level risk
        if portfolio_config.get('enabled') and total_invested > 0:
            threshold = portfolio_config['value']
            if portfolio_config['type'] == 'percent':
                threshold = total_invested * (portfolio_config['value'] / 100)
            
            if total_loss >= threshold:
                action = portfolio_config.get('action', 'exit_losers')
                if action == 'exit_all':
                    tickers_to_exit = list(holdings.keys())
                    trigger_reason = f"Portfolio loss ₹{total_loss:,.0f} exceeded threshold ₹{threshold:,.0f}"
                else:  # exit_losers
                    tickers_to_exit = losing_tickers.copy()
                    trigger_reason = f"Portfolio loss ₹{total_loss:,.0f} exceeded threshold - exiting losers"
                
                print(f"RISK [{date}]: {trigger_reason}")
                return True, tickers_to_exit, trigger_reason
        
        # Check Trade-level risk
        if trade_config.get('enabled'):
            breached_tickers = []
            
            for ticker, data in position_losses.items():
                if not data['is_losing']:
                    continue
                
                threshold = trade_config['value']
                if trade_config['type'] == 'percent':
                    threshold = data['cost'] * (trade_config['value'] / 100)
                
                if data['loss'] >= threshold:
                    breached_tickers.append(ticker)
                    print(f"RISK [{date}]: {ticker} loss ₹{data['loss']:,.0f} exceeded threshold ₹{threshold:,.0f}")
            
            if breached_tickers:
                action = trade_config.get('action', 'exit_breached')
                if action == 'exit_all':
                    tickers_to_exit = list(holdings.keys())
                    trigger_reason = f"Trade risk breached on {breached_tickers} - exiting all"
                elif action == 'exit_losers':
                    tickers_to_exit = losing_tickers.copy()
                    trigger_reason = f"Trade risk breached on {breached_tickers} - exiting all losers"
                else:  # exit_breached
                    tickers_to_exit = breached_tickers.copy()
                    trigger_reason = f"Trade risk breached - exiting {breached_tickers}"
                
                return True, tickers_to_exit, trigger_reason
        
        return False, [], ""

    def _check_regime_filter(self, date, regime_config, current_equity=0, peak_equity=0):
        """Check if regime filter is triggered.
        
        For EQUITY type: checks drawdown from peak equity
        For other types: checks technical indicators on index
        
        Returns: (triggered: bool, action: str, drawdown_pct: float)
        """
        if not regime_config:
            return False, 'none', 0.0  # No filter active
        
        regime_type = regime_config['type']
        
        if regime_type == 'EQUITY':
            # Check drawdown from peak equity
            sl_pct = regime_config['value']
            if peak_equity > 0:
                drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
            else:
                drawdown_pct = 0.0
            
            if drawdown_pct > sl_pct:
                return True, regime_config['action'], drawdown_pct
            return False, 'none', drawdown_pct
        
        if regime_type == 'EQUITY_MA':
            # EQUITY_MA is handled separately in the main loop
            # This is just a placeholder - actual check done in run_rebalance_strategy
            return False, 'none', 0.0
        
        # For EMA, MACD, SUPERTREND - need index data
        if self.regime_index_data is None or self.regime_index_data.empty:
            return False, 'none', 0.0
        
        # Use nearest available date if exact date not found (handles holidays)
        if date not in self.regime_index_data.index:
            nearest = self.regime_index_data.index.asof(date)
            if pd.isna(nearest):
                return False, 'none', 0.0
            row = self.regime_index_data.loc[nearest]
        else:
            row = self.regime_index_data.loc[date]
        
        # Helper to extract scalar from potential Series
        def get_scalar(val):
            if hasattr(val, 'iloc'):
                return float(val.iloc[0])
            return float(val) if val is not None else 0.0
        
        # SMA timeframe variants
        if regime_type in ['SMA_1D', 'SMA_1W', 'SMA_1M']:
            # Map to correct direction column
            direction_col = f'{regime_type}_Direction'
            sma_direction = row.get(direction_col, 'BUY')
            if hasattr(sma_direction, 'iloc'):
                sma_direction = sma_direction.iloc[0]
            triggered = sma_direction == 'SELL'
            close_price = get_scalar(row.get('Close', 0))
            sma_value = get_scalar(row.get(regime_type, 0))
            print(f"REGIME CHECK [{date}]: {regime_type} Close={close_price:.2f}, Value={sma_value:.2f}, Direction={sma_direction}, Triggered={triggered}")
            if triggered:
                return True, regime_config['action'], 0.0
        
        # EMA timeframe variants
        elif regime_type in ['EMA', 'EMA_1D', 'EMA_1W', 'EMA_1M']:
            # Handle legacy EMA type
            if regime_type == 'EMA':
                ema_period = regime_config['value']
                ema_col = f'EMA_{ema_period}'
                close_price = get_scalar(row.get('Close', 0))
                ema_value = get_scalar(row.get(ema_col, 0))
                triggered = ema_col in row.index and ema_value > 0 and close_price < ema_value
                print(f"REGIME CHECK [{date}]: Close={close_price:.2f}, {ema_col}={ema_value:.2f}, Triggered={triggered}")
            else:
                # New EMA timeframe variants
                direction_col = f'{regime_type}_Direction'
                ema_direction = row.get(direction_col, 'BUY')
                if hasattr(ema_direction, 'iloc'):
                    ema_direction = ema_direction.iloc[0]
                triggered = ema_direction == 'SELL'
                close_price = get_scalar(row.get('Close', 0))
                ema_value = get_scalar(row.get(regime_type, 0))
                print(f"REGIME CHECK [{date}]: {regime_type} Close={close_price:.2f}, Value={ema_value:.2f}, Direction={ema_direction}, Triggered={triggered}")
            
            # Triggered when index closes BELOW EMA
            if triggered:
                return True, regime_config['action'], 0.0
        
        elif regime_type == 'MACD':
            macd_val = get_scalar(row.get('MACD', 0))
            signal_val = get_scalar(row.get('MACD_Signal', 0))
            triggered = macd_val < signal_val
            print(f"REGIME CHECK [{date}]: MACD={macd_val:.2f}, Signal={signal_val:.2f}, Triggered={triggered}")
            if triggered:
                return True, regime_config['action'], 0.0
        
        elif regime_type in ['SUPERTREND', 'SUPERTREND_1D', 'SUPERTREND_1W', 'SUPERTREND_1M']:
            # Use appropriate Supertrend_Direction column based on timeframe
            # Map to correct column: SUPERTREND/SUPERTREND_1D -> Supertrend_Direction, others use suffix
            if regime_type in ['SUPERTREND', 'SUPERTREND_1D']:
                direction_col = 'Supertrend_Direction'
            elif regime_type == 'SUPERTREND_1W':
                direction_col = 'Supertrend_1W_Direction'
            else:  # SUPERTREND_1M
                direction_col = 'Supertrend_1M_Direction'
            
            st_direction = row.get(direction_col, 'BUY')
            if hasattr(st_direction, 'iloc'):
                st_direction = st_direction.iloc[0]
            triggered = st_direction == 'SELL'
            print(f"REGIME CHECK [{date}]: {regime_type} Direction={st_direction}, Triggered={triggered}")
            if triggered:
                return True, regime_config['action'], 0.0
        
        elif regime_type == 'DONCHIAN':
            # Donchian Channel regime filter (Turtle Trading rules)
            # Stateful: once triggered, stays triggered until recovery
            exit_period = regime_config.get('exit_period', 55)
            recovery_period = regime_config.get('recovery_period', 20)
            
            low_col = f'Donchian_Low_{exit_period}'
            high_col = f'Donchian_High_{recovery_period}'
            
            close_price = get_scalar(row.get('Close', 0))
            donchian_low = get_scalar(row.get(low_col, 0))
            donchian_high = get_scalar(row.get(high_col, 0))
            
            # Check if currently in triggered state (stored in regime_config)
            is_donchian_triggered = regime_config.get('_donchian_active', False)
            
            if is_donchian_triggered:
                # Check for recovery: close above donchian high
                if close_price > donchian_high and donchian_high > 0:
                    regime_config['_donchian_active'] = False
                    print(f"🟢 DONCHIAN RECOVERED [{date}]: Close={close_price:.2f} > {high_col}={donchian_high:.2f}")
                    return False, 'none', 0.0
                else:
                    print(f"⏳ DONCHIAN STILL ACTIVE [{date}]: Close={close_price:.2f}, Waiting for > {donchian_high:.2f}")
                    return True, regime_config['action'], 0.0
            else:
                # Check for trigger: close below donchian low
                if close_price < donchian_low and donchian_low > 0:
                    regime_config['_donchian_active'] = True
                    print(f"🔴 DONCHIAN TRIGGERED [{date}]: Close={close_price:.2f} < {low_col}={donchian_low:.2f}")
                    return True, regime_config['action'], 0.0
        
        elif regime_type == 'SWING_ATR':
            # Swing pivot with ATR buffer
            swing_period = regime_config.get('swing_period', 20)
            atr_buffer = regime_config.get('atr_buffer', 1.5)
            
            low_col = f'Swing_Low_{swing_period}'
            high_col = f'Swing_High_{swing_period}'
            
            close_price = get_scalar(row.get('Close', 0))
            swing_low = get_scalar(row.get(low_col, 0))
            swing_high = get_scalar(row.get(high_col, 0))
            atr = get_scalar(row.get('ATR_14', 0))
            
            # Buffered levels
            exit_level = swing_low - (atr_buffer * atr)
            recovery_level = swing_high + (atr_buffer * atr)
            
            # Check if currently in triggered state
            is_swing_triggered = regime_config.get('_swing_active', False)
            
            if is_swing_triggered:
                # Check for recovery
                if close_price > recovery_level and recovery_level > 0:
                    regime_config['_swing_active'] = False
                    print(f"🟢 SWING_ATR RECOVERED [{date}]: Close={close_price:.2f} > Recovery={recovery_level:.2f}")
                    return False, 'none', 0.0
                else:
                    print(f"⏳ SWING_ATR STILL ACTIVE [{date}]: Close={close_price:.2f}, Waiting for > {recovery_level:.2f}")
                    return True, regime_config['action'], 0.0
            else:
                # Check for trigger
                if close_price < exit_level and exit_level > 0:
                    regime_config['_swing_active'] = True
                    print(f"🔴 SWING_ATR TRIGGERED [{date}]: Close={close_price:.2f} < Exit={exit_level:.2f} (Swing={swing_low:.2f} - {atr_buffer}×ATR={atr:.2f})")
                    return True, regime_config['action'], 0.0
        
        elif regime_type == 'BREADTH':
            # Market breadth regime filter
            # Check percentage of constituent stocks above 200 SMA
            breadth_pct = self._calculate_market_breadth(date, regime_config)
            threshold = regime_config.get('breadth_threshold', 60)
            hysteresis = regime_config.get('breadth_hysteresis', 5)
            
            is_breadth_triggered = regime_config.get('_breadth_active', False)
            
            if is_breadth_triggered:
                # Check for recovery (with hysteresis to avoid whipsaw)
                recovery_threshold = threshold + hysteresis
                if breadth_pct >= recovery_threshold:
                    regime_config['_breadth_active'] = False
                    print(f"🟢 BREADTH RECOVERED [{date}]: {breadth_pct:.1f}% >= {recovery_threshold}%")
                    return False, 'none', 0.0
                else:
                    print(f"⏳ BREADTH STILL ACTIVE [{date}]: {breadth_pct:.1f}% < {recovery_threshold}% (waiting for recovery)")
                    return True, regime_config['action'], 0.0
            else:
                # Check for trigger
                if breadth_pct < threshold:
                    regime_config['_breadth_active'] = True
                    print(f"🔴 BREADTH TRIGGERED [{date}]: {breadth_pct:.1f}% < {threshold}%")
                    return True, regime_config['action'], 0.0
                else:
                    print(f"✅ BREADTH OK [{date}]: {breadth_pct:.1f}% >= {threshold}%")
        
        return False, 'none', 0.0
    
    def _calculate_market_breadth(self, date, regime_config):
        """Calculate percentage of constituent stocks above 200 SMA.
        
        Loads constituent data from historical_constituents store and
        checks each stock's position relative to its 200 SMA.
        
        Returns: float (0-100) representing percentage above 200 SMA
        """
        try:
            # Get the breadth index (which index's constituents to use)
            breadth_index = regime_config.get('breadth_index', 'NIFTY50')
            
            # Load constituent data for the given date
            from historical_constituents.store import load_all_snapshots, get_available_indices
            
            # Check if we have cached constituents for this index
            if not hasattr(self, '_breadth_constituents_cache'):
                self._breadth_constituents_cache = {}
            
            if breadth_index not in self._breadth_constituents_cache:
                # Load all snapshots for this index
                available = get_available_indices()
                index_key = breadth_index.lower().replace(' ', '')
                
                if index_key.upper() in [i.upper() for i in available]:
                    snapshots = load_all_snapshots(index_key)
                    self._breadth_constituents_cache[breadth_index] = snapshots
                else:
                    print(f"   [BREADTH] No constituent data for {breadth_index}")
                    return 100.0  # Default to not triggered if no data
            
            snapshots = self._breadth_constituents_cache.get(breadth_index, [])
            if not snapshots:
                return 100.0
            
            # Find the appropriate snapshot for this date (most recent before date)
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            target_quarter = None
            
            for snapshot in reversed(snapshots):
                if snapshot.effective_date <= date_str:
                    target_quarter = snapshot
                    break
            
            if target_quarter is None and snapshots:
                target_quarter = snapshots[0]  # Use earliest if date is before all snapshots
            
            if target_quarter is None:
                return 100.0
            
            constituents = target_quarter.symbols
            
            # Count stocks above 200 SMA
            above_200_count = 0
            total_valid = 0
            
            for ticker in constituents:
                if ticker in self.data:
                    df = self.data[ticker]
                    if date in df.index:
                        row = df.loc[date]
                        close = row.get('Close', 0)
                        sma_200 = row.get('SMA_200', 0)
                        
                        # Extract scalar if needed
                        if hasattr(close, 'iloc'):
                            close = float(close.iloc[0])
                        if hasattr(sma_200, 'iloc'):
                            sma_200 = float(sma_200.iloc[0])
                        
                        if sma_200 > 0:  # Valid SMA exists
                            total_valid += 1
                            if close > sma_200:
                                above_200_count += 1
            
            if total_valid == 0:
                return 100.0  # Default to not triggered if no valid data
            
            breadth_pct = (above_200_count / total_valid) * 100
            return breadth_pct
            
        except Exception as e:
            print(f"   [BREADTH] Error calculating breadth: {e}")
            return 100.0  # Default to not triggered on error

    def _apply_put_hedge(self, date, cash, holdings, put_hedge_df, put_hedge_config):
        """
        Buy NIFTY ATM Weekly Puts as a delta-neutral portfolio hedge.

        Lot calculation (delta-neutral):
            portfolio_NIFTY_units = portfolio_value × beta / nifty_spot
            lots = portfolio_NIFTY_units × hedge_ratio / (ATM_delta × lot_size)

        Returns:
            (cash_after, hedge_position_dict, cost_paid)
        """
        from nifty_put_hedge import (
            get_put_premium_on_date, get_nifty_spot_on_date,
            get_atm_strike, get_next_expiry, get_option_ticker_name,
            delta_neutral_lots, get_nifty_lot_size, build_fallback_put_series,
        )

        as_of_date   = date.date() if hasattr(date, 'date') else date
        hedge_ratio  = put_hedge_config.get('hedge_ratio', 1.0)  # 1.0 = full delta neutral
        beta         = put_hedge_config.get('portfolio_beta', 1.0)

        # ── Current portfolio value (cash + stock holdings) ─────────────────
        portfolio_value = cash
        for ticker, shares in holdings.items():
            if ticker in self.data and date in self.data[ticker].index:
                portfolio_value += shares * self._get_scalar(self.data[ticker].loc[date, 'Close'])

        # ── NIFTY spot ───────────────────────────────────────────────────────
        nifty_spot = get_nifty_spot_on_date(
            date if isinstance(date, pd.Timestamp) else pd.Timestamp(date),
            put_hedge_df,
        )

        if nifty_spot <= 0:
            print(f"[PUT HEDGE] Cannot determine NIFTY spot on {date}. Skipping hedge.")
            return cash, {}, 0.0

        # ── ATM strike & next expiry ─────────────────────────────────────────
        atm_strike  = get_atm_strike(nifty_spot)
        expiry_type = put_hedge_config.get('expiry_type', 'WEEKLY')
        expiry_date = get_next_expiry(as_of_date, expiry_type=expiry_type)
        option_ticker = get_option_ticker_name(atm_strike, expiry_date)

        # ── Put premium ──────────────────────────────────────────────────────
        put_premium = get_put_premium_on_date(
            date if isinstance(date, pd.Timestamp) else pd.Timestamp(date),
            put_hedge_df,
        )

        if put_premium <= 0:
            # On-the-fly VIX/B-S fallback for this single date
            fallback_df = build_fallback_put_series(as_of_date, as_of_date, expiry_type=expiry_type)
            if fallback_df is not None and not fallback_df.empty:
                put_premium = get_put_premium_on_date(
                    pd.Timestamp(as_of_date), fallback_df
                )

        if put_premium <= 0:
            print(f"[PUT HEDGE] No put premium for {date}. Skipping hedge.")
            return cash, {}, 0.0

        # ── Delta-neutral lots ───────────────────────────────────────────────
        lot_size = get_nifty_lot_size(as_of_date)
        n_lots   = delta_neutral_lots(portfolio_value, nifty_spot, as_of_date,
                                      hedge_ratio=hedge_ratio, beta=beta)
        if n_lots <= 0:
            print(f"[PUT HEDGE] Portfolio too small for even 1 lot. Spot={nifty_spot:.0f}")
            return cash, {}, 0.0

        cost = n_lots * lot_size * put_premium
        # Trim if insufficient cash
        while cost > cash and n_lots > 0:
            n_lots -= 1
            cost = n_lots * lot_size * put_premium

        if n_lots <= 0:
            print(f"[PUT HEDGE] Insufficient cash (₹{cash:.0f}) for hedge.")
            return cash, {}, 0.0

        cash -= cost
        self.trades.append({
            'Date':        date,
            'Ticker':      option_ticker,           # e.g. NIFTY25000PE03JUL2025
            'Action':      'BUY_HEDGE',
            'Shares':      n_lots * lot_size,       # total quantity
            'Price':       put_premium,
            'Value':       cost,
            'Score':       0,
            'Rank':        'Put Hedge',
            'Reason':      'REGIME_HEDGE',
            'Strike':      atm_strike,
            'Expiry':      str(expiry_date),
            'NIFTY_Spot':  nifty_spot,
            'Lots':        n_lots,
        })

        hedge_position = {
            'lots':         n_lots,
            'premium_paid': put_premium,
            'entry_date':   date,
            'lot_size':     lot_size,
            'atm_strike':   atm_strike,
            'expiry_date':  expiry_date,
            'option_ticker': option_ticker,
        }
        print(f"[PUT HEDGE] Bought {n_lots} lots {option_ticker} @ ₹{put_premium:.2f}"
              f" | NIFTY={nifty_spot:.0f} | Cost=₹{cost:,.0f}")
        return cash, hedge_position, cost

    def _close_put_hedge(self, date, hedge_position, put_hedge_df):
        """
        Close (sell) an existing NIFTY Put hedge position.

        Returns:
            proceeds (float) — cash received from closing the puts.
        """
        if not hedge_position:
            return 0.0

        from nifty_put_hedge import get_put_premium_on_date, build_fallback_put_series

        n_lots        = hedge_position.get('lots', 0)
        lot_size      = hedge_position.get('lot_size', 75)
        entry_premium = hedge_position.get('premium_paid', 0)
        option_ticker = hedge_position.get('option_ticker', 'NIFTY_ATM_PE')

        if n_lots <= 0:
            return 0.0

        # Current put price
        exit_premium = get_put_premium_on_date(
            date if isinstance(date, pd.Timestamp) else pd.Timestamp(date),
            put_hedge_df,
        )

        if exit_premium <= 0:
            as_of = date.date() if hasattr(date, 'date') else date
            fallback_df = build_fallback_put_series(as_of, as_of)
            if fallback_df is not None and not fallback_df.empty:
                exit_premium = get_put_premium_on_date(pd.Timestamp(as_of), fallback_df)

        # Puts likely decayed if market recovered → assume 30% residual
        if exit_premium <= 0:
            exit_premium = entry_premium * 0.30

        proceeds = n_lots * lot_size * exit_premium
        pnl      = proceeds - (n_lots * lot_size * entry_premium)

        self.trades.append({
            'Date':       date,
            'Ticker':     option_ticker,
            'Action':     'SELL_HEDGE',
            'Shares':     n_lots * lot_size,
            'Price':      exit_premium,
            'Value':      proceeds,
            'Score':      0,
            'Rank':       'Put Hedge',
            'Reason':     'REGIME_RECOVERY',
            'Strike':     hedge_position.get('atm_strike', 0),
            'Expiry':     str(hedge_position.get('expiry_date', '')),
            'Lots':       n_lots,
        })
        print(f"[PUT HEDGE] Closed {n_lots} lots {option_ticker} @ ₹{exit_premium:.2f}"
              f" | Entry=₹{entry_premium:.2f} | P&L=₹{pnl:,.0f}")
        return proceeds


    def run_rebalance_strategy(self, scoring_formula, num_stocks, exit_rank, 
                              rebal_config, regime_config=None, uncorrelated_config=None, 
                              reinvest_profits=True, position_sizing_config=None,
                              historical_universe_config=None, risk_config=None):
        """
        Advanced backtesting engine with all Sigma Scanner features.
        
        position_sizing_config: dict with 'method' (equal_weight, inverse_volatility, 
                               score_weighted, risk_parity), 'use_cap' (bool), 'max_pct' (int)
        historical_universe_config: dict with 'enabled' (bool), 'universe_name' (str)
                                   If enabled, uses point-in-time index constituents
        risk_config: dict with 'portfolio' and 'trade' sub-configs for intraday risk management
        """
        if not self.data:
            print("No data available")
            return
        
        # Validate formula
        is_valid, msg = self.parser.validate_formula(scoring_formula)
        if not is_valid:
            print(f"Invalid formula: {msg}")
            return
        
        # Calculate indicators on-demand based on formula
        self.calculate_indicators_for_formula(scoring_formula, regime_config)
        
        # Load regime filter index data if needed (skip for EQUITY, EQUITY_MA, and Stock-level filtering)
        if regime_config and regime_config['type'] not in ['EQUITY', 'EQUITY_MA'] and regime_config.get('index') != 'Stock':
            regime_index = regime_config['index']
            # Map universe names to Yahoo Finance tickers
            index_map = {
                # Broad Market Indices
                'NIFTY 50': '^NSEI',
                'NIFTY NEXT 50': '^NSMIDCP',
                'NIFTY 100': '^CNX100',
                'NIFTY 200': '^CNX200',
                'NIFTY 500': '^CRSLDX',
                'NIFTY MIDCAP 50': '^NSEMDCP50',
                'NIFTY MIDCAP 100': '^CNXMC',
                'NIFTY SMALLCAP 50': '^NIFTYSMCP50',
                'NIFTY SMALLCAP 100': '^CNXSC',
                'NIFTY LARGEMIDCAP 250': '^CNXLM250',
                'NIFTY MIDSMALLCAP 400': '^CNXMSC400',
                # Sectoral Indices
                'NIFTY BANK': '^NSEBANK',
                'NIFTY FINANCIAL SERVICES': '^CNXFINANCE',
                'NIFTY IT': '^CNXIT',
                'NIFTY PHARMA': '^CNXPHARMA',
                'NIFTY AUTO': '^CNXAUTO',
                'NIFTY FMCG': '^CNXFMCG',
                'NIFTY METAL': '^CNXMETAL',
                'NIFTY REALTY': '^CNXREALTY',
                'NIFTY ENERGY': '^CNXENERGY',
                'NIFTY CONSUMPTION': '^CNXCONSUM',
                'NIFTY MEDIA': '^CNXMEDIA',
                'NIFTY INFRASTRUCTURE': '^CNXINFRA',
                # Thematic
                'NIFTY PSU': '^CNXPSE',
                'NIFTY MNC': '^CNXMNC'
            }
            index_ticker = index_map.get(regime_index, '^NSEI')
            
            try:
                # Download EXTRA historical data (300 days before start_date) 
                # so EMA 200 can be properly calculated from day 1 of backtest
                extended_start = pd.Timestamp(self.start_date) - timedelta(days=400)  # ~300 trading days
                regime_data = yf.download(index_ticker, start=extended_start, end=self.end_date, progress=False)
                if not regime_data.empty:
                    print(f"Downloaded {len(regime_data)} days of regime index data (with 400-day pre-buffer for EMA)")
                    
                    # Parse SuperTrend parameters if this is a SuperTrend regime filter
                    st_period = 7  # default
                    st_mult = 3    # default
                    if regime_config.get('type', '').startswith('SUPERTREND'):
                        st_value = regime_config.get('value', '7-3')
                        if isinstance(st_value, str) and '-' in st_value:
                            try:
                                st_period, st_mult = map(float, st_value.split('-'))
                                st_period = int(st_period)
                                print(f"Using SuperTrend Period={st_period}, Multiplier={st_mult}")
                            except:
                                pass
                    
                    # Parse SMA period if this is an SMA regime filter
                    sma_period = 50  # default
                    if regime_config.get('type', '').startswith('SMA'):
                        sma_period = int(regime_config.get('value', 50))
                        print(f"Using SMA Period={sma_period}")
                    
                    # Parse EMA period if this is an EMA regime filter
                    ema_period = 68  # default
                    if regime_config.get('type', '').startswith('EMA'):
                        ema_period = int(regime_config.get('value', 68))
                        print(f"Using EMA Period={ema_period}")
                    
                    regime_data = IndicatorLibrary.add_regime_filters(regime_data, 
                                                                       supertrend_period=st_period, 
                                                                       supertrend_multiplier=st_mult,
                                                                       sma_period=sma_period,
                                                                       ema_period=ema_period)
                    
                    # Add Donchian channels if needed
                    if regime_config.get('type') == 'DONCHIAN':
                        exit_period = regime_config.get('exit_period', 55)
                        recovery_period = regime_config.get('recovery_period', 20)
                        regime_data = IndicatorLibrary.add_donchian_channels(regime_data, exit_period, recovery_period)
                        print(f"Calculated Donchian channels ({exit_period}/{recovery_period})")
                    
                    # Add Swing + ATR if needed
                    if regime_config.get('type') == 'SWING_ATR':
                        swing_period = regime_config.get('swing_period', 20)
                        regime_data = IndicatorLibrary.add_swing_atr(regime_data, swing_period)
                        print(f"Calculated Swing+ATR (period={swing_period})")
                    
                    self.regime_index_data = regime_data
                    # Debug: show first few EMA values
                    if 'EMA_200' in regime_data.columns:
                        print(f"EMA_200 range: {regime_data['EMA_200'].min():.2f} - {regime_data['EMA_200'].max():.2f}")
            except Exception as e:
                print(f"Could not load regime index data: {e}")

        # ── Load Nifty Put Hedge data if action is 'Nifty Put Hedge' ────────────
        put_hedge_df = None
        is_put_hedge_regime = (
            regime_config is not None
            and regime_config.get('action') == 'Nifty Put Hedge'
        )
        if is_put_hedge_regime:
            _from = self.start_date
            _to   = self.end_date
            try:
                from nifty_put_hedge import load_or_build_hedge_data
                # New signature: (from_date, to_date, use_fallback=True)
                # No strike_offset or expiry_type — always ATM Weekly per design
                put_hedge_df = load_or_build_hedge_data(
                    _from, _to,
                    use_fallback=True,
                )
                if put_hedge_df is not None and not put_hedge_df.empty:
                    print(f"[PUT HEDGE] Loaded {len(put_hedge_df)} days of put data for backtest.")
                else:
                    print("[PUT HEDGE] No API data — VIX/B-S fallback will apply per rebalance date.")
                    put_hedge_df = pd.DataFrame()   # Empty DF: per-date fallback still fires
            except Exception as e:
                print(f"[PUT HEDGE] Failed to load hedge data: {e}. Per-date fallback will apply.")
                put_hedge_df = pd.DataFrame()       # Empty DF, not None — allows per-date fallback

        # Get common date range
        all_dates = sorted(list(set().union(*[df.index for df in self.data.values()])))
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(all_dates, rebal_config)
        
        # Initialize portfolio
        cash = self.initial_capital
        holdings = {}  # {ticker: shares}
        entry_prices = {}  # {ticker: cost_basis_per_share} for risk management
        portfolio_history = []
        regime_active = False
        regime_cash_reserve = 0
        last_known_prices = {}  # Track last known prices for holdings (for data gaps)
        risk_events = []  # Track risk management exits

        # Put Hedge state tracking
        put_hedge_position  = {}   # {'lots', 'premium_paid', 'entry_date', 'lot_size'}
        hedge_total_cost    = 0.0  # Cumulative premium paid
        hedge_total_proceeds = 0.0  # Cumulative proceeds from closing hedges
        hedge_event_count   = 0    # Number of hedge activations

        # EQUITY regime filter tracking
        peak_equity = self.initial_capital  # Track highest ACTUAL equity reached
        theoretical_peak = self.initial_capital  # Track highest THEORETICAL equity reached (for recovery check)
        equity_regime_active = False  # True when drawdown exceeds threshold, waiting for recovery
        theoretical_history = []  # Track what would happen without EQUITY filter
        theoretical_holdings = {}  # Separate holdings for theoretical curve
        theoretical_cash = self.initial_capital
        is_equity_regime = regime_config and regime_config['type'] == 'EQUITY'
        equity_sl_pct = regime_config['value'] if is_equity_regime else 0
        # Recovery threshold - defaults to same as trigger if not specified
        recovery_dd_pct = regime_config.get('recovery_dd', equity_sl_pct) if is_equity_regime else 0
        if recovery_dd_pct is None or recovery_dd_pct >= equity_sl_pct:
            recovery_dd_pct = equity_sl_pct  # Fallback to same as trigger
        self.regime_trigger_events = []
        
        # EQUITY_MA regime filter tracking
        is_equity_ma_regime = regime_config and regime_config['type'] == 'EQUITY_MA'
        equity_ma_period = regime_config.get('ma_period', 50) if is_equity_ma_regime else 50
        equity_history_for_ma = []  # List of recent equity values for MA calculation
        equity_ma_triggered = False  # True when equity < MA
        
        # Flag to track theoretical equity for ANY regime filter (for comparison)
        has_regime_filter = regime_config is not None
        
        # Track regime_action state across non-rebalance days for weekly options rolling
        regime_action = 'none'
        
        for date in all_dates:
            is_rebalance = date in rebalance_dates
            
            # Calculate current equity value FIRST (before any trades)
            current_holdings_value = 0.0
            for ticker, shares in holdings.items():
                if ticker in self.data:
                    if date in self.data[ticker].index:
                        cp = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        last_known_prices[ticker] = cp
                    elif ticker in last_known_prices:
                        cp = last_known_prices[ticker]
                    else:
                        continue
                    current_holdings_value += shares * cp
            current_equity = cash + current_holdings_value
            
            # --- WEEKLY PUT HEDGE ROLL LOGIC ---
            if put_hedge_position and put_hedge_df is not None:
                expiry_str = put_hedge_position.get('expiry_date')
                if expiry_str:
                    expiry_dt = pd.Timestamp(expiry_str)
                    current_dt = pd.Timestamp(date)
                    if current_dt >= expiry_dt:
                        # 1. Close expired option
                        proceeds = self._close_put_hedge(date, put_hedge_position, put_hedge_df)
                        cash += proceeds
                        hedge_total_proceeds += proceeds
                        put_hedge_position = {}
                        
                        # 2. If it's NOT a rebalance day, re-buy immediately if regime is still active AND rolling is enabled.
                        put_hedge_cfg = regime_config.get('put_hedge_config', {}) if regime_config else {}
                        roll_expiries = put_hedge_cfg.get('roll_expiries', True)
                        
                        if not is_rebalance and regime_action == 'Nifty Put Hedge' and roll_expiries:
                            cash, put_hedge_position, hedge_cost_now = self._apply_put_hedge(
                                date, cash, holdings, put_hedge_df, put_hedge_cfg
                            )
                            if hedge_cost_now > 0:
                                hedge_total_cost += hedge_cost_now
                                hedge_event_count += 1
            # -----------------------------------
            
            # Track theoretical equity (without ANY regime filter affecting trades)
            if has_regime_filter:
                theoretical_holdings_value = 0.0
                for ticker, shares in theoretical_holdings.items():
                    if ticker in self.data:
                        if date in self.data[ticker].index:
                            cp = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        elif ticker in last_known_prices:
                            cp = last_known_prices[ticker]
                        else:
                            continue
                        theoretical_holdings_value += shares * cp
                theoretical_equity = theoretical_cash + theoretical_holdings_value
                
                # Update theoretical peak (always track the highest theoretical value)
                if theoretical_equity > theoretical_peak:
                    theoretical_peak = theoretical_equity
                
                theoretical_history.append({
                    'Date': date,
                    'Theoretical_Equity': theoretical_equity,
                    'Theoretical_Holdings': theoretical_holdings_value
                })
            
            # Update peak equity (only when not in equity regime active mode)
            if not equity_regime_active and current_equity > peak_equity:
                peak_equity = current_equity
            
            # RISK MANAGEMENT: Check portfolio/trade level risk (on every day)
            if risk_config and holdings and not equity_regime_active and not regime_active:
                risk_triggered, tickers_to_exit, trigger_reason = self._check_risk_management(
                    date, holdings, entry_prices, risk_config
                )
                
                if risk_triggered and tickers_to_exit:
                    # Execute risk exits
                    for exit_ticker in tickers_to_exit:
                        if exit_ticker in holdings:
                            exit_shares = holdings[exit_ticker]
                            if exit_ticker in self.data and date in self.data[exit_ticker].index:
                                sell_price = self._get_scalar(self.data[exit_ticker].loc[date, 'Close'])
                                proceeds = exit_shares * sell_price
                                cash += proceeds
                                
                                self.trades.append({
                                    'Date': date,
                                    'Ticker': exit_ticker,
                                    'Action': 'SELL',
                                    'Shares': exit_shares,
                                    'Price': sell_price,
                                    'Value': proceeds,
                                    'Score': 0,
                                    'Rank': 0,
                                    'Reason': 'RISK_EXIT'
                                })
                                
                                del holdings[exit_ticker]
                                if exit_ticker in entry_prices:
                                    del entry_prices[exit_ticker]
                    
                    # Track risk event
                    risk_events.append({
                        'date': date,
                        'reason': trigger_reason,
                        'tickers_exited': tickers_to_exit.copy()
                    })
            
            # EQUITY REGIME: Check for mid-day drawdown breach (on any day, not just rebalance)
            if is_equity_regime and not equity_regime_active and holdings:
                drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100 if peak_equity > 0 else 0
                
                if drawdown_pct > equity_sl_pct:
                    # TRIGGER: Drawdown exceeded threshold - SELL ALL immediately
                    print(f"🔴 EQUITY REGIME TRIGGERED [{date.date()}]: Drawdown={drawdown_pct:.2f}% > SL={equity_sl_pct}% (Peak={peak_equity:.0f}, Current={current_equity:.0f})")
                    
                    equity_regime_active = True
                    self.regime_trigger_events.append({
                        'date': date,
                        'type': 'trigger',
                        'drawdown': drawdown_pct,
                        'peak': peak_equity,
                        'current': current_equity
                    })
                    
                    # Sell all holdings immediately (mid-day sell)
                    for ticker, shares in list(holdings.items()):
                        if ticker in self.data and date in self.data[ticker].index:
                            sell_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                            proceeds = shares * sell_price
                            cash += proceeds
                            
                            self.trades.append({
                                'Date': date,
                                'Ticker': ticker,
                                'Action': 'SELL',
                                'Shares': shares,
                                'Price': sell_price,
                                'Value': proceeds,
                                'Reason': 'EQUITY_REGIME_TRIGGER'
                            })
                    holdings = {}
                    regime_active = True
            
            # INTRADAY REGIME CHECK: For non-EQUITY types when exit_check is 'Intraday (Daily Check)'
            # This checks the regime filter daily (not just on rebalance) and exits immediately when triggered
            is_intraday_exit = regime_config and regime_config.get('exit_check', 'Intraday (Daily Check)') == 'Intraday (Daily Check)'
            is_non_equity_regime = regime_config and regime_config.get('type') not in ['EQUITY', 'EQUITY_MA']
            
            if is_intraday_exit and is_non_equity_regime and not regime_active and holdings:
                # Check regime filter daily (not just on rebalance days)
                intraday_triggered, intraday_action, _ = self._check_regime_filter(date, regime_config, current_equity, peak_equity)
                
                if intraday_triggered:
                    # TRIGGER: Regime filter triggered - SELL ALL immediately
                    regime_type = regime_config.get('type', 'UNKNOWN')
                    print(f"🔴 {regime_type} REGIME TRIGGERED (INTRADAY) [{date.date()}]: Selling all positions")
                    
                    regime_active = True
                    self.regime_trigger_events.append({
                        'date': date,
                        'type': 'trigger',
                        'drawdown': 0,
                        'peak': peak_equity,
                        'current': current_equity
                    })
                    
                    # Sell all holdings immediately
                    for ticker, shares in list(holdings.items()):
                        if ticker in self.data and date in self.data[ticker].index:
                            sell_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                            proceeds = shares * sell_price
                            cash += proceeds
                            
                            self.trades.append({
                                'Date': date,
                                'Ticker': ticker,
                                'Action': 'SELL',
                                'Shares': shares,
                                'Price': sell_price,
                                'Value': proceeds,
                                'Reason': f'{regime_type}_REGIME_TRIGGER'
                            })
                    holdings = {}
            
            if is_rebalance:
                # Sell all current holdings (regular rebalance sell)
                for ticker, shares in list(holdings.items()):
                    if ticker in self.data and date in self.data[ticker].index:
                        sell_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        proceeds = shares * sell_price
                        cash += proceeds
                        
                        self.trades.append({
                            'Date': date,
                            'Ticker': ticker,
                            'Action': 'SELL',
                            'Shares': shares,
                            'Price': sell_price,
                            'Value': proceeds
                        })
                
                holdings = {}
                
                # Recalculate current equity after sells
                current_equity = cash
                
                # Apply reinvest option
                if reinvest_profits:
                    # Use all available cash (capital + profits)
                    investable_capital = float(cash)
                else:
                    # Cap at initial capital only
                    investable_capital = min(float(cash), self.initial_capital)
                
                # EQUITY REGIME: Check for recovery on rebalance day
                if is_equity_regime and equity_regime_active:
                    # Calculate THEORETICAL drawdown using theoretical equity's OWN peak
                    # This gives proper 0-100% drawdown values
                    theoretical_drawdown = ((theoretical_peak - theoretical_equity) / theoretical_peak) * 100 if theoretical_peak > 0 else 0
                    
                    # Use stricter recovery threshold (recovery_dd_pct) to avoid whipsaw
                    # Recovery happens when theoretical portfolio recovers to within X% of its own peak
                    if theoretical_drawdown <= recovery_dd_pct:
                        # RECOVERY: Market (theoretical) has recovered sufficiently - resume normal trading
                        print(f"🟢 EQUITY REGIME RECOVERED [{date.date()}]: Theoretical Drawdown={theoretical_drawdown:.2f}% <= Recovery Threshold={recovery_dd_pct}%")
                        print(f"   (Trigger was at {equity_sl_pct}%, Recovery requires <={recovery_dd_pct}%)")
                        print(f"   Theoretical Peak={theoretical_peak:.0f}, Theoretical Current={theoretical_equity:.0f}")
                        print(f"   Actual Cash={cash:.0f}")
                        equity_regime_active = False
                        regime_active = False
                        self.regime_trigger_events.append({
                            'date': date,
                            'type': 'recovery',
                            'drawdown': theoretical_drawdown,
                            'peak': theoretical_peak,  # Use theoretical peak for logging
                            'current': theoretical_equity  # Use theoretical equity for logging
                        })
                        # NOTE: We do NOT reset peak_equity here
                        # Peak only updates when actual equity makes NEW highs (line 687-688)
                        # This ensures cummax drawdown never exceeds trigger threshold
                        print(f"   Peak stays at: ₹{peak_equity:.0f} (will only update on new highs)")
                    else:
                        print(f"⏳ EQUITY REGIME STILL ACTIVE [{date.date()}]: Theoretical Drawdown={theoretical_drawdown:.2f}% > Recovery Threshold={recovery_dd_pct}%")
                
                # Check regime filter for non-EQUITY types, or use equity_regime_active for EQUITY type
                # For Stock-level filtering, per-stock filtering happens during buy phase, not here
                is_stock_regime = regime_config and regime_config.get('index') == 'Stock'
                
                if is_stock_regime:
                    # Stock-level regime: don't trigger portfolio-wide regime
                    # Individual stocks are filtered during the buy phase
                    regime_triggered = False
                    regime_action = 'none'
                    current_drawdown = 0.0
                elif is_equity_regime:
                    regime_triggered = equity_regime_active
                    regime_action = regime_config['action'] if equity_regime_active else 'none'
                    current_drawdown = ((peak_equity - current_equity) / peak_equity) * 100 if peak_equity > 0 else 0
                elif is_equity_ma_regime:
                    # EQUITY_MA: Check if current equity is below its moving average
                    if len(equity_history_for_ma) >= equity_ma_period:
                        equity_ma = sum(equity_history_for_ma[-equity_ma_period:]) / equity_ma_period
                        equity_ma_triggered = current_equity < equity_ma
                        if equity_ma_triggered:
                            print(f"📉 EQUITY_MA TRIGGERED [{date.date()}]: Equity={current_equity:.0f} < MA{equity_ma_period}={equity_ma:.0f}")
                        else:
                            print(f"✅ EQUITY_MA OK [{date.date()}]: Equity={current_equity:.0f} >= MA{equity_ma_period}={equity_ma:.0f}")
                    else:
                        equity_ma_triggered = False  # Not enough data yet
                    regime_triggered = equity_ma_triggered
                    regime_action = regime_config['action'] if equity_ma_triggered else 'none'
                    current_drawdown = 0.0
                else:
                    regime_triggered, regime_action, current_drawdown = self._check_regime_filter(date, regime_config, current_equity, peak_equity)
                
                # Calculate allocations based on regime filter + uncorrelated interaction
                stocks_target = 0.0
                uncorrelated_target = 0.0

                
                if regime_triggered:
                    # For EQUITY regime: ALWAYS go 100% cash when triggered (drawdown protection)
                    # For other regimes: respect the action setting
                    if is_equity_regime:
                        # EQUITY regime: 100% cash until recovery - no stocks, no uncorrelated
                        stocks_target = 0.0
                        uncorrelated_target = 0.0  # Force 100% cash for max protection
                        regime_active = True
                        print(f"   EQUITY REGIME ACTIVE: 100% cash (no stocks, no uncorrelated) until recovery")
                    elif regime_action == 'Go Cash':
                        # 0% to stocks, uncorrelated gets its % from total, rest is cash
                        stocks_target = 0.0
                        if uncorrelated_config:
                            # Support both old 'allocation_pct' and new 'total_pct' format
                            total_alloc = uncorrelated_config.get('total_pct', uncorrelated_config.get('allocation_pct', 100))
                            allocation_pct = total_alloc / 100.0
                            uncorrelated_target = investable_capital * allocation_pct
                        regime_active = True

                    elif regime_action == 'Half Portfolio':
                        # ALWAYS 50% to stocks, uncorrelated from remaining 50%
                        stocks_target = investable_capital * 0.5
                        if uncorrelated_config:
                            # Support both old 'allocation_pct' and new 'total_pct' format
                            total_alloc = uncorrelated_config.get('total_pct', uncorrelated_config.get('allocation_pct', 100))
                            allocation_pct = total_alloc / 100.0
                            # Uncorrelated from the OTHER 50% (cash reserve)
                            uncorrelated_target = (investable_capital * 0.5) * allocation_pct
                        regime_active = True

                    elif regime_action == 'Nifty Put Hedge':
                        # ── NEW: Buy NIFTY ATM Puts as hedge ────────────────────
                        put_hedge_cfg  = regime_config.get('put_hedge_config', {})
                        keep_stocks    = put_hedge_cfg.get('keep_stocks', True)

                        if keep_stocks:
                            # Keep full equity exposure AND add put hedge
                            stocks_target     = investable_capital
                            uncorrelated_target = 0.0
                        else:
                            # Sell stocks, use capital to buy puts + uncorrelated
                            stocks_target     = 0.0
                            if uncorrelated_config:
                                total_alloc = uncorrelated_config.get('total_pct', uncorrelated_config.get('allocation_pct', 100))
                                allocation_pct = total_alloc / 100.0
                                uncorrelated_target = investable_capital * allocation_pct
                            else:
                                uncorrelated_target = 0.0

                        # Buy puts if we don't already have an open hedge
                        if not put_hedge_position:
                            cash, put_hedge_position, hedge_cost_now = self._apply_put_hedge(
                                date, cash, holdings, put_hedge_df, put_hedge_cfg
                            )
                            if hedge_cost_now > 0:
                                hedge_total_cost  += hedge_cost_now
                                hedge_event_count += 1
                                # Reduce investable capital by hedge cost if keep_stocks=False
                                if not keep_stocks:
                                    investable_capital = max(cash, 0)
                        elif put_hedge_position:
                            print(f"[PUT HEDGE] Hedge already active ({put_hedge_position.get('lots',0)} lots). Skipping re-buy.")
                        else:
                            print("[PUT HEDGE] No put data — falling back to Go Cash.")
                            stocks_target = 0.0

                        regime_active = True

                else:
                    # ── No regime triggered ──────────────────────────────────────
                    # If we had an active put hedge, close it now (recovery)
                    if put_hedge_position:
                        proceeds = self._close_put_hedge(date, put_hedge_position, put_hedge_df)
                        cash += proceeds
                        hedge_total_proceeds += proceeds
                        put_hedge_position = {}
                        # Recalculate investable capital after receiving proceeds
                        if reinvest_profits:
                            investable_capital = float(cash)
                        else:
                            investable_capital = min(float(cash), self.initial_capital)

                    regime_active     = False
                    uncorrelated_target = 0.0
                    stocks_target     = investable_capital
                
                # Debug: Log allocations on rebalance days
                if stocks_target == 0:
                    print(f"REBALANCE {date.date()}: REGIME={regime_triggered} ({regime_action}) - NO STOCKS, uncorrelated={uncorrelated_target:.0f}")
                else:
                    print(f"REBALANCE {date.date()}: REGIME={regime_triggered} - stocks={stocks_target:.0f}")
                
                # Execute uncorrelated asset purchases (supports multiple assets)
                if uncorrelated_target > 0 and uncorrelated_config:
                    # Support both old format {'asset': 'X', 'allocation_pct': N} and new {'assets': [...]}
                    assets_list = uncorrelated_config.get('assets', [])
                    
                    # Backward compatibility: handle old single-asset config
                    if not assets_list and 'asset' in uncorrelated_config:
                        assets_list = [{'ticker': uncorrelated_config['asset'], 'pct': 100}]
                    
                    for asset_config in assets_list:
                        asset_ticker = asset_config['ticker']
                        asset_pct = asset_config['pct'] / 100.0  # Convert to decimal
                        asset_target = uncorrelated_target * asset_pct
                        
                        # Download if needed
                        if asset_ticker not in self.data:
                            try:
                                ticker_ns = asset_ticker if asset_ticker.endswith(('.NS', '.BO')) else f"{asset_ticker}.NS"
                                unc_df = yf.download(ticker_ns, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                                if not unc_df.empty:
                                    unc_df.reset_index(inplace=True)
                                    unc_df['Date'] = pd.to_datetime(unc_df['Date'])
                                    unc_df.set_index('Date', inplace=True)
                                    # Reindex to all trading dates and forward-fill gaps
                                    unc_df = unc_df.reindex(all_dates).ffill().bfill()
                                    self.data[asset_ticker] = unc_df
                            except Exception as e:
                                print(f"Could not download {asset_ticker}: {e}")
                        
                        # Buy uncorrelated asset
                        if asset_ticker in self.data and date in self.data[asset_ticker].index:
                            unc_price = self._get_scalar(self.data[asset_ticker].loc[date, 'Close'])
                            unc_shares = int(asset_target / unc_price)
                            
                            if unc_shares > 0:
                                unc_cost = unc_shares * unc_price
                                cash -= unc_cost
                                # Add to existing holdings or create new
                                holdings[asset_ticker] = holdings.get(asset_ticker, 0) + unc_shares
                                
                                self.trades.append({
                                    'Date': date,
                                    'Ticker': asset_ticker,
                                    'Action': 'BUY',
                                    'Shares': unc_shares,
                                    'Price': unc_price,
                                    'Value': unc_cost,
                                    'Score': 0,
                                    'Rank': 'Uncorrelated'
                                })
                
                # stocks_target is now the amount available for stocks
                available_for_stocks = stocks_target
                
                # Calculate scores for all stocks - OPTIMIZED VECTORIZED VERSION
                # Exclude uncorrelated assets from stock scoring
                scores = {}
                
                # Build set of uncorrelated tickers to exclude
                uncorrelated_tickers = set()
                if uncorrelated_config:
                    assets_list = uncorrelated_config.get('assets', [])
                    if not assets_list and 'asset' in uncorrelated_config:
                        uncorrelated_tickers.add(uncorrelated_config['asset'])
                    else:
                        uncorrelated_tickers = {a['ticker'] for a in assets_list}
                
                # Get historical universe if enabled
                historical_universe = None
                if historical_universe_config and historical_universe_config.get('enabled'):
                    try:
                        from nifty_universe import get_universe
                        universe_name = historical_universe_config.get('universe_name', 'NIFTY 500')
                        historical_universe = set(get_universe(universe_name, as_of_date=date))
                        print(f"   [HISTORICAL] Using {len(historical_universe)} stocks from {universe_name} as of {date.date()}")
                    except Exception as e:
                        print(f"   [WARN] Historical universe lookup failed: {e}")
                
                # Collect all rows for this date (excluding uncorrelated assets)
                date_rows = {}
                for ticker, df in self.data.items():
                    # Skip uncorrelated assets - they are not stocks
                    if ticker in uncorrelated_tickers:
                        continue
                    # Skip if not in historical universe (survivorship bias fix)
                    if historical_universe is not None and ticker not in historical_universe:
                        continue
                    if date in df.index:
                        date_rows[ticker] = df.loc[date]
                
                # Score all stocks at once using vectorized calculation
                if date_rows:
                    # Create a DataFrame from all rows
                    all_rows_df = pd.DataFrame(date_rows).T
                    
                    # Calculate scores using vectorized method
                    try:
                        scores_series = self.parser.calculate_scores(all_rows_df, scoring_formula)
                        scores = scores_series.to_dict()
                        
                        # Filter out invalid scores
                        scores = {k: v for k, v in scores.items() if v > -999999}
                    except:
                        # Fallback to row-by-row if vectorized fails
                        for ticker, row in date_rows.items():
                            score = self.parser.parse_and_calculate(scoring_formula, row)
                            if score > -999999:
                                scores[ticker] = score
                
                # Rank stocks
                ranked_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                # Debug: Log scoring results
                if len(scores) == 0:
                    print(f"   [WARN] No stocks scored on {date.date()} - check indicator columns")
                elif len(ranked_stocks) < num_stocks:
                    print(f"   [WARN] Only {len(ranked_stocks)} stocks scored (need {num_stocks}) on {date.date()}")
                
                # Select top N stocks
                top_stocks = ranked_stocks[:num_stocks]
                
                # Stock-level regime filter: exclude stocks that fail their individual regime filter
                is_stock_regime = regime_config and regime_config.get('index') == 'Stock'
                if is_stock_regime and top_stocks:
                    filtered_top_stocks = []
                    for ticker, score in top_stocks:
                        if not self._check_stock_regime_filter(ticker, date, regime_config):
                            filtered_top_stocks.append((ticker, score))
                        else:
                            print(f"   [STOCK REGIME] Excluding {ticker} - price below {regime_config['type']} {regime_config['value']}")
                    
                    if len(filtered_top_stocks) < len(top_stocks):
                        print(f"   [STOCK REGIME] {len(top_stocks) - len(filtered_top_stocks)} stocks filtered out, {len(filtered_top_stocks)} remaining")
                    top_stocks = filtered_top_stocks
                
                # Buy stocks with available_for_stocks amount
                if top_stocks and available_for_stocks > 0:
                    # Get position sizing config (default to equal weight)
                    sizing_method = 'equal_weight'
                    use_cap = False
                    max_pct = 15
                    if position_sizing_config:
                        sizing_method = position_sizing_config.get('method', 'equal_weight')
                        use_cap = position_sizing_config.get('use_cap', False)
                        max_pct = position_sizing_config.get('max_pct', 15)
                    
                    # Calculate weights based on method
                    weights = {}
                    
                    if sizing_method == 'equal_weight':
                        # Simple equal allocation
                        for ticker, score in top_stocks:
                            weights[ticker] = 1.0 / len(top_stocks)
                    
                    elif sizing_method == 'inverse_volatility':
                        # Allocate more to low-volatility stocks
                        volatilities = {}
                        for ticker, score in top_stocks:
                            if ticker in self.data and len(self.data[ticker]) >= 20:
                                returns = self.data[ticker]['Close'].pct_change().dropna()
                                # Use last 60 days if available, else all data
                                recent_returns = returns.iloc[-60:] if len(returns) > 60 else returns
                                vol = recent_returns.std() * (252 ** 0.5)  # Annualized
                                volatilities[ticker] = vol if vol > 0 else 0.01  # Avoid div by zero
                            else:
                                volatilities[ticker] = 0.3  # Default 30% annual vol
                        
                        # Inverse volatility weights
                        inv_vols = {t: 1/v for t, v in volatilities.items()}
                        total_inv_vol = sum(inv_vols.values())
                        for ticker, score in top_stocks:
                            weights[ticker] = inv_vols[ticker] / total_inv_vol
                    
                    elif sizing_method == 'score_weighted':
                        # Allocate more to higher-scoring stocks
                        total_score = sum(abs(score) for _, score in top_stocks)
                        if total_score > 0:
                            for ticker, score in top_stocks:
                                weights[ticker] = abs(score) / total_score
                        else:
                            # Fallback to equal weight
                            for ticker, score in top_stocks:
                                weights[ticker] = 1.0 / len(top_stocks)
                    
                    elif sizing_method == 'inverse_downside_vol':
                        # Allocate more to stocks with lower downside volatility (semi-deviation)
                        # Only considers negative returns
                        downside_vols = {}
                        for ticker, score in top_stocks:
                            if ticker in self.data and len(self.data[ticker]) >= 20:
                                returns = self.data[ticker]['Close'].pct_change().dropna()
                                recent_returns = returns.iloc[-60:] if len(returns) > 60 else returns
                                # Only negative returns
                                negative_returns = recent_returns[recent_returns < 0]
                                if len(negative_returns) > 5:
                                    downside_vol = negative_returns.std() * (252 ** 0.5)
                                    downside_vols[ticker] = downside_vol if downside_vol > 0 else 0.01
                                else:
                                    downside_vols[ticker] = 0.2  # Default 20%
                            else:
                                downside_vols[ticker] = 0.2
                        
                        # Inverse downside volatility weights
                        inv_dvols = {t: 1/v for t, v in downside_vols.items()}
                        total_inv_dvol = sum(inv_dvols.values())
                        for ticker, score in top_stocks:
                            weights[ticker] = inv_dvols[ticker] / total_inv_dvol
                    
                    elif sizing_method == 'inverse_max_drawdown':
                        # Allocate more to stocks with lower max drawdown
                        max_drawdowns = {}
                        for ticker, score in top_stocks:
                            if ticker in self.data and len(self.data[ticker]) >= 20:
                                close = self.data[ticker]['Close']
                                recent_close = close.iloc[-60:] if len(close) > 60 else close
                                # Calculate max drawdown
                                rolling_max = recent_close.cummax()
                                drawdown = (recent_close - rolling_max) / rolling_max
                                max_dd = abs(drawdown.min())
                                max_drawdowns[ticker] = max_dd if max_dd > 0.01 else 0.01
                            else:
                                max_drawdowns[ticker] = 0.15  # Default 15%
                        
                        # Inverse max drawdown weights
                        inv_mdd = {t: 1/v for t, v in max_drawdowns.items()}
                        total_inv_mdd = sum(inv_mdd.values())
                        for ticker, score in top_stocks:
                            weights[ticker] = inv_mdd[ticker] / total_inv_mdd
                    
                    elif sizing_method == 'risk_parity':
                        # Equal risk contribution (volatility-adjusted)
                        volatilities = {}
                        for ticker, score in top_stocks:
                            if ticker in self.data and len(self.data[ticker]) >= 20:
                                returns = self.data[ticker]['Close'].pct_change().dropna()
                                recent_returns = returns.iloc[-60:] if len(returns) > 60 else returns
                                vol = recent_returns.std() * (252 ** 0.5)
                                volatilities[ticker] = vol if vol > 0 else 0.01
                            else:
                                volatilities[ticker] = 0.3
                        
                        # Risk parity: weight inversely proportional to vol squared
                        inv_vol_sq = {t: 1/(v**2) for t, v in volatilities.items()}
                        total_inv_vol_sq = sum(inv_vol_sq.values())
                        for ticker, score in top_stocks:
                            weights[ticker] = inv_vol_sq[ticker] / total_inv_vol_sq
                    
                    else:
                        # Default: equal weight
                        for ticker, score in top_stocks:
                            weights[ticker] = 1.0 / len(top_stocks)
                    
                    # Apply max position cap if enabled
                    if use_cap:
                        max_weight = max_pct / 100.0
                        # Cap weights and redistribute excess
                        excess = 0.0
                        uncapped_count = 0
                        for ticker in weights:
                            if weights[ticker] > max_weight:
                                excess += weights[ticker] - max_weight
                                weights[ticker] = max_weight
                            else:
                                uncapped_count += 1
                        
                        # Redistribute excess proportionally to uncapped positions
                        if uncapped_count > 0 and excess > 0:
                            redistribute = excess / uncapped_count
                            for ticker in weights:
                                if weights[ticker] < max_weight:
                                    weights[ticker] += redistribute
                    
                    # Buy based on calculated weights
                    for ticker, score in top_stocks:
                        position_value = available_for_stocks * weights[ticker]
                        buy_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        shares = int(position_value / buy_price)
                        
                        if shares > 0:
                            cost = shares * buy_price
                            cash -= cost
                            holdings[ticker] = shares
                            entry_prices[ticker] = buy_price  # Track for risk management
                            
                            self.trades.append({
                                'Date': date,
                                'Ticker': ticker,
                                'Action': 'BUY',
                                'Shares': shares,
                                'Price': buy_price,
                                'Value': cost,
                                'Score': score,
                                'Rank': ranked_stocks.index((ticker, score)) + 1
                            })
                # Update theoretical holdings (for ANY regime filter comparison)
                # Theoretical curve assumes NO regime filter - always trades normally
                if has_regime_filter and is_rebalance:
                    # First, SELL all theoretical holdings (same as actual rebalance does)
                    for ticker, shares in theoretical_holdings.items():
                        if ticker in self.data and date in self.data[ticker].index:
                            sell_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                            theoretical_cash += shares * sell_price
                    
                    # Reset theoretical holdings
                    theoretical_holdings = {}
                    
                    # Buy top stocks with all theoretical cash (no regime filter reducing allocation)
                    if top_stocks and theoretical_cash > 0:
                        theo_position_value = theoretical_cash / max(1, len(top_stocks))
                        for ticker, score in top_stocks:
                            buy_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                            theo_shares = int(theo_position_value / buy_price)
                            if theo_shares > 0:
                                theoretical_holdings[ticker] = theo_shares
                                theoretical_cash -= theo_shares * buy_price

            
            
            # Calculate portfolio value - use last known price if current data missing
            holdings_value = 0.0
            for ticker, shares in holdings.items():
                if ticker in self.data:
                    if date in self.data[ticker].index:
                        close_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        last_known_prices[ticker] = close_price  # Track last price
                    elif ticker in last_known_prices:
                        close_price = last_known_prices[ticker]  # Use last known
                    else:
                        continue  # No price available at all
                    holdings_value += shares * close_price
            
            total_value = cash + holdings_value
            
            # Update equity history for EQUITY_MA calculation
            if is_equity_ma_regime:
                equity_history_for_ma.append(total_value)
                # Keep only the last 2x MA period to save memory
                if len(equity_history_for_ma) > equity_ma_period * 2:
                    equity_history_for_ma = equity_history_for_ma[-equity_ma_period * 2:]
            
            # Calculate current drawdown for tracking
            current_dd = ((peak_equity - total_value) / peak_equity) * 100 if peak_equity > 0 else 0
            
            # Calculate equity MA for tracking
            equity_ma_value = 0.0
            if is_equity_ma_regime and len(equity_history_for_ma) >= equity_ma_period:
                equity_ma_value = sum(equity_history_for_ma[-equity_ma_period:]) / equity_ma_period
            
            portfolio_history.append({
                'Date': date,
                'Cash': cash,
                'Holdings': holdings_value,
                'Portfolio Value': total_value,
                'Positions': len(holdings),
                'Regime_Active': regime_active,
                'Put_Hedge_Active': bool(put_hedge_position),
                'Peak_Equity': peak_equity,
                'Drawdown_Pct': current_dd,
                'Equity_Regime_Active': equity_regime_active if is_equity_regime else False,
                'Equity_MA': equity_ma_value if is_equity_ma_regime else 0.0,
                'Equity_MA_Triggered': equity_ma_triggered if is_equity_ma_regime else False
            })
        
        # Store results
        self.portfolio_df = pd.DataFrame(portfolio_history).set_index('Date')
        self.trades_df = pd.DataFrame(self.trades)
        
        # Store final holdings for get_open_positions()
        self.final_holdings = holdings.copy()  # {ticker: shares}
        self.final_last_known_prices = last_known_prices.copy()  # {ticker: price}
        self.final_cash = cash
        
        # Store put hedge summary for metrics
        self.put_hedge_summary = {
            'total_cost':     hedge_total_cost,
            'total_proceeds': hedge_total_proceeds,
            'event_count':    hedge_event_count,
            'net_pnl':        hedge_total_proceeds - hedge_total_cost,
            'efficiency_pct': (
                ((hedge_total_proceeds - hedge_total_cost) / hedge_total_cost * 100)
                if hedge_total_cost > 0 else 0.0
            ),
        } if is_put_hedge_regime else None

        # Store regime analysis data for ANY regime filter (for comparison)
        if has_regime_filter and theoretical_history:
            self.equity_regime_analysis = {
                'theoretical_curve': pd.DataFrame(theoretical_history).set_index('Date'),
                'trigger_events': self.regime_trigger_events if is_equity_regime else [],
                'sl_threshold': equity_sl_pct if is_equity_regime else 0,
                'is_equity_ma': is_equity_ma_regime,
                'ma_period': equity_ma_period if is_equity_ma_regime else None,
                'regime_type': regime_config['type'] if regime_config else None,
                'regime_value': regime_config['value'] if regime_config else None
            }
    
    def get_equity_regime_analysis(self):
        """Return regime filter analysis data for visualization.
        
        Returns:
            dict with 'theoretical_curve', 'trigger_events', 'regime_type', etc.
            or None if no regime filter was used
        """
        return self.equity_regime_analysis
    
    def get_open_positions(self):
        """Get current open positions with P&L information.
        
        Returns:
            List of dicts with Stock, Shares, Buy Price, Current Price, 
            Unrealized P&L, and Unrealized ROI %
        """
        if not hasattr(self, 'final_holdings') or not self.final_holdings:
            return []
        
        positions = []
        
        # Get buy prices from trades
        buy_prices = {}
        if hasattr(self, 'trades') and self.trades:
            for trade in self.trades:
                if trade.get('Action') == 'BUY':
                    ticker = trade.get('Stock')
                    # Keep the most recent buy price for each ticker
                    buy_prices[ticker] = trade.get('Price', 0)
        
        for ticker, shares in self.final_holdings.items():
            if shares <= 0:
                continue
            
            # Get current price
            current_price = self.final_last_known_prices.get(ticker, 0)
            
            # Get buy price (from trades or fall back to current price)
            buy_price = buy_prices.get(ticker, current_price)
            
            # Calculate P&L
            unrealized_pnl = (current_price - buy_price) * shares
            unrealized_roi = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
            
            positions.append({
                'Stock': ticker,
                'Shares': shares,
                'Buy Price': round(buy_price, 2),
                'Current Price': round(current_price, 2),
                'Unrealized P&L': round(unrealized_pnl, 2),
                'Unrealized ROI %': round(unrealized_roi, 2)
            })
        
        # Sort by unrealized ROI
        positions.sort(key=lambda x: x['Unrealized ROI %'], reverse=True)
        
        return positions
    
    def get_metrics(self):
        """Calculate comprehensive performance metrics."""
        if self.portfolio_df.empty:
            return None

        final_value = self.portfolio_df['Portfolio Value'].iloc[-1]
        total_return = final_value - self.initial_capital
        return_pct = (total_return / self.initial_capital) * 100

        # CAGR
        days = (self.portfolio_df.index[-1] - self.portfolio_df.index[0]).days
        years = days / 365.25
        cagr = ((final_value / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Max Drawdown
        running_max = self.portfolio_df['Portfolio Value'].cummax()
        drawdown = (self.portfolio_df['Portfolio Value'] - running_max) / running_max * 100
        max_dd = abs(drawdown.min())

        # Volatility
        returns = self.portfolio_df['Portfolio Value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        # Sharpe Ratio
        rf_rate = 0.05
        sharpe = (cagr / 100 - rf_rate) / (volatility / 100) if volatility > 0 else 0

        # Win Rate and Trade Statistics
        wins = 0
        losses = 0
        win_amounts = []
        loss_amounts = []
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        last_was_win = None
        total_trades = 0
        trade_returns = []
        trade_maes = []
        
        if not self.trades_df.empty and 'Action' in self.trades_df.columns:
            # Get BUY and SELL trades
            buy_trades = self.trades_df[self.trades_df['Action'] == 'BUY'].copy()
            sell_trades = self.trades_df[self.trades_df['Action'] == 'SELL'].copy()
            
            # For each SELL, find a matching BUY to calculate PnL
            # Group by Date to get rebalance-level PnL
            if not sell_trades.empty:
                for date in sell_trades['Date'].unique():
                    sells_on_date = sell_trades[sell_trades['Date'] == date]
                    total_sell = sells_on_date['Value'].sum()
                    
                    # Find corresponding previous BUY values (from holdings bought earlier)
                    # For simplicity, calculate rebalance-level PnL (sell_value - buy_value for same tickers)
                    for _, sell_row in sells_on_date.iterrows():
                        ticker = sell_row['Ticker']
                        sell_value = sell_row['Value']
                        
                        # Find previous BUY for this ticker (most recent before this sell)
                        prev_buys = buy_trades[(buy_trades['Ticker'] == ticker) & (buy_trades['Date'] < date)]
                        if not prev_buys.empty:
                            buy_row = prev_buys.iloc[-1]
                            buy_value = buy_row['Value']
                            pnl = sell_value - buy_value
                            
                            total_trades += 1
                            
                            if pnl > 0:
                                wins += 1
                                win_amounts.append(pnl)
                                if last_was_win == True:
                                    current_streak += 1
                                else:
                                    current_streak = 1
                                max_consecutive_wins = max(max_consecutive_wins, current_streak)
                                last_was_win = True
                            elif pnl < 0:
                                losses += 1
                                loss_amounts.append(abs(pnl))
                                if last_was_win == False:
                                    current_streak += 1
                                else:
                                    current_streak = 1
                                max_consecutive_losses = max(max_consecutive_losses, current_streak)
                                last_was_win = False
                                
                            # --- New Risk Metrics (MAE & Returns) ---
                            trade_ret = pnl / buy_value if buy_value > 0 else 0
                            trade_returns.append(trade_ret)
                            
                            # Calculate MAE (Maximum Adverse Excursion)
                            # Worst unrealized loss during the trade
                            try:
                                if ticker in self.data:
                                    # Ensure we have data for the period
                                    ticker_data = self.data[ticker]
                                    buy_date = buy_row['Date']
                                    sell_date = date # sell_row['Date']
                                    
                                    # Slice data from entry to exit (inclusive)
                                    mask = (ticker_data.index >= buy_date) & (ticker_data.index <= sell_date)
                                    period_data = ticker_data[mask]
                                    
                                    if not period_data.empty and 'Low' in period_data.columns:
                                        # Use Low for Long trades (assuming Long only system for now)
                                        # If we support shorting, we'd use High
                                        min_price = float(period_data['Low'].min())
                                        buy_price = float(buy_row['Price'])
                                        
                                        if buy_price > 0:
                                            mae = (min_price - buy_price) / buy_price
                                            trade_maes.append(float(mae))
                            except Exception as e:
                                # Fail silently for MAE calc to not break backtest
                                pass
            
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        else:
            win_rate = 0
        
        # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
        avg_win = np.mean(win_amounts) if win_amounts else 0
        avg_loss = np.mean(loss_amounts) if loss_amounts else 0
        win_pct = wins / total_trades if total_trades > 0 else 0
        loss_pct = losses / total_trades if total_trades > 0 else 0
        expectancy = (win_pct * avg_win) - (loss_pct * avg_loss)
        
        # Drawdown Recovery Analysis
        running_max = self.portfolio_df['Portfolio Value'].cummax()
        is_in_drawdown = self.portfolio_df['Portfolio Value'] < running_max
        
        # Find drawdown periods and recovery
        recovery_days = 0
        recovery_trades = 0
        max_recovery_days = 0
        max_recovery_trades = 0
        
        if is_in_drawdown.any():
            # Find where drawdown starts and ends
            drawdown_start = None
            for i, (date, in_dd) in enumerate(is_in_drawdown.items()):
                if in_dd and drawdown_start is None:
                    drawdown_start = date
                elif not in_dd and drawdown_start is not None:
                    # Recovered from drawdown
                    days_in_dd = (date - drawdown_start).days
                    max_recovery_days = max(max_recovery_days, days_in_dd)
                    
                    # Count trades during this period
                    if not self.trades_df.empty and 'Date' in self.trades_df.columns:
                        trades_in_period = self.trades_df[
                            (self.trades_df['Date'] >= drawdown_start) & 
                            (self.trades_df['Date'] <= date)
                        ]
                        max_recovery_trades = max(max_recovery_trades, len(trades_in_period) // 2)
                    
                    drawdown_start = None
        
        # Zerodha Equity Delivery Charges Calculation
        # STT/CTT: 0.1% on buy & sell
        # Transaction charges: NSE 0.00297%
        # GST: 18% on (SEBI + transaction charges)
        # SEBI: ₹10/crore = 0.0001%
        # Stamp: 0.015% on buy side only
        
        total_turnover = 0
        total_buy_value = 0
        total_sell_value = 0
        
        if not self.trades_df.empty and 'Action' in self.trades_df.columns:
            buy_trades = self.trades_df[self.trades_df['Action'] == 'BUY']
            sell_trades = self.trades_df[self.trades_df['Action'] == 'SELL']
            total_buy_value = buy_trades['Value'].sum() if not buy_trades.empty else 0
            total_sell_value = sell_trades['Value'].sum() if not sell_trades.empty else 0
            total_turnover = total_buy_value + total_sell_value
        
        # Calculate charges
        stt_ctt = total_turnover * 0.001  # 0.1% on both sides
        transaction_charges = total_turnover * 0.0000297  # NSE 0.00297%
        sebi_charges = total_turnover * 0.000001  # ₹10/crore = 0.0001%
        stamp_charges = total_buy_value * 0.00015  # 0.015% on buy side
        gst = (transaction_charges + sebi_charges) * 0.18  # 18% GST
        
        total_charges = stt_ctt + transaction_charges + sebi_charges + stamp_charges + gst
        
        # Calculate Risk Metrics Stats
        mae_median = np.median(trade_maes) * 100 if trade_maes else 0
        mae_95 = np.percentile(trade_maes, 5) * 100 if trade_maes else 0  # 5th percentile because MAE is negative
        mae_max = np.min(trade_maes) * 100 if trade_maes else 0
        
        # Calculate CVaR (5%)
        cvar_5 = 0
        if trade_returns:
            returns_array = np.array(trade_returns)
            cutoff = np.percentile(returns_array, 5)
            tail_losses = returns_array[returns_array <= cutoff]
            cvar_5 = tail_losses.mean() * 100 if len(tail_losses) > 0 else 0

        # ── Put Hedge metrics ───────────────────────────────────────────────
        hedge_summary = getattr(self, 'put_hedge_summary', None) or {}

        return {
            'Final Value': final_value,
            'Total Return': total_return,
            'Return %': return_pct,
            'CAGR %': cagr,
            'Max Drawdown %': max_dd,
            'Volatility %': volatility,
            'Sharpe Ratio': sharpe,
            'Win Rate %': win_rate,
            'Total Trades': total_trades,
            # New metrics
            'Max Consecutive Wins': max_consecutive_wins,
            'Max Consecutive Losses': max_consecutive_losses,
            'Days to Recover from DD': max_recovery_days,
            'Trades to Recover from DD': max_recovery_trades,
            'Expectancy': expectancy,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            # Zerodha charges
            'Total Turnover': total_turnover,
            'STT/CTT': stt_ctt,
            'Transaction Charges': transaction_charges,
            'SEBI Charges': sebi_charges,
            'Stamp Charges': stamp_charges,
            'GST': gst,
            'Total Charges': total_charges,
            # Risk Metrics
            'MAE Median %': mae_median,
            'MAE 95% %': mae_95,
            'MAE Max %': mae_max,
            'CVaR 5% %': cvar_5,
            # Put Hedge metrics (0 when hedge not used)
            'Hedge Cost Total': hedge_summary.get('total_cost', 0.0),
            'Hedge Proceeds Total': hedge_summary.get('total_proceeds', 0.0),
            'Hedge Net PnL': hedge_summary.get('net_pnl', 0.0),
            'Hedge Efficiency %': hedge_summary.get('efficiency_pct', 0.0),
            'Hedge Events': hedge_summary.get('event_count', 0),
        }

    def get_monthly_returns(self):
        """Calculate monthly returns table similar to the format shown."""
        if self.portfolio_df.empty:
            return pd.DataFrame()

        # Get monthly portfolio values
        df = self.portfolio_df.copy()
        df['Year'] = df.index.year
        df['Month'] = df.index.month

        # Get last value of each month
        monthly_values = df.groupby(['Year', 'Month'])['Portfolio Value'].last()

        # Calculate monthly returns
        monthly_returns = monthly_values.pct_change() * 100

        # Pivot to year x month format
        monthly_df = monthly_returns.reset_index()
        monthly_df.columns = ['Year', 'Month', 'Return']

        # Create pivot table
        pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')

        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_names[int(m)-1] for m in pivot.columns]

        # Calculate yearly total (compound returns)
        yearly_totals = []
        for year in pivot.index:
            year_data = df[df['Year'] == year]['Portfolio Value']
            if len(year_data) > 0:
                year_return = ((year_data.iloc[-1] / year_data.iloc[0]) - 1) * 100
                yearly_totals.append(year_return)
            else:
                yearly_totals.append(None)

        pivot['Total'] = yearly_totals

        # Reorder columns to have all 12 months + Total
        all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in all_months:
            if month not in pivot.columns:
                pivot[month] = None

        # Reorder columns
        pivot = pivot[all_months + ['Total']]

        # Format as percentages with proper display
        pivot = pivot.round(3)

        return pivot
