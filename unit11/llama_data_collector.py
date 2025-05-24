from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from data_collector import StockDataCollector
from logger import api_logger
from config import DATA_COLLECTION_CONFIG

class DataCollector(StockDataCollector):
    def __init__(self):
        super().__init__()
        self.cache_dir = Path(DATA_COLLECTION_CONFIG["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = DATA_COLLECTION_CONFIG["max_retries"]
        self.timeout = DATA_COLLECTION_CONFIG["timeout"]

    def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get stock data from CAFE with caching support
        """
        cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.csv"
        
        if use_cache and cache_file.exists():
            api_logger.info(f"Loading cached data for {symbol}")
            return pd.read_csv(cache_file)

        try:
            # Use the parent class method for fetching data
            df = self.fetch_stock_data(symbol, start_date, end_date)
            
            # Save to cache
            if use_cache:
                df.to_csv(cache_file, index=False)
            
            return df
            
        except Exception as e:
            api_logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def get_multiple_stocks(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_stock_data(
                    symbol,
                    start_date,
                    end_date,
                    use_cache
                )
            except Exception as e:
                api_logger.error(f"Error fetching data for {symbol}: {str(e)}")
                results[symbol] = None
        return results

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information using vnquant
        """
        try:
            # Use the parent class method to fetch current data
            info_df = self.fetch_stock_data(
                symbol, 
                datetime.now().strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            return info_df.to_dict(orient='records')[0] if info_df is not None and len(info_df) > 0 else {}
            
        except Exception as e:
            api_logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            raise

    def get_market_index(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get market index data using vnquant
        """
        try:
            # Use the parent class method for fetching index data
            return self.fetch_stock_data(index_code, start_date, end_date)
            
        except Exception as e:
            api_logger.error(f"Error fetching index data for {index_code}: {str(e)}")
            raise

    def clear_cache(self, days: Optional[int] = None):
        """
        Clear cache files
        If days is specified, only clear files older than that
        """
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            for cache_file in self.cache_dir.glob("*.csv"):
                if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_date:
                    cache_file.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.csv"):
                cache_file.unlink() 


if __name__ == "__main__":
    collector = DataCollector()
    print("\n\n\nget_stock_data")
    print(collector.get_stock_data("HAG", "2024-01-01", "2024-01-01"))
    print("\n\n\nget_company_info")
    print(collector.get_company_info("HAG"))
    print("\n\n\nget_market_index")
    print(collector.get_market_index("VNINDEX", "2024-01-01", "2024-01-01"))
    print("\n\n\nclear_cache")
    print(collector.clear_cache(days=1))