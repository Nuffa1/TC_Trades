from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


# Economic Regime Enum  Key words[MacroCycle, RegimeStates, Enum]
class EconomicRegime(Enum):
    EARLY_EXPANSION = auto()
    MID_EXPANSION = auto()
    LATE_EXPANSION = auto()
    RECESSION = auto()
    RECOVERY = auto()


# Macro Config Dataclass  Key words[Weights, Indicators, Composite]
@dataclass
class MacroIndicatorConfig:
    consumer_weight: float = 0.25
    labor_weight: float = 0.25
    inflation_weight: float = 0.25
    credit_weight: float = 0.25


# Macro Regime Detector Engine  Key words[Classifier, MacroSignals, GrowthInflation]
class MacroRegimeDetector:

    def __init__(self, config: MacroIndicatorConfig):
        self.config = config

    # Normalizer  Key words[zscore, scaling, preprocessing]
    def _normalize(self, s: pd.Series) -> pd.Series:
        return (s - s.mean()) / (s.std() + 1e-8)

    # Composite Score Calculator  Key words[Growth, Inflation, Stress]
    def compute_composite_scores(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=macro_df.index)

        consumer = self._normalize(macro_df['consumer'])
        labor = self._normalize(macro_df['labor'])
        inflation = self._normalize(macro_df['inflation'])
        credit = self._normalize(macro_df['credit'])

        out['growth'] = (
            self.config.consumer_weight * consumer +
            self.config.labor_weight * labor
        )
        out['inflation_pressure'] = (
            self.config.inflation_weight * inflation
        )
        out['financial_stress'] = (
            self.config.credit_weight * credit
        )

        return out

    # Regime Label Generator  Key words[RuleBased, Scoring, Classification]
    def detect_regime_series(self, macro_df: pd.DataFrame) -> pd.Series:
        scores = self.compute_composite_scores(macro_df)
        regimes = []

        for dt, row in scores.iterrows():
            growth = row['growth']
            inflation = row['inflation_pressure']
            stress = row['financial_stress']

            if growth > 0.7 and stress < 0:
                regime = EconomicRegime.EARLY_EXPANSION
            elif growth > 0.3 and stress <= 0.3:
                regime = EconomicRegime.MID_EXPANSION
            elif growth > 0 and (inflation > 0.5 or stress > 0.3):
                regime = EconomicRegime.LATE_EXPANSION
            elif growth < -0.3 and stress > 0.5:
                regime = EconomicRegime.RECESSION
            else:
                regime = EconomicRegime.RECOVERY

            regimes.append(regime)

        return pd.Series(regimes, index=scores.index, name="regime")


# Sector Mapping  Key words[SectorRotation, RegimeToSector, ETFMap]
class SectorMapper:

    def __init__(self):
        self.mapping: Dict[EconomicRegime, List[str]] = {
            EconomicRegime.EARLY_EXPANSION: ['XLY', 'XLF', 'XLI', 'XLK'],
            EconomicRegime.MID_EXPANSION: ['XLK', 'XLI', 'XLB'],
            EconomicRegime.LATE_EXPANSION: ['XLE', 'XLF', 'XLP', 'XLU'],
            EconomicRegime.RECESSION: ['XLP', 'XLU', 'XLV'],
            EconomicRegime.RECOVERY: ['XLY', 'XLI', 'XLF'],
        }

    # Sector Selector  Key words[RegimeLookup, ETFSelection]
    def get_sectors_for_regime(self, regime: EconomicRegime) -> List[str]:
        return self.mapping.get(regime, [])


# Asset-Level Filters  Key words[Momentum, Volatility, RelativeStrength]
class AssetFilters:

    def __init__(
        self,
        momentum_lookback: int = 126,
        vol_lookback: int = 63,
        max_vol_percentile: float = 0.8
    ):
        self.mom_lb = momentum_lookback
        self.vol_lb = vol_lookback
        self.max_vol_pct = max_vol_percentile

    # Momentum Calculator  Key words[Momentum, Returns, Lookback]
    def compute_momentum(self, prices: pd.DataFrame) -> pd.Series:
        return prices.pct_change(self.mom_lb).iloc[-1]

    # Volatility Calculator  Key words[StdDev, RiskFilter, Volatility]
    def compute_volatility(self, prices: pd.DataFrame) -> pd.Series:
        rets = prices.pct_change().dropna()
        return rets.tail(self.vol_lb).std()

    # Relative Strength Engine  Key words[Benchmark, FactorContrast]
    def compute_relative_strength(
        self,
        prices: pd.DataFrame,
        benchmark: pd.Series
    ) -> pd.Series:
        asset_ret = prices.pct_change(self.mom_lb).iloc[-1]
        bench_ret = benchmark.pct_change(self.mom_lb).iloc[-1]
        return asset_ret - bench_ret

    # Filter Pipeline  Key words[Ranking, Screens, TopN]
    def filter_assets(
        self,
        prices: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        top_n: Optional[int] = None
    ) -> List[str]:
        momentum = self.compute_momentum(prices)
        vol = self.compute_volatility(prices)

        if benchmark is not None:
            rs = self.compute_relative_strength(prices, benchmark)
        else:
            rs = momentum.copy()

        vol_thresh = vol.quantile(self.max_vol_pct)
        allowed = vol[vol <= vol_thresh].index

        score = 0.6 * momentum + 0.4 * rs
        score = score.loc[allowed].sort_values(ascending=False)

        if top_n is not None:
            score = score.head(top_n)

        return list(score.index)


# Portfolio Config  Key words[Weights, Constraints, Rebalance]
@dataclass
class PortfolioConfig:
    max_weight_per_sector: float = 0.3
    rebalance_frequency: str = 'M'
    transaction_cost_bps: float = 5


# Portfolio Constructor  Key words[Allocator, Weighting, Normalization]
class PortfolioConstructor:

    def __init__(self, config: PortfolioConfig):
        self.config = config

    # Weight Builder  Key words[EqualWeight, Caps, Normalization]
    def build_portfolio(
        self,
        allowed_tickers: List[str]
    ) -> Dict[str, float]:
        if not allowed_tickers:
            return {}

        n = len(allowed_tickers)
        base_weight = 1.0 / n
        max_w = self.config.max_weight_per_sector

        weights = {t: min(base_weight, max_w) for t in allowed_tickers}
        total = sum(weights.values())

        return {t: w / total for t, w in weights.items()}


# Backtest Engine  Key words[Simulation, Rebalance, EquityCurve]
class Backtester:

    def __init__(
        self,
        prices: pd.DataFrame,
        regime_series: pd.Series,
        sector_mapper: SectorMapper,
        asset_filters: AssetFilters,
        portfolio_constructor: PortfolioConstructor,
        portfolio_config: PortfolioConfig
    ):
        self.prices = prices
        self.regime_series = regime_series
        self.mapper = sector_mapper
        self.filters = asset_filters
        self.pc = portfolio_constructor
        self.cfg = portfolio_config

    # Rebalance Schedule Generator  Key words[Calendar, Monthly, Dates]
    def _get_rebalance_dates(self) -> List[pd.Timestamp]:
        return list(self.prices.resample(self.cfg.rebalance_frequency).last().index)

    # Backtest Runner  Key words[DailyReturns, Execution, PortfolioPath]
    def run(self, initial_capital: float = 1_000_000) -> pd.DataFrame:
        dates = self.prices.index
        rebal_dates = self._get_rebalance_dates()

        equity = pd.Series(index=dates, dtype=float)
        equity.iloc[0] = initial_capital

        current_weights: Dict[str, float] = {}
        last_date = dates[0]

        for i, date in enumerate(dates[1:], start=1):

            if date in rebal_dates:
                regime = self.regime_series.asof(date)
                sectors = self.mapper.get_sectors_for_regime(regime)

                sector_prices = self.prices[sectors].loc[:date].dropna()

                if not sector_prices.empty:
                    allowed = self.filters.filter_assets(sector_prices, top_n=len(sectors))
                    current_weights = self.pc.build_portfolio(allowed)
                else:
                    current_weights = {}

            if current_weights:
                day_rets = self.prices.loc[[last_date, date], list(current_weights.keys())].pct_change().iloc[-1]
                portfolio_ret = sum(current_weights[t] * day_rets[t] for t in current_weights)
            else:
                portfolio_ret = 0.0

            if date in rebal_dates and i != 0 and current_weights:
                cost = self.cfg.transaction_cost_bps / 10_000.0
                portfolio_ret -= cost

            equity.iloc[i] = equity.iloc[i - 1] * (1 + portfolio_ret)
            last_date = date

        return equity.to_frame(name="equity")


# Pipeline Wrapper  Key words[EndToEnd, Automation, Workflow]
def run_pipeline(
    macro_df: pd.DataFrame,
    price_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:

    macro_config = MacroIndicatorConfig()
    detector = MacroRegimeDetector(macro_config)
    regime_series = detector.detect_regime_series(macro_df)

    sector_mapper = SectorMapper()
    asset_filters = AssetFilters()

    portfolio_cfg = PortfolioConfig()
    portfolio_constructor = PortfolioConstructor(portfolio_cfg)

    backtester = Backtester(
        prices=price_df,
        regime_series=regime_series,
        sector_mapper=sector_mapper,
        asset_filters=asset_filters,
        portfolio_constructor=portfolio_constructor,
        portfolio_config=portfolio_cfg,
    )

    equity_curve = backtester.run(initial_capital=1_000_000)

    return equity_curve, regime_series
