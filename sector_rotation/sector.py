from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# Accept both canonical macro names and raw ticker columns from macro.csv
MACRO_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "consumer": ("consumer", "Consumer", "xrt", "XRT"),
    "labor": ("labor", "Labor", "employment", "VTI", "vti"),
    "inflation": ("inflation", "Inflation", "infl", "TIP", "tip"),
    "credit": ("credit", "Credit", "stress", "HYG", "hyg"),
}


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
    zscore_window: int = 36
    threshold_window: int = 36
    smoothing_window: int = 3
    macro_lags: Dict[str, int] = field(default_factory=dict)


# Macro Regime Detector Engine  Key words[Classifier, MacroSignals, GrowthInflation]
class MacroRegimeDetector:

    def __init__(self, config: MacroIndicatorConfig):
        self.config = config

    def _normalize_macro_inputs(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        if macro_df is None or macro_df.empty:
            return macro_df

        rename_map: Dict[str, str] = {}
        lower_lookup = {str(col).lower(): col for col in macro_df.columns}

        for canonical, aliases in MACRO_COLUMN_ALIASES.items():
            for alias in aliases:
                key = alias.lower()
                if key in lower_lookup:
                    rename_map[lower_lookup[key]] = canonical
                    break

        normalized = macro_df.rename(columns=rename_map)
        missing = [col for col in MACRO_COLUMN_ALIASES if col not in normalized.columns]
        if missing:
            missing_cols = ", ".join(missing)
            raise KeyError(f"Missing macro columns: {missing_cols}")

        return normalized

    # Normalizer  Key words[zscore, scaling, preprocessing]
    def _normalize(self, s: pd.Series) -> pd.Series:
        window = max(2, int(getattr(self.config, "zscore_window", 0)))
        if window <= 2:
            expanded_mean = s.expanding().mean()
            expanded_std = s.expanding().std().replace(0, np.nan)
            return ((s - expanded_mean) / (expanded_std + 1e-8)).fillna(0.0)

        min_periods = max(5, window // 2)
        rolling = s.rolling(window, min_periods=min_periods)
        rolling_mean = rolling.mean()
        rolling_std = rolling.std().replace(0, np.nan)

        expanding = s.expanding(min_periods=min_periods)
        expanding_mean = expanding.mean()
        expanding_std = expanding.std().replace(0, np.nan)

        mean = rolling_mean.combine_first(expanding_mean)
        std = rolling_std.combine_first(expanding_std)

        z = (s - mean) / (std + 1e-8)
        return z.fillna(0.0)

    def _compute_quantiles(
        self,
        series: pd.Series,
        quantiles: List[float],
        window: int
    ) -> Dict[float, pd.Series]:
        min_periods = max(5, window // 2)
        rolling = series.rolling(window, min_periods=min_periods)
        expanding = series.expanding(min_periods=min_periods)

        out: Dict[float, pd.Series] = {}
        for q in quantiles:
            roll_q = rolling.quantile(q)
            exp_q = expanding.quantile(q)
            out[q] = roll_q.combine_first(exp_q).ffill()
        return out

    def _smooth_regimes(self, regimes: pd.Series, window: int) -> pd.Series:
        if window <= 1 or regimes.empty:
            return regimes

        codes = regimes.map(lambda r: r.value)

        def _mode_from_codes(arr: np.ndarray) -> float:
            clean = arr[~np.isnan(arr)]
            if clean.size == 0:
                return np.nan
            ints = clean.astype(int)
            counts = np.bincount(ints)
            return float(counts.argmax())

        smoothed_codes = codes.rolling(window, min_periods=1).apply(
            _mode_from_codes,
            raw=True
        )
        smoothed_codes = smoothed_codes.ffill().bfill()
        mapping = {reg.value: reg for reg in EconomicRegime}
        smoothed = smoothed_codes.round().astype(int).map(mapping)
        return smoothed.rename("regime")

    # Composite Score Calculator  Key words[Growth, Inflation, Stress]
    def compute_composite_scores(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        macro_df = macro_df.copy()
        macro_df = self._normalize_macro_inputs(macro_df)
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
        macro_df = macro_df.copy()
        macro_df = self._normalize_macro_inputs(macro_df)
        macro_df = macro_df.dropna(how="any")
        if macro_df.empty:
            return pd.Series(dtype=object, name="regime")

        scores = self.compute_composite_scores(macro_df)
        regimes = []

        window = max(6, int(getattr(self.config, "threshold_window", 36)))
        growth_q = self._compute_quantiles(scores['growth'], [0.7, 0.55, 0.3], window)
        stress_q = self._compute_quantiles(scores['financial_stress'], [0.7, 0.5], window)
        inflation_q = self._compute_quantiles(scores['inflation_pressure'], [0.65], window)

        def _get(q_dict, key, dt, default=0.0):
            val = q_dict[key].get(dt, np.nan)
            return default if pd.isna(val) else float(val)

        for dt, row in scores.iterrows():
            growth = row['growth']
            inflation = row['inflation_pressure']
            stress = row['financial_stress']

            growth_high = _get(growth_q, 0.7, dt)
            growth_mid = _get(growth_q, 0.55, dt)
            growth_low = _get(growth_q, 0.3, dt)
            stress_high = _get(stress_q, 0.7, dt)
            stress_mid = _get(stress_q, 0.5, dt)
            inflation_high = _get(inflation_q, 0.65, dt)

            if growth > growth_high and stress < stress_mid:
                regime = EconomicRegime.EARLY_EXPANSION
            elif growth > growth_mid and stress <= stress_mid:
                regime = EconomicRegime.MID_EXPANSION
            elif growth > growth_low and (inflation > inflation_high or stress > stress_mid):
                regime = EconomicRegime.LATE_EXPANSION
            elif growth < growth_low and stress > stress_high:
                regime = EconomicRegime.RECESSION
            else:
                regime = EconomicRegime.RECOVERY

            regimes.append(regime)

        return pd.Series(regimes, index=scores.index, name="regime")

    def detect_regime_series_smoothed(
        self,
        macro_df: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.Series:
        raw = self.detect_regime_series(macro_df)
        smooth_window = window or getattr(self.config, "smoothing_window", 1)
        return self._smooth_regimes(raw, smooth_window)


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

    @staticmethod
    def _calculate_turnover(
        previous_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> float:
        tickers = set(previous_weights.keys()) | set(new_weights.keys())
        return sum(abs(new_weights.get(t, 0.0) - previous_weights.get(t, 0.0)) for t in tickers)

    # Backtest Runner  Key words[DailyReturns, Execution, PortfolioPath]
    def run(self, initial_capital: float = 1_000_000) -> pd.DataFrame:
        dates = self.prices.index
        rebal_dates = self._get_rebalance_dates()
        rebal_dates_set = set(rebal_dates)

        equity = pd.Series(index=dates, dtype=float)
        equity.iloc[0] = initial_capital

        current_weights: Dict[str, float] = {}
        last_date = dates[0]

        for i, date in enumerate(dates[1:], start=1):
            turnover = 0.0

            if date in rebal_dates_set:
                regime = self.regime_series.asof(date)
                sectors = self.mapper.get_sectors_for_regime(regime)

                available_sectors = [s for s in sectors if s in self.prices.columns]
                if available_sectors:
                    sector_prices = self.prices[available_sectors].loc[:date].dropna()
                else:
                    sector_prices = pd.DataFrame()

                if not sector_prices.empty:
                    benchmark = sector_prices.mean(axis=1)
                    allowed = self.filters.filter_assets(
                        sector_prices,
                        benchmark=benchmark,
                        top_n=len(available_sectors)
                    )
                    new_weights = self.pc.build_portfolio(allowed)
                else:
                    new_weights = {}

                turnover = self._calculate_turnover(current_weights, new_weights)
                current_weights = new_weights

            if current_weights:
                cols = [t for t in current_weights.keys() if t in self.prices.columns]
                if cols and len(cols) != len(current_weights):
                    total = sum(current_weights[t] for t in cols)
                    if total > 0:
                        current_weights = {t: current_weights[t] / total for t in cols}
                    else:
                        current_weights = {}
                        cols = []
                if cols:
                    day_prices = self.prices.loc[[last_date, date], cols]
                    day_rets = day_prices.pct_change().iloc[-1]
                    portfolio_ret = sum(current_weights[t] * day_rets.get(t, 0.0) for t in cols)
                else:
                    portfolio_ret = 0.0
            else:
                portfolio_ret = 0.0

            if turnover > 0:
                cost = turnover * self.cfg.transaction_cost_bps / 10_000.0
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
    regime_series = detector.detect_regime_series_smoothed(macro_df)

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
