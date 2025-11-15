import backtrader as bt
import pandas as pd
from sector import (
    MacroIndicatorConfig, MacroRegimeDetector,
    SectorMapper, AssetFilters, PortfolioConfig,
    PortfolioConstructor
)

class RegimeStrategy(bt.Strategy):

    params = dict(
        rebalance_days=21,
        lookback=126,
        capital=10000
    )

    def __init__(self):
        self.day_count = 0
        self.macro_df = pd.read_csv("macro.csv", index_col=0, parse_dates=True).sort_index()
        self.detector = MacroRegimeDetector(MacroIndicatorConfig())
        self.regimes = self.detector.detect_regime_series_smoothed(self.macro_df)

        self.mapper = SectorMapper()
        self.filters = AssetFilters()
        self.port_cfg = PortfolioConfig()
        self.pc = PortfolioConstructor(self.port_cfg)
        self.data_lookup = {d._name: d for d in self.datas}

    def next(self):
        self.day_count += 1
        if self.day_count % self.p.rebalance_days != 0:
            return

        today = self.data.datetime.date(0)
        regime = self.regimes.asof(today)
        if regime is None:
            return
        sectors = self.mapper.get_sectors_for_regime(regime)
        if not sectors:
            return

        price_history = {
            d._name: d.close.get(size=self.p.lookback)
            for d in self.datas
        }
        df = pd.DataFrame(price_history).dropna(axis=1)
        available = [s for s in sectors if s in df.columns]
        if not available:
            return
        df = df[available]

        benchmark = df.mean(axis=1)
        allowed = self.filters.filter_assets(df, benchmark=benchmark, top_n=len(available))
        weights = self.pc.build_portfolio(allowed)
        if not weights:
            return

        for d in self.datas:
            if self.getposition(d).size != 0:
                self.close(d)

        equity = self.broker.getvalue()
        for ticker, w in weights.items():
            data_feed = self.data_lookup.get(ticker)
            if data_feed is None:
                continue
            price = data_feed.close[0]
            if price <= 0 or pd.isna(price):
                continue
            amount = equity * w
            size = amount / price
            self.buy(data=data_feed, size=size)
