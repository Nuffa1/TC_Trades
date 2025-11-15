import backtrader as bt
from backtrader.feeds import AlpacaData
from backtrader.brokers import AlpacaBroker
from strategy_live import RegimeStrategy

ALPACA_API_KEY = "PK66EMPLFW536VYOWWUQGHEPOK"
ALPACA_SECRET = "DERR6rvMZAkAnLQC7KrLPwbWRMuwD3VKPaUmEYDWRi6k"
ALPACA_PAPER = True

tickers = ["XLY","XLF","XLI","XLK","XLB","XLE","XLP","XLU","XLV"]

cerebro = bt.Cerebro()
cerebro.addstrategy(RegimeStrategy)

for t in tickers:
    data = AlpacaData(
        dataname=t,
        timeframe=bt.TimeFrame.Days,
        historical=False,
        apikey=ALPACA_API_KEY,
        secretkey=ALPACA_SECRET,
        paper=ALPACA_PAPER,
    )
    cerebro.adddata(data, name=t)

broker = AlpacaBroker(
    apikey=ALPACA_API_KEY,
    secretkey=ALPACA_SECRET,
    paper=ALPACA_PAPER
)
cerebro.setbroker(broker)

cerebro.broker.setcash(10000.0)

print("Starting Portfolio Value:", cerebro.broker.getvalue())
cerebro.run()
print("Final Portfolio Value:", cerebro.broker.getvalue())
