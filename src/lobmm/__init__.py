"""lob-market-maker: a limit order book market-making simulator.

Modules, in the order the simulation flows through them:

- ``book``        the matching engine (price-time priority, integer ticks)
- ``events``      the event queue the simulation runs on
- ``config``      every parameter, as dataclasses with documented defaults
- ``flow``        synthetic order flow: noise limit/market orders, cancellations, a latent fair value, informed traders
- ``agents``      the market makers: fixed spread, Avellaneda-Stoikov, and a probe quoter used for calibration
- ``accounting``  cash, inventory, mark-to-mid equity, markouts and PnL attribution
- ``sim``         wires it together and runs one session
- ``calibrate``   sigma from mid prices; A and k of the fill-intensity curve from probe sessions
- ``analytics``   summaries over seeds, parameter sweeps, ablations
- ``plots``       the figures
- ``cli``         ``lobmm run|compare|sweep|calibrate|ablate|bench|figures``
"""

from lobmm.book import BUY, SELL, Fill, Order, OrderBook, Submission
from lobmm.config import FlowConfig, MakerConfig, MarketConfig, SimConfig

__all__ = [
    "BUY",
    "SELL",
    "Fill",
    "FlowConfig",
    "MakerConfig",
    "MarketConfig",
    "Order",
    "OrderBook",
    "SimConfig",
    "Submission",
]
__version__ = "0.1.0"
