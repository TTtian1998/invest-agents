"""Generate deterministic, explicitly fictional bars; weekdays are NOT exchange calendars."""
from pathlib import Path
import math

import pandas as pd

out = Path(__file__).parent / "synthetic_bars.csv"
rows = []
for i, day in enumerate(pd.bdate_range("2024-01-02", periods=180)):
    for j, symbol in enumerate(("DEMO_A", "DEMO_B", "DEMO_C")):
        price = 10 + j * 3 + (0.02 - j * 0.015) * i + math.sin(i / 13 + j)
        rows.append({"date": day.date().isoformat(), "symbol": symbol,
                     "open": round(price * 0.999, 6), "close": round(price, 6),
                     "tradable": 1, "limit_up": 0, "limit_down": 0})
pd.DataFrame(rows).to_csv(out, index=False)
print(out)
