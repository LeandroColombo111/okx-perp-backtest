"""Minimal HTTP access to OKX's PUBLIC market-data endpoints. No credentials are read, sent or needed."""
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OKX_BASE = "https://www.okx.com"


def get_json(path: str, params: dict, timeout: float) -> dict:
    request = Request(OKX_BASE + path + "?" + urlencode(params), headers={"User-Agent": "okx-perp-backtest"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())
