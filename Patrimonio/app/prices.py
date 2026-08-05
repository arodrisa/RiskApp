import json
from datetime import datetime
from urllib.parse import quote
from urllib.request import urlopen


class PriceLookupError(Exception):
    pass


def normalize_provider(provider: str = None):
    return (provider or 'manual').strip().lower()


def get_quote(provider: str, symbol: str):
    provider = normalize_provider(provider)
    symbol = (symbol or '').strip()
    if provider == 'manual':
        raise PriceLookupError('Manual assets do not have a price provider')
    if not symbol:
        raise PriceLookupError('Price symbol is required')
    if provider != 'yahoo':
        raise PriceLookupError(f'Unsupported price provider: {provider}')

    encoded_symbol = quote(symbol)
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?range=5d&interval=1d'
    try:
        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except Exception as exc:
        raise PriceLookupError(f'Could not fetch price for {symbol}') from exc

    result = (payload.get('chart') or {}).get('result') or []
    if not result:
        raise PriceLookupError(f'No quote returned for {symbol}')

    quote_data = (result[0].get('indicators') or {}).get('quote') or []
    closes = quote_data[0].get('close') if quote_data else []
    price = next((float(value) for value in reversed(closes or []) if value is not None), None)
    if price is None:
        raise PriceLookupError(f'No close price returned for {symbol}')

    currency = (result[0].get('meta') or {}).get('currency')
    timestamp = next((value for value in reversed(result[0].get('timestamp') or []) if value), None)
    as_of = datetime.utcfromtimestamp(timestamp).isoformat() if timestamp else None
    return {
        'provider': provider,
        'symbol': symbol,
        'price': price,
        'currency': currency,
        'as_of': as_of,
    }
