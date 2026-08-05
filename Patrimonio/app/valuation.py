from typing import Dict, Any

def valuation_market_minus_debt(row: Dict[str, Any]) -> float:
    # expect keys: market_value, mortgage or debt
    mv = None
    for k in ('market_value', 'valor_mercado', 'valor de mercado', 'Market Value', 'Valor Mercado'):
        if k in row and row[k] is not None:
            mv = float(row[k])
            break
    debt = None
    for k in ('debt', 'mortgage', 'hipoteca', 'Deuda'):
        if k in row and row[k] is not None:
            debt = float(row[k])
            break
    mv = mv or 0.0
    debt = debt or 0.0
    return mv - debt

def valuation_company_net_assets(row: Dict[str, Any]) -> float:
    # expect cash, property_value, liabilities
    cash = 0.0
    prop = 0.0
    liabilities = 0.0
    for k in ('cash', 'efectivo', 'Caja'):
        if k in row and row[k] is not None:
            cash = float(row[k]); break
    for k in ('property_value', 'property', 'prop_value', 'valor_inmueble', 'Valor Inmueble'):
        if k in row and row[k] is not None:
            prop = float(row[k]); break
    for k in ('liabilities', 'pasivos', 'deudas', 'Liabilities'):
        if k in row and row[k] is not None:
            liabilities = float(row[k]); break
    return cash + prop - liabilities

def valuation_market_direct(row: Dict[str, Any]) -> float:
    for k in ('market_value', 'valor_mercado', 'Market Value'):
        if k in row and row[k] is not None:
            return float(row[k])
    return 0.0

VAL_RULES = {
    'market_minus_debt': valuation_market_minus_debt,
    'company_net_assets': valuation_company_net_assets,
    'market_direct': valuation_market_direct,
}

def compute_value(row: Dict[str, Any], rule: str):
    fn = VAL_RULES.get(rule)
    if not fn:
        return 0.0
    try:
        return fn(row)
    except Exception:
        return 0.0
