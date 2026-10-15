"""Prices and discount codes."""

DISCOUNTS = {"SAVE10": 0.10, "HALF": 0.50}


def round_price(amount):
    return round(amount, 2)


def apply_discount(amount, code=None):
    if code is None:
        return round_price(amount)
    rate = DISCOUNTS[code]
    return round_price(amount * rate)
