from toyshop.pricing import apply_discount


class Cart:
    def __init__(self):
        self.items = []

    def add(self, name, price, qty=1):
        self.items.append((name, price, qty))

    def subtotal(self):
        return sum(price * qty for _, price, qty in self.items)

    def total(self, code=None):
        return apply_discount(self.subtotal(), code)
