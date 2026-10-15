from toyshop.cart import Cart


def format_line(name, price, qty):
    return f"{qty} x {name:<12} {price * qty:>8.2f}"


def render(cart: Cart, code=None):
    lines = [format_line(*item) for item in cart.items]
    lines.append(f"{'TOTAL':<16} {cart.total(code):>8.2f}")
    return "\n".join(lines)
