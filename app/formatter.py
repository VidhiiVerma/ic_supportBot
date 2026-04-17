def format_currency(value):
    try:
        return f"${float(value):,.0f}"
    except:
        return value


def format_number(value):
    try:
        return f"{float(value):,.0f}"
    except:
        return value


def format_percentage(value):
    try:
        return f"{float(value) * 100:.0f}%"
    except:
        return value