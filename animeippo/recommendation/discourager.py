DISCOURAGE_AMOUNT = 0.25


def apply_discourage_on_repeating_items(items):
    return items["discourage_score"] - DISCOURAGE_AMOUNT
