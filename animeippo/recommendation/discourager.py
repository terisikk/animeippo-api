DISCOURAGE_AMOUNT = 0.25


def discourage_row(row):
    return row["discourage_score"] - DISCOURAGE_AMOUNT


def apply_discourage_on_repeating_items(items):
    return items.apply(discourage_row, axis=1)
