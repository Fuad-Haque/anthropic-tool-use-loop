def calculate_monthly_total(usage_units: int, rate_per_unit: float) -> float:
    """Return the billed amount for a given usage period."""
    return round(usage_units * rate_per_unit, 2)