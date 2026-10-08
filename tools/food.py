"""
Restaurant search and order-total tools used by DoorDashAgent.

All lookups run against data/sample_restaurants.json, a committed set of
fictional restaurants, menus and fees. It is sample data for exercising the
agent offline; nothing here talks to DoorDash or places an order.
"""

import json
from functools import lru_cache
from pathlib import Path

from tools.timeutil import as_int, to_hhmm, to_minutes

RESTAURANTS_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_restaurants.json"
DATA_NOTICE = "Results come from bundled SAMPLE restaurant data (fictional menus, prices and fees)."

SMALL_ORDER_FEE = 2.00   # sample rule: applied when subtotal is below the restaurant's min_order
DEFAULT_TAX_RATE = 0.0625


@lru_cache(maxsize=4)
def load_restaurants(path: str | None = None) -> dict:
    with open(path or RESTAURANTS_PATH) as f:
        return json.load(f)


def _get_restaurant(restaurant_id: str) -> dict:
    for r in load_restaurants()["restaurants"]:
        if r["id"] == restaurant_id:
            return r
    raise ValueError(f"No restaurant with id {restaurant_id!r} in the sample data")


def search_restaurants(
    cuisine: str | None = None,
    dietary: list[str] | None = None,
    avoid_allergens: list[str] | None = None,
    max_delivery_minutes: int | None = None,
) -> dict:
    """
    Filter sample restaurants and return the dishes that satisfy every
    dietary tag (e.g. "vegan", "gluten-free") and contain none of the
    allergens to avoid. Restaurants with no matching dish are excluded.
    """
    dietary = [d.lower() for d in (dietary or [])]
    avoid = [a.lower() for a in (avoid_allergens or [])]
    results = []
    for r in load_restaurants()["restaurants"]:
        if cuisine and r["cuisine"].lower() != cuisine.lower():
            continue
        if max_delivery_minutes is not None and r["delivery_minutes"] > as_int(max_delivery_minutes, "max_delivery_minutes"):
            continue
        dishes = [
            {"id": d["id"], "name": d["name"], "price": d["price"], "tags": d["tags"]}
            for d in r["menu"]
            if all(tag in d["tags"] for tag in dietary) and not any(a in d["allergens"] for a in avoid)
        ]
        if not dishes:
            continue
        results.append(
            {
                "id": r["id"],
                "name": r["name"],
                "cuisine": r["cuisine"],
                "rating": r["rating"],
                "delivery_minutes": r["delivery_minutes"],
                "delivery_fee": r["delivery_fee"],
                "min_order": r["min_order"],
                "matching_dishes": dishes,
            }
        )
    results.sort(key=lambda r: (-r["rating"], r["delivery_minutes"]))
    return {"notice": DATA_NOTICE, "count": len(results), "restaurants": results}


def estimate_order_total(
    restaurant_id: str,
    items: list[dict],
    tip_percent: float = 15,
    tax_rate: float = DEFAULT_TAX_RATE,
) -> dict:
    """
    Itemised estimate for an order. items: [{"item_id": str, "quantity": int}].

    total = subtotal + delivery fee + service fee (% of subtotal)
            + small-order fee (if below min_order) + tax (on subtotal) + tip (% of subtotal)
    """
    restaurant = _get_restaurant(restaurant_id)
    menu = {d["id"]: d for d in restaurant["menu"]}
    lines = []
    for item in items:
        dish = menu.get(item["item_id"])
        if dish is None:
            raise ValueError(f"{item['item_id']!r} is not on {restaurant['name']}'s menu")
        qty = as_int(item.get("quantity", 1), "quantity")
        if qty <= 0:
            raise ValueError("quantity must be positive")
        lines.append({"name": dish["name"], "quantity": qty, "unit_price": dish["price"], "line_total": round(dish["price"] * qty, 2)})
    if not lines:
        raise ValueError("items must not be empty")

    subtotal = round(sum(l["line_total"] for l in lines), 2)
    service_fee = round(subtotal * restaurant["service_fee_pct"] / 100, 2)
    small_order_fee = SMALL_ORDER_FEE if subtotal < restaurant["min_order"] else 0.0
    tax = round(subtotal * float(tax_rate), 2)
    tip = round(subtotal * float(tip_percent) / 100, 2)
    total = round(subtotal + restaurant["delivery_fee"] + service_fee + small_order_fee + tax + tip, 2)
    return {
        "notice": DATA_NOTICE,
        "restaurant": restaurant["name"],
        "lines": lines,
        "subtotal": subtotal,
        "delivery_fee": restaurant["delivery_fee"],
        "service_fee": service_fee,
        "small_order_fee": small_order_fee,
        "tax": tax,
        "tip": tip,
        "total": total,
        "amount_to_reach_min_order": round(max(0.0, restaurant["min_order"] - subtotal), 2),
    }


def latest_order_time(meal_time: str, restaurant_id: str, buffer_minutes: int = 10) -> dict:
    """When to place an order so it arrives by meal_time (delivery estimate + buffer)."""
    restaurant = _get_restaurant(restaurant_id)
    lead = restaurant["delivery_minutes"] + as_int(buffer_minutes, "buffer_minutes")
    return {
        "restaurant": restaurant["name"],
        "meal_time": to_hhmm(to_minutes(meal_time)),
        "order_by": to_hhmm(to_minutes(meal_time) - lead),
        "lead_minutes": lead,
    }
