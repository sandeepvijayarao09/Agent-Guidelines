"""
Product search, price comparison and budget tools used by AmazonAgent and
ShoppingAgent.

All lookups run against data/sample_products.json, a small committed catalog
of fictional products and made-up prices. It is sample data for exercising
the agents offline, not a live retailer feed.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from tools.timeutil import as_int

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_products.json"
DATA_NOTICE = "Results come from a bundled SAMPLE catalog (fictional products, made-up prices)."


@lru_cache(maxsize=4)
def load_catalog(path: str | None = None) -> dict:
    with open(path or CATALOG_PATH) as f:
        return json.load(f)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _listing_total(listing: dict) -> float:
    return round(listing["price"] + listing.get("shipping", 0), 2)


def _summary(product: dict, listings: list[dict]) -> dict:
    best = min(listings, key=_listing_total)
    return {
        "id": product["id"],
        "name": product["name"],
        "category": product["category"],
        "rating": product["rating"],
        "review_count": product["review_count"],
        "best_price": _listing_total(best),
        "best_retailer": best["retailer"],
        "retailers": sorted(l["retailer"] for l in listings),
    }


def search_products(
    query: str = "",
    category: str | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    retailer: str | None = None,
    limit: int = 5,
) -> dict:
    """
    Keyword search over the sample catalog.

    Every query word must appear in the product name, category or tags.
    Results are ranked by rating (then review count), and max_price applies
    to the cheapest listing (price + shipping) at the allowed retailers.
    """
    words = _tokens(query)
    limit = max(1, as_int(limit, "limit"))
    results = []
    for product in load_catalog()["products"]:
        if category and product["category"].lower() != category.lower():
            continue
        if min_rating is not None and product["rating"] < float(min_rating):
            continue
        haystack = set(_tokens(" ".join([product["name"], product["category"], *product["tags"]])))
        if any(w not in haystack for w in words):
            continue
        listings = product["listings"]
        if retailer:
            listings = [l for l in listings if l["retailer"].lower() == retailer.lower()]
            if not listings:
                continue
        summary = _summary(product, listings)
        if max_price is not None and summary["best_price"] > float(max_price):
            continue
        results.append(summary)

    results.sort(key=lambda p: (-p["rating"], -p["review_count"], p["best_price"]))
    return {"notice": DATA_NOTICE, "count": len(results[:limit]), "products": results[:limit]}


def compare_prices(product_id: str) -> dict:
    """All retailer listings for one product, cheapest first, with the spread."""
    for product in load_catalog()["products"]:
        if product["id"] == product_id:
            offers = sorted(
                (
                    {
                        "retailer": l["retailer"],
                        "price": l["price"],
                        "shipping": l.get("shipping", 0),
                        "total": _listing_total(l),
                        "prime": l.get("prime", False),
                    }
                    for l in product["listings"]
                ),
                key=lambda o: (o["total"], o["retailer"]),
            )
            return {
                "notice": DATA_NOTICE,
                "product": product["name"],
                "offers": offers,
                "cheapest": offers[0]["retailer"],
                "savings_vs_highest": round(offers[-1]["total"] - offers[0]["total"], 2),
            }
    raise ValueError(f"No product with id {product_id!r} in the sample catalog")


def allocate_budget(items: list[dict], budget: float) -> dict:
    """
    Fit a shopping list into a budget.

    items: [{"name": str, "price": float, "priority": 1|2|3}]  (1 = must-have)
    Items are taken in priority order (cheaper first within a priority) while
    they fit; the rest are returned as "dropped".
    """
    budget = round(float(budget), 2)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    ranked = sorted(
        enumerate(items),
        key=lambda pair: (as_int(pair[1].get("priority", 2), "priority"), float(pair[1]["price"]), pair[0]),
    )
    kept, dropped, spent = [], [], 0.0
    for _, item in ranked:
        price = round(float(item["price"]), 2)
        if spent + price <= budget + 1e-9:
            kept.append({"name": item["name"], "price": price})
            spent = round(spent + price, 2)
        else:
            dropped.append({"name": item["name"], "price": price})
    return {
        "budget": budget,
        "kept": kept,
        "dropped": dropped,
        "total": spent,
        "remaining": round(budget - spent, 2),
        "over_budget_by": round(max(0.0, sum(float(i["price"]) for i in items) - budget), 2),
    }
