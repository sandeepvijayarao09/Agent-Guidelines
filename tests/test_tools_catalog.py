import pytest

from tools.catalog import allocate_budget, compare_prices, load_catalog, search_products


def test_catalog_is_labelled_as_sample_data():
    assert load_catalog()["_note"].startswith("SAMPLE DATA")


def test_search_matches_all_keywords_and_ranks_by_rating():
    result = search_products("wireless")
    assert result["notice"].startswith("Results come from a bundled SAMPLE catalog")
    ratings = [p["rating"] for p in result["products"]]
    assert ratings == sorted(ratings, reverse=True)
    assert search_products("wireless headphones")["products"][0]["id"] == "p001"
    assert search_products("unicorn saddle")["count"] == 0


def test_search_filters():
    cheap = search_products(category="kitchen", max_price=30)
    assert [p["id"] for p in cheap["products"]] == ["p011"]
    assert all(p["rating"] >= 4.6 for p in search_products(min_rating=4.6, limit=50)["products"])
    amazon = search_products("air fryer", retailer="Amazon")["products"][0]
    assert amazon["best_retailer"] == "Amazon" and amazon["best_price"] == 89.99
    anywhere = search_products("air fryer")["products"][0]
    assert anywhere["best_retailer"] == "Walmart" and anywhere["best_price"] == 79.00
    assert search_products(limit=3)["count"] == 3


def test_compare_prices_includes_shipping():
    result = compare_prices("p006")
    totals = [o["total"] for o in result["offers"]]
    assert totals == sorted(totals)
    walmart = next(o for o in result["offers"] if o["retailer"] == "Walmart")
    assert walmart["total"] == 258.99  # 239.00 + 19.99 shipping
    assert result["cheapest"] == "Amazon"
    assert result["savings_vs_highest"] == round(259.99 - 249.00, 2)
    with pytest.raises(ValueError):
        compare_prices("nope")


def test_allocate_budget_keeps_priorities_first():
    result = allocate_budget(
        [
            {"name": "Monitor", "price": 279.99, "priority": 1},
            {"name": "Lamp", "price": 34.99, "priority": 3},
            {"name": "Keyboard", "price": 79.99, "priority": 2},
            {"name": "Mouse", "price": 32.50, "priority": 1},
        ],
        budget=400,
    )
    assert [i["name"] for i in result["kept"]] == ["Mouse", "Monitor", "Keyboard"]
    assert [i["name"] for i in result["dropped"]] == ["Lamp"]
    assert result["total"] == 392.48
    assert result["remaining"] == 7.52
    assert result["over_budget_by"] == 27.47
