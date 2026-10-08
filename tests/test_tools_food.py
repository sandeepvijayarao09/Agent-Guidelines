import pytest

from tools.food import estimate_order_total, latest_order_time, load_restaurants, search_restaurants


def test_restaurants_are_labelled_as_sample_data():
    assert load_restaurants()["_note"].startswith("SAMPLE DATA")


def test_search_by_dietary_tags_and_allergens():
    result = search_restaurants(dietary=["vegan", "gluten-free"])
    for r in result["restaurants"]:
        assert r["matching_dishes"]
        for dish in r["matching_dishes"]:
            assert {"vegan", "gluten-free"} <= set(dish["tags"])
    no_soy = search_restaurants(cuisine="japanese", dietary=["vegan"], avoid_allergens=["soy"])
    assert no_soy["count"] == 0


def test_search_by_delivery_time_sorted_by_rating():
    result = search_restaurants(max_delivery_minutes=35)
    assert all(r["delivery_minutes"] <= 35 for r in result["restaurants"])
    ratings = [r["rating"] for r in result["restaurants"]]
    assert ratings == sorted(ratings, reverse=True)


def test_estimate_order_total_breakdown():
    est = estimate_order_total("r01", [{"item_id": "r01-1", "quantity": 2}, {"item_id": "r01-4", "quantity": 1}], tip_percent=20)
    assert est["subtotal"] == 30.95
    assert est["service_fee"] == 3.71  # 12%
    assert est["tax"] == 1.93  # 6.25%
    assert est["tip"] == 6.19
    assert est["small_order_fee"] == 0.0
    assert est["total"] == round(30.95 + 2.99 + 3.71 + 1.93 + 6.19, 2)


def test_estimate_applies_small_order_fee():
    est = estimate_order_total("r03", [{"item_id": "r03-4", "quantity": 1}], tip_percent=0, tax_rate=0)
    assert est["small_order_fee"] == 2.0
    assert est["amount_to_reach_min_order"] == 12.0


def test_estimate_rejects_unknown_items():
    with pytest.raises(ValueError):
        estimate_order_total("r01", [{"item_id": "r02-1", "quantity": 1}])
    with pytest.raises(ValueError):
        estimate_order_total("r99", [{"item_id": "x"}])


def test_latest_order_time():
    result = latest_order_time("12:30", "r03")
    assert result["order_by"] == "11:35"  # 45 min delivery + 10 min buffer
