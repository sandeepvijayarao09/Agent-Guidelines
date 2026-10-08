"""
Gemini function declarations for the specialist tools.

Each ToolSpec pairs a plain Python function with the schema Gemini sees.
Agents pick the specs they need; BaseAgent wires them into the
function-calling loop the same way master_agent.py exposes the specialists.
"""

from dataclasses import dataclass
from typing import Callable

from google.genai import types

from tools import catalog, food, routines, scheduling


@dataclass(frozen=True)
class ToolSpec:
    name: str
    function: Callable[..., dict]
    description: str
    parameters: dict

    def declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema.model_validate(self.parameters),
        )


def _str(desc: str) -> dict:
    return {"type": "STRING", "description": desc}


def _num(desc: str) -> dict:
    return {"type": "NUMBER", "description": desc}


def _int(desc: str) -> dict:
    return {"type": "INTEGER", "description": desc}


def _list(items: dict, desc: str) -> dict:
    return {"type": "ARRAY", "items": items, "description": desc}


def _obj(properties: dict, required: list[str] | None = None, desc: str | None = None) -> dict:
    schema = {"type": "OBJECT", "properties": properties}
    if required:
        schema["required"] = required
    if desc:
        schema["description"] = desc
    return schema


HHMM = "24-hour clock time, e.g. '07:30'"

# ── Planner ──────────────────────────────────────────────────────────────────

DATE_INFO = ToolSpec(
    name="date_info",
    function=scheduling.date_info,
    description="Get the weekday, ISO week and day offset of a date relative to a reference date.",
    parameters=_obj(
        {
            "target_date": _str("ISO date, e.g. '2026-05-12'"),
            "reference_date": _str("ISO date to measure from; defaults to today"),
        },
        ["target_date"],
    ),
)

BUILD_SCHEDULE = ToolSpec(
    name="build_schedule",
    function=scheduling.build_schedule,
    description=(
        "Build a non-overlapping time-blocked schedule. Places tasks by priority into the gaps "
        "between fixed events, splits long tasks, adds breaks, and reports conflicts and anything "
        "that did not fit. Always use this instead of doing clock math yourself."
    ),
    parameters=_obj(
        {
            "day_start": _str(HHMM),
            "day_end": _str(HHMM),
            "tasks": _list(
                _obj(
                    {
                        "name": _str("Task name"),
                        "minutes": _int("Estimated duration in minutes"),
                        "priority": _int("1 = urgent/important, 2 = normal, 3 = low"),
                    },
                    ["name", "minutes"],
                ),
                "Flexible tasks to place",
            ),
            "fixed_events": _list(
                _obj({"name": _str("Event"), "start": _str(HHMM), "end": _str(HHMM)}, ["name", "start", "end"]),
                "Commitments that cannot move (meetings, meals, commute)",
            ),
            "break_minutes": _int("Break after each work block (default 10)"),
            "max_block_minutes": _int("Longest uninterrupted work block (default 90)"),
        },
        ["day_start", "day_end", "tasks"],
    ),
)

# ── Daily routine ────────────────────────────────────────────────────────────

ROUTINE_TIMELINE = ToolSpec(
    name="routine_timeline",
    function=routines.routine_timeline,
    description=(
        "Turn an ordered list of routine steps into clock times. Use anchor='start' for a routine "
        "that begins at a time (wake-up) or anchor='end' for one that must finish by a time (bedtime)."
    ),
    parameters=_obj(
        {
            "anchor_time": _str(HHMM),
            "steps": _list(_obj({"name": _str("Step"), "minutes": _int("Duration")}, ["name", "minutes"]), "Steps in order"),
            "anchor": _str("'start' or 'end'"),
            "max_minutes": _int("Optional time budget for the whole routine"),
        },
        ["anchor_time", "steps"],
    ),
)


def _habit_streaks_from_list(check_ins: list[dict], today: str | None = None) -> dict:
    # Gemini schemas can't express free-form dict keys, so the tool takes a list.
    return routines.habit_streaks({c["habit"]: list(c.get("dates", [])) for c in check_ins}, today)


HABIT_STREAKS = ToolSpec(
    name="habit_streaks",
    function=_habit_streaks_from_list,
    description=(
        "Compute current streak, longest streak and 7-day completion % per habit from completion dates. "
        "Pass check_ins as a list of {habit, dates}."
    ),
    parameters=_obj(
        {
            "check_ins": _list(
                _obj({"habit": _str("Habit name"), "dates": _list(_str("ISO date"), "Days it was done")}, ["habit", "dates"]),
                "Completion history",
            ),
            "today": _str("ISO date to compute streaks as of; defaults to today"),
        },
        ["check_ins"],
    ),
)

# ── Shopping (sample catalog) ────────────────────────────────────────────────

_SEARCH_PROPS = {
    "query": _str("Keywords; every word must match the product name, category or tags"),
    "category": _str("electronics, furniture, kitchen, fitness or home"),
    "max_price": _num("Maximum price in USD including shipping"),
    "min_rating": _num("Minimum star rating, e.g. 4.0"),
    "limit": _int("Maximum results (default 5)"),
}

SEARCH_PRODUCTS = ToolSpec(
    name="search_products",
    function=catalog.search_products,
    description="Search the bundled sample product catalog across all retailers. Returns best price per product.",
    parameters=_obj({**_SEARCH_PROPS, "retailer": _str("Restrict to one retailer: Amazon, Walmart, Target or Best Buy")}),
)


def _search_amazon(**kwargs) -> dict:
    kwargs.pop("retailer", None)
    return catalog.search_products(retailer="Amazon", **kwargs)


SEARCH_AMAZON = ToolSpec(
    name="search_products",
    function=_search_amazon,
    description="Search the Amazon listings in the bundled sample product catalog.",
    parameters=_obj(dict(_SEARCH_PROPS)),
)

COMPARE_PRICES = ToolSpec(
    name="compare_prices",
    function=catalog.compare_prices,
    description="Compare every retailer's price for one sample-catalog product (use the id from search_products).",
    parameters=_obj({"product_id": _str("Product id, e.g. 'p001'")}, ["product_id"]),
)

ALLOCATE_BUDGET = ToolSpec(
    name="allocate_budget",
    function=catalog.allocate_budget,
    description="Fit a shopping list into a budget by priority; returns what to keep and what to drop.",
    parameters=_obj(
        {
            "items": _list(
                _obj(
                    {"name": _str("Item"), "price": _num("Price in USD"), "priority": _int("1 = must-have, 2 = nice, 3 = optional")},
                    ["name", "price"],
                ),
                "Shopping list",
            ),
            "budget": _num("Total budget in USD"),
        },
        ["items", "budget"],
    ),
)

# ── Food delivery (sample restaurants) ───────────────────────────────────────

SEARCH_RESTAURANTS = ToolSpec(
    name="search_restaurants",
    function=food.search_restaurants,
    description="Search the bundled sample restaurants and return dishes matching dietary needs.",
    parameters=_obj(
        {
            "cuisine": _str("indian, salads, italian, japanese, mexican or american"),
            "dietary": _list(_str("Tag"), "Required tags: vegan, vegetarian, gluten-free"),
            "avoid_allergens": _list(_str("Allergen"), "e.g. dairy, gluten, egg, fish, soy"),
            "max_delivery_minutes": _int("Maximum delivery time"),
        }
    ),
)

ESTIMATE_ORDER_TOTAL = ToolSpec(
    name="estimate_order_total",
    function=food.estimate_order_total,
    description="Itemised total (fees, tax, tip) for an order from a sample restaurant.",
    parameters=_obj(
        {
            "restaurant_id": _str("Restaurant id from search_restaurants, e.g. 'r01'"),
            "items": _list(_obj({"item_id": _str("Dish id"), "quantity": _int("Quantity")}, ["item_id"]), "Dishes to order"),
            "tip_percent": _num("Tip as % of subtotal (default 15)"),
            "tax_rate": _num("Tax rate as a fraction (default 0.0625)"),
        },
        ["restaurant_id", "items"],
    ),
)

LATEST_ORDER_TIME = ToolSpec(
    name="latest_order_time",
    function=food.latest_order_time,
    description="Latest time to place an order so it arrives by the meal time.",
    parameters=_obj(
        {
            "meal_time": _str(HHMM),
            "restaurant_id": _str("Restaurant id"),
            "buffer_minutes": _int("Extra safety margin (default 10)"),
        },
        ["meal_time", "restaurant_id"],
    ),
)
