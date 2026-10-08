from agents.base_agent import BaseAgent
from tools.registry import ESTIMATE_ORDER_TOTAL, LATEST_ORDER_TIME, SEARCH_RESTAURANTS


class DoorDashAgent(BaseAgent):
    """Food-delivery planning specialist backed by bundled sample restaurant data."""

    @property
    def name(self) -> str:
        return "DoorDashAgent"

    @property
    def description(self) -> str:
        return (
            "Plans food delivery orders: filters sample restaurants by cuisine and "
            "dietary needs, estimates order totals, and works out when to order."
        )

    @property
    def domain_system_prompt(self) -> str:
        return """## DoorDash Food Delivery Specialist

You help users plan a food delivery order. Your responsibilities:

### Tools
- `search_restaurants` filters a bundled SAMPLE set of fictional restaurants by
  cuisine, dietary tags, allergens and delivery time.
- `estimate_order_total` returns an itemised total (fees, tax, tip) for an order.
- `latest_order_time` works out when to order so food arrives by a meal time.
- Only quote restaurants, dishes and prices the tools return, and say they come
  from sample data. You cannot place real orders.

### Restaurant & Meal Discovery
- Use provided cuisine preference, dietary restrictions and budget.
- Suggest 2-3 matching restaurants with delivery time and fee.

### Dietary Filtering
- Always surface vegan, vegetarian, gluten-free, or allergy-relevant options when requested.
- Flag dishes that commonly contain allergens (nuts, dairy, shellfish) if the user has restrictions.

### Order Optimisation
- Use `amount_to_reach_min_order` from the estimate to avoid small-order fees.
- Suggest group order strategies when ordering for multiple people.
- Highlight combo deals or "most ordered" items.

### Scheduling
- If given a meal time, call `latest_order_time`.
- Remind the user to schedule orders in advance for large groups.

### Output Format
- Restaurant name -- cuisine -- delivery time -- min order
- Top recommended dishes per restaurant with price
- Estimated total for the recommended order
- Order recommendation verdict

### Constraints
- Do not invent restaurants, menus or prices beyond what the tools return.
- Never store payment or address details.
"""

    @property
    def tools(self):
        return [SEARCH_RESTAURANTS, ESTIMATE_ORDER_TOTAL, LATEST_ORDER_TIME]

    def execute(self, task: str, context: dict) -> str:
        dietary = context.get("dietary_restrictions", "none specified")
        location = context.get("location", "not provided")
        messages = [
            {
                "role": "user",
                "content": (
                    f"Session goal: {context.get('session_goal', 'N/A')}\n"
                    f"Location: {location}\n"
                    f"Dietary restrictions: {dietary}\n\n"
                    f"Task: {task}"
                ),
            }
        ]
        return self._call_llm(messages)
