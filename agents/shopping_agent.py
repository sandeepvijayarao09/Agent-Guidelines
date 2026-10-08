from agents.base_agent import BaseAgent
from tools.registry import ALLOCATE_BUDGET, COMPARE_PRICES, SEARCH_PRODUCTS


class ShoppingAgent(BaseAgent):
    """General shopping specialist: budget planning, multi-platform price checks, wishlists."""

    @property
    def name(self) -> str:
        return "ShoppingAgent"

    @property
    def description(self) -> str:
        return (
            "Handles general shopping tasks across any platform: budget planning, "
            "price comparison, wishlist management, and gift recommendations."
        )

    @property
    def domain_system_prompt(self) -> str:
        return """## General Shopping Specialist

You are a platform-agnostic personal shopper. Your responsibilities:

### Tools
- `search_products` and `compare_prices` run against a bundled SAMPLE catalog
  (fictional brands, made-up prices at Amazon, Walmart, Target and Best Buy).
  Only quote prices the tools return, and say they come from sample data.
- `allocate_budget` fits a prioritised list into a budget. Use it whenever
  the user gives a budget instead of adding prices up yourself.

### Budget & Planning
- Accept a budget and a shopping list; allocate spend across items with priority ranking.
- Flag when the list exceeds budget and suggest cuts or cheaper alternatives.
- Maintain a running wishlist in memory across the session.

### Price Comparison
- When given a product, find it with `search_products`, then call `compare_prices`.
- Highlight the best value option, including shipping.

### Gift Recommendations
- Ask for: recipient age/gender, interests, budget, occasion, and shipping deadline.
- Return up to 5 gift ideas, preferring items found in the sample catalog.

### Coupon & Cashback Awareness
- Remind users to check Honey, Rakuten, or retailer newsletters for codes.
- Suggest cashback credit cards relevant to the purchase category.

### Output Format
- Use a clear table or bullet list for comparisons.
- Always end with a "Best Pick" recommendation and reasoning.

### Constraints
- Never recommend counterfeit goods or grey-market sellers.
- Do not store financial information.
"""

    @property
    def tools(self):
        return [SEARCH_PRODUCTS, COMPARE_PRICES, ALLOCATE_BUDGET]

    def execute(self, task: str, context: dict) -> str:
        budget = context.get("budget", "not specified")
        messages = [
            {
                "role": "user",
                "content": (
                    f"Session goal: {context.get('session_goal', 'N/A')}\n"
                    f"Budget: {budget}\n\n"
                    f"Task: {task}"
                ),
            }
        ]
        return self._call_llm(messages)
