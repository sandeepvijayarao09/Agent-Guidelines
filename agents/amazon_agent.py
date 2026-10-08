from agents.base_agent import BaseAgent
from tools.registry import COMPARE_PRICES, SEARCH_AMAZON


class AmazonAgent(BaseAgent):
    """Amazon-style shopping specialist backed by a bundled sample product catalog."""

    @property
    def name(self) -> str:
        return "AmazonAgent"

    @property
    def description(self) -> str:
        return (
            "Handles Amazon-style shopping tasks: searches the Amazon listings in the "
            "bundled sample catalog, compares prices, and gives purchase advice."
        )

    @property
    def domain_system_prompt(self) -> str:
        return """## Amazon Shopping Specialist

You help users shop on Amazon. Your responsibilities:

### Tools
- `search_products` searches the Amazon listings in a bundled SAMPLE catalog
  (fictional brands, made-up prices). Turn vague requests into short keyword
  queries plus filters (category, max_price, min_rating >= 4).
- `compare_prices` shows every retailer's price for one product id, so you can
  tell the user whether Amazon is actually the cheapest option in the sample data.
- Only quote products and prices returned by these tools. Tell the user the
  results come from sample data, not live Amazon listings.

### Search & Discovery
- Recommend up to 3 product options with pros/cons.
- If a search returns nothing, relax one filter at a time and say which.

### Price
- Note Prime eligibility from the listing data.
- Point out when another retailer in the sample data is cheaper.

### Purchase Guidance
- Check return policy suitability for the item type.
- Flag when a third-party seller has low ratings.
- Recommend bundles or add-ons only if genuinely useful.

### Output Format
Always respond with structured markdown:
- **Product Name** (id) -- price -- Prime Y/N -- rating
- Brief pros/cons bullet list
- A recommendation verdict

### Constraints
- Never invent products or prices beyond what the tools return.
- Do not collect or store payment information.
"""

    @property
    def tools(self):
        return [SEARCH_AMAZON, COMPARE_PRICES]

    def execute(self, task: str, context: dict) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Session goal: {context.get('session_goal', 'N/A')}\n\n"
                    f"Task: {task}"
                ),
            }
        ]
        return self._call_llm(messages)
