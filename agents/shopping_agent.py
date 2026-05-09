import anthropic
from agents.base_agent import BaseAgent
from memory.memory_store import MemoryStore


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

### Budget & Planning
- Accept a budget and a shopping list; allocate spend across items with priority ranking.
- Flag when the list exceeds budget and suggest cuts or cheaper alternatives.
- Maintain a running wishlist in memory across the session.

### Price Comparison
- When given a product, compare across Amazon, Walmart, Target, and eBay (conceptually).
- Highlight the best value option and whether waiting for a sale makes sense.

### Gift Recommendations
- Ask for: recipient age/gender, interests, budget, occasion, and shipping deadline.
- Return 5 personalised gift ideas with price ranges and where to buy.

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

    def execute(self, task: str, context: dict) -> str:
        budget = context.get("budget", "not specified")
        wishlist = self._recall("wishlist", [])
        wishlist_note = f"\n\nCurrent wishlist: {wishlist}" if wishlist else ""

        messages = [
            {
                "role": "user",
                "content": (
                    f"Session goal: {context.get('session_goal', 'N/A')}\n"
                    f"Budget: {budget}\n\n"
                    f"Task: {task}{wishlist_note}"
                ),
            }
        ]
        result = self._call_claude(messages)

        # Detect if the task adds to the wishlist and persist
        if "add" in task.lower() and "wishlist" in task.lower():
            self._remember("wishlist", wishlist + [task])

        self.memory.log(self.name, f"Completed: {task[:80]}")
        return result
