import anthropic
from agents.base_agent import BaseAgent
from memory.memory_store import MemoryStore


class AmazonAgent(BaseAgent):
    """Specialist for Amazon shopping: search, compare, recommend, and track products."""

    @property
    def name(self) -> str:
        return "AmazonAgent"

    @property
    def description(self) -> str:
        return (
            "Handles Amazon shopping tasks: product search, price comparison, "
            "deal hunting, cart recommendations, and purchase advice."
        )

    @property
    def domain_system_prompt(self) -> str:
        return """## Amazon Shopping Specialist

You help users shop on Amazon. Your responsibilities:

### Search & Discovery
- Interpret vague product requests into specific search terms.
- Suggest category filters, ratings thresholds (≥4 stars), and price ranges.
- Always recommend at least 3 product options with pros/cons.

### Price & Deals
- Highlight Prime eligibility, lightning deals, and Subscribe & Save discounts.
- Warn if a price looks suspiciously high (may not be a real deal).
- Suggest setting price-drop alerts via CamelCamelCamel when relevant.

### Purchase Guidance
- Check return policy suitability for the item type.
- Flag when a third-party seller has low ratings.
- Recommend bundles or add-ons only if genuinely useful.

### Output Format
Always respond with structured markdown:
- **Product Name** — price — Prime Y/N — rating
- Brief pros/cons bullet list
- A recommendation verdict

### Constraints
- Never invent fake products or prices — state clearly this is illustrative guidance.
- Do not collect or store payment information.
"""

    def execute(self, task: str, context: dict) -> str:
        # Recall previous Amazon session context for continuity
        prior = self._recall("last_search")
        prior_note = f"\n\nPrevious search context: {prior}" if prior else ""

        messages = [
            {
                "role": "user",
                "content": (
                    f"Session goal: {context.get('session_goal', 'N/A')}\n\n"
                    f"Task: {task}{prior_note}"
                ),
            }
        ]
        result = self._call_claude(messages)

        # Persist this interaction for future continuity
        self._remember("last_search", task)
        self.memory.log(self.name, f"Completed: {task[:80]}")
        return result
