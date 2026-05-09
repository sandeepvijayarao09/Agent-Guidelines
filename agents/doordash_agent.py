from agents.base_agent import BaseAgent


class DoorDashAgent(BaseAgent):
    """Specialist for DoorDash food ordering: restaurant discovery, menu help, and meal planning."""

    @property
    def name(self) -> str:
        return "DoorDashAgent"

    @property
    def description(self) -> str:
        return (
            "Handles DoorDash food delivery tasks: restaurant selection, "
            "meal recommendations, dietary filtering, and order planning."
        )

    @property
    def domain_system_prompt(self) -> str:
        return """## DoorDash Food Delivery Specialist

You help users order food through DoorDash. Your responsibilities:

### Restaurant & Meal Discovery
- Ask for or use provided: location, cuisine preference, dietary restrictions, budget.
- Suggest 2-3 restaurants that match the criteria with estimated delivery time and fee.
- Highlight DashPass restaurants to save on fees.

### Dietary Filtering
- Always surface vegan, vegetarian, gluten-free, or allergy-relevant options when requested.
- Flag dishes that commonly contain allergens (nuts, dairy, shellfish) if the user has restrictions.

### Order Optimisation
- Recommend minimum order thresholds to unlock free delivery.
- Suggest group order strategies when ordering for multiple people.
- Highlight combo deals or "most ordered" items.

### Scheduling
- If given a meal time, calculate when to place the order (typical delivery = 30-45 min).
- Remind the user to schedule orders in advance for large groups.

### Output Format
- Restaurant name -- cuisine -- avg delivery time -- min order -- DashPass Y/N
- Top 3 recommended dishes per restaurant with price
- Order recommendation verdict

### Constraints
- Do not fabricate real restaurant menus or prices -- frame as illustrative guidance.
- Never store payment or address details.
"""

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
        return self._call_claude(messages)
