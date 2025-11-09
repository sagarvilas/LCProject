from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Optional, List
import json
from langchain.tools import tool


_llm_instance = None

def get_llm():
    """Get or create shared LLM instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOllama(
            model="mistral",
            validate_model_on_init=True,
            temperature=0.5
        )
    return _llm_instance

class ReviewInput(BaseModel):
    product_name: str = Field(description="Name of the product")
    product_category: str = Field(description="Category (electronics, books, etc.)")
    rating: int = Field(description="Star rating (1-5)", ge=1, le=5)
    pros: List[str] = Field(description="Positive aspects")
    cons: Optional[List[str]] = Field(description="Negative aspects", default=[])
    purchase_context: Optional[str] = Field(description="Why/when purchased", default="")
    usage_duration: Optional[str] = Field(description="How long used", default="")
    tone: Optional[str] = Field(description="casual, professional, enthusiastic", default="casual")


# Create review generation tool
@tool
def generate_review(input_data: str) -> str:
    """Generates Amazon product reviews. Input should be a JSON string with:
    - product_name: string
    - product_category: string
    - rating: integer (1-5)
    - pros: list of strings
    - cons: list of strings (optional)
    - purchase_context: string (optional)
    - usage_duration: string (optional)
    - tone: string (casual/professional/enthusiastic, optional)
    """
    try:
        data = json.loads(input_data)
        review_input = ReviewInput(**data)

        # Create detailed prompt
        prompt = f"""Write an authentic Amazon product review with the following details:

Product: {review_input.product_name}
Category: {review_input.product_category}
Rating: {review_input.rating} stars
Tone: {review_input.tone}

Positive aspects:
{chr(10).join(f"- {pro}" for pro in review_input.pros)}

{f"Negative aspects:{chr(10)}" + chr(10).join(f"- {con}" for con in review_input.cons) if review_input.cons else ""}

{f"Purchase context: {review_input.purchase_context}" if review_input.purchase_context else ""}
{f"Usage duration: {review_input.usage_duration}" if review_input.usage_duration else ""}

Requirements:
- Write in first person
- Sound like a real customer (natural, conversational)
- Include specific details from the pros/cons
- Match the specified tone
- Be 150-300 words
- Include a helpful title
- Make it authentic and believable
"""

        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Error generating review: {str(e)}"


def create_review_agent():

    # Create prompt template
    prompt = """You are a helpful assistant that generates Amazon product reviews.

Your role:
1. Gather information from the user about their product experience
2. Ask clarifying questions if needed (pros, cons, rating, etc.)
3. Use the generate_review tool to create the review
4. Present the review to the user and offer revisions

Be conversational and helpful. Guide users through the review creation process."""

    # Create tools list
    tools = [generate_review]

    model = get_llm()
    # # Create agent
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt,
        debug=True
    )


if __name__ == "__main__":
    agent = create_review_agent()

    # Example 1: Direct structured input
    review_data = {
        "product_name": "Sony WH-1000XM5 Headphones",
        "product_category": "Electronics",
        "rating": 5,
        "pros": [
            "Exceptional noise cancellation",
            "Comfortable for long periods",
            "Great sound quality",
            "Long battery life"
        ],
        "cons": [
            "Expensive",
            "Case is bulky"
        ],
        "purchase_context": "Bought for daily commute and travel",
        "usage_duration": "3 months",
        "tone": "enthusiastic"
    }


    output =  agent.invoke({
    "messages": [
        {"role": "user", "content": f"Generate a review with this data: {json.dumps(review_data)}"}
    ]})

    print(output)

    # Example 2: Conversational input
    # response = agent.invoke({
    #     "input": "I need help writing a review for a coffee maker I bought"
    # })
    # print(response)