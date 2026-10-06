"""An agent that illustrates a blog post outline with real, credited photos.

    pip install langchain-pexafy langchain langchain-anthropic
    export PEXAFY_API_KEY=... ANTHROPIC_API_KEY=...
    python examples/agent.py
"""

from langchain.agents import create_agent

from langchain_pexafy import PexafyToolkit

agent = create_agent(
    "anthropic:claude-haiku-4-5",
    tools=PexafyToolkit(max_results=4).get_tools(),
    system_prompt=(
        "You illustrate articles. For each section, pick one photo, give its URL, "
        "alt text and the exact credit line."
    ),
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Blog post 'A weekend in Lisbon': sections 'Trams', 'Pastéis de nata', "
                   "'Sunset at Miradouro'. One photo each.",
    }]
})
print(result["messages"][-1].content)
