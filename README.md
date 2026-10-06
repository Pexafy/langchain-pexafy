# langchain-pexafy

[LangChain](https://docs.langchain.com) and [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)
tools for [Pexafy](https://pexafy.com): semantic search over 9M+ free stock photos
from Unsplash, Pexels, Pixabay and six more libraries, behind one API. Your agent
describes the picture it needs in plain English and gets back real photographs,
each with its image URL, alt text, licence and the credit line to print.

```bash
pip install -U langchain-pexafy
export PEXAFY_API_KEY="pexafy_api_..."
```

Get a key at [pexafy.com/dashboard/api-keys/create](https://pexafy.com/dashboard/api-keys/create/) (choose **Pexafy API**).
The free plan gives 5,000 requests a month at 20 a minute; the key is issued
immediately, with no card and no app review.

## Tools

| Tool | Name the model sees | What it does |
|---|---|---|
| `PexafySearchPhotos` | `pexafy_search_photos` | Search by describing the scene; optional `orientation` filter |
| `PexafyFindSimilarPhotos` | `pexafy_find_similar_photos` | Photos that look like one already found, by `photo_id` |
| `PexafyGetPhoto` | `pexafy_get_photo` | One photo's details and credit line, by `photo_id` |
| `PexafyToolkit` | — | The three tools above, sharing one key |

```python
from langchain_pexafy import PexafySearchPhotos

search = PexafySearchPhotos(max_results=3)
print(search.invoke({"query": "a red bicycle leaning against a white wall"}))
```

Each photo comes back as a compact record, sized for a model's context:

```json
{
  "rank": 1,
  "photo_id": "019e1ea7-2e82-7a34-a451-4c6c7c8250f4",
  "alt_text": "Red bicycle parked against white wall with front wheel facing left and back wheel right",
  "url": "https://images.unsplash.com/photo-1520538254843-27a40bae5e3a?w=1280",
  "thumbnail_url": "https://images.unsplash.com/photo-1520538254843-27a40bae5e3a?w=400",
  "width": 4896,
  "height": 3264,
  "orientation": "landscape",
  "dominant_color": "#CAB8B4",
  "photographer": "Mitchel Lensink",
  "source": "Unsplash",
  "source_page_url": "https://unsplash.com/photos/red-bicycle-near-white-wall-Hx_dY7Xeszo",
  "license": "free",
  "credit": "Photo by Mitchel Lensink on Unsplash (https://pexafy.com/legal/licenses/#unsplash)"
}
```

When an agent calls the tool, the `ToolMessage` content is that JSON list and its
`artifact` holds the full API records (five image sizes, blur hash, HTML credit…),
so your code can use them without spending the model's tokens.

## In an agent

`create_agent` runs on LangGraph:

```python
from langchain.agents import create_agent
from langchain_pexafy import PexafyToolkit

agent = create_agent(
    "anthropic:claude-haiku-4-5",
    tools=PexafyToolkit(max_results=4).get_tools(),
    system_prompt="You illustrate articles. For each section, pick one photo, "
                  "give its URL, alt text and the exact credit line.",
)

result = agent.invoke({"messages": [{
    "role": "user",
    "content": "Blog post 'A weekend in Lisbon': sections 'Trams', "
               "'Pastéis de nata', 'Sunset at Miradouro'. One photo each.",
}]})
print(result["messages"][-1].content)
```

The tools work the same in a hand-built LangGraph graph (`ToolNode(tools)`) and
in any model's `bind_tools(...)`. Every tool has an async path (`ainvoke`).

## Writing queries

Search runs on meaning, not keywords: `two people hiking on a ridge at dawn` ranks
better than `hiking dawn people`. Describe what should be in the picture rather
than what it is for. Leave `orientation` unset unless a shape is really needed: it
removes every photo of another shape before ranking.

The tool descriptions already tell the model this, so an agent writes good queries
without help.

## Errors

Rate limits, exhausted quotas and unknown photo ids come back to the model as a
short message it can act on (`handle_tool_error=True`). A missing key fails at
construction, before any call. Set `handle_tool_error=False` to get the exception
instead.

## Credits and licences

Photos are free to use under their library's licence; `license` and `source_page_url`
say which. On the free plan, print the `credit` line next to each photo you
publish. Pexafy finds existing photographs: it does not generate images.

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit_tests                                  # offline
PEXAFY_API_KEY=... pytest tests/integration_tests        # live API
```

Both suites are LangChain's standard tool tests (`langchain-tests`).

## Links

- [Pexafy API docs](https://docs.pexafy.com) · [Python SDK](https://github.com/Pexafy/pexafy-python) (`pip install pexafy`)
- [Pexafy MCP server](https://github.com/Pexafy/pexafy-mcp) for MCP clients (`https://mcp.pexafy.com/mcp`)

MIT licence.
