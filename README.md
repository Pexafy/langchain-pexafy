# langchain-pexafy

LangChain and LangGraph tools for [Pexafy](https://pexafy.com): semantic search over
9M+ free stock photos (Unsplash, Pexels, Pixabay and six more sources). Each photo
comes back with its URL, alt text, licence and credit line.

## Install

```bash
pip install -U langchain-pexafy
export PEXAFY_API_KEY="pexafy_..."
```

Free key, no card: [pexafy.com/dashboard/api-keys/create](https://pexafy.com/dashboard/api-keys/create/)
(5,000 requests a month).

## Use

```bash
pip install -U langchain langchain-anthropic  # for this example; any LangChain chat model works
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
from langchain.agents import create_agent
from langchain_pexafy import PexafyToolkit

agent = create_agent("anthropic:claude-haiku-4-5", tools=PexafyToolkit().get_tools())

result = agent.invoke({"messages": [
    {"role": "user", "content": "Find a header photo for a post about remote work, with its credit line."}
]})
print(result["messages"][-1].content)
```

`create_agent` runs on LangGraph. The tools also work with `ToolNode` and any
model's `bind_tools()`, sync or async.

## Tools

| Class | Tool name | Arguments |
|---|---|---|
| `PexafySearchPhotos` | `pexafy_search_photos` | `query`, optional `orientation`, `count` |
| `PexafyFindSimilarPhotos` | `pexafy_find_similar_photos` | `photo_id`, optional `count` |
| `PexafyGetPhoto` | `pexafy_get_photo` | `photo_id` |

`PexafyToolkit().get_tools()` returns all three. Options on each: `api_key`,
`max_results` (photos per search when the model gives no `count`; default 6, max 20).

## Output

The tool message is a JSON list, one record per photo:

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

`ToolMessage.artifact` holds the full API records (every image size).

## Errors

- Spent quota, unknown `photo_id`, server error: returned to the model as a tool error.
- Per-minute rate limit: retried automatically.
- Invalid key: raises `pexafy.AuthenticationError`.

## Licence

MIT. On the free plan, show the `credit` line next to each photo you publish.
