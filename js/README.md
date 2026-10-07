# langchain-pexafy (LangChain.js)

[LangChain.js](https://docs.langchain.com/oss/javascript/langchain/overview) and LangGraph.js tools
for [Pexafy](https://pexafy.com): semantic search over 9M+ free stock photos (Unsplash, Pexels,
Pixabay and six more sources). Each photo comes back with its URL, alt text, licence and credit line.
For Python, see the [root of this repository](https://github.com/Pexafy/langchain-pexafy).

## Install

```bash
npm install langchain-pexafy @langchain/core
export PEXAFY_API_KEY="pexafy_..."
```

Free key, no card: [pexafy.com/dashboard/api-keys/create](https://pexafy.com/dashboard/api-keys/create/)
(5,000 requests a month).

## Use

```bash
npm install langchain @langchain/anthropic  # for this example; any LangChain chat model works
export ANTHROPIC_API_KEY="sk-ant-..."
```

```ts
import { createAgent } from "langchain";
import { PexafyToolkit } from "langchain-pexafy";

const agent = createAgent({
  model: "anthropic:claude-haiku-4-5",
  tools: new PexafyToolkit().getTools(),
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "Find a header photo for a post about remote work, with its credit line." }],
});
console.log(result.messages.at(-1)?.content);
```

`createAgent` runs on LangGraph.js. The tools also work with `ToolNode` and any model's `bindTools()`.

## Tools

| Class | Tool name | Arguments |
|---|---|---|
| `PexafySearchPhotos` | `pexafy_search_photos` | `query`, optional `orientation`, `count` |
| `PexafyFindSimilarPhotos` | `pexafy_find_similar_photos` | `photo_id`, optional `count` |
| `PexafyGetPhoto` | `pexafy_get_photo` | `photo_id` |

`new PexafyToolkit().getTools()` returns all three. Options on each: `apiKey` (default
`PEXAFY_API_KEY`), `maxResults` (photos per search when the model gives no `count`; default 6,
max 20), `timeoutMs` (default 30000), `maxRetries` (default 2).

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

- Spent quota, unknown `photo_id`, invalid key: the tool throws an error with a short message;
  LangGraph's `ToolNode` passes it to the model as a tool error.
- Per-minute rate limit, server error, network failure: retried automatically.

## Licence

MIT. On the free plan, show the `credit` line next to each photo you publish.
