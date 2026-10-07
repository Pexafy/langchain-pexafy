// The README example. Run: PEXAFY_API_KEY=... ANTHROPIC_API_KEY=... npx tsx examples/agent.ts
import { createAgent } from "langchain";
import { PexafyToolkit } from "../src/index";

const agent = createAgent({
  model: "anthropic:claude-haiku-4-5",
  tools: new PexafyToolkit().getTools(),
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "Find a header photo for a post about remote work, with its credit line." }],
});
console.log(result.messages.at(-1)?.content);
