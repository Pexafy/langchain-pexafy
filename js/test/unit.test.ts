import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { describe, expect, it, vi } from "vitest";
import pkg from "../package.json";
import {
  cleanCredit,
  cleanName,
  cleanText,
  PexafyFindSimilarPhotos,
  PexafyGetPhoto,
  PexafySearchPhotos,
  PexafyToolkit,
  summarize,
  VERSION,
} from "../src/index";

const PHOTO = {
  photo_id: "019e1ea7-2e82-7a34-a451-4c6c7c8250f4",
  image_url: "https://images.example.com/full.jpg",
  urls: { thumb: "t", small: "s", regular: "r" },
  width: 4896,
  height: 3264,
  orientation: "landscape",
  color_hex: "#CAB8B4",
  photographer_username: "nympha57",
  photographer_full_name: "nympha57 None",
  source: "Pexels",
  license_type: "free",
  source_image_url: "https://www.pexels.com/photo/1",
  alt_description: "Red bicycle​ against\n a white wall",
  blur_hash: "LZN^6#a}",
  attribution: { plain: "Photo by nympha57 None on Pexels (https://pexafy.com/legal/licenses/#pexels)" },
};

type Reply = { status: number; body?: unknown; headers?: Record<string, string>; raw?: string };
type Call = { url: URL; headers: Record<string, string> };

/** A fetch that plays the replies in order; "network" throws, "hang" waits for the abort. */
function fakeFetch(replies: Array<Reply | "network" | "hang">) {
  const calls: Call[] = [];
  const fetch = (async (url: URL, init: RequestInit) => {
    calls.push({ url, headers: init.headers as Record<string, string> });
    const r = replies.shift()!;
    if (r === "network") throw new TypeError("fetch failed");
    if (r === "hang") {
      return new Promise((_, reject) =>
        init.signal!.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))),
      );
    }
    return new Response(r.raw ?? JSON.stringify(r.body), { status: r.status, headers: r.headers });
  }) as unknown as typeof globalThis.fetch;
  return { fetch, calls };
}

const ok = (data: unknown): Reply => ({ status: 200, body: { success: true, data } });
const fail = (status: number, code: string, message: string, headers?: Record<string, string>): Reply => ({
  status,
  headers,
  body: { success: false, data: null, error: { code, message } },
});
const toolCall = (name: string, args: Record<string, unknown>) => ({ type: "tool_call" as const, id: "call_1", name, args });

describe("what the model receives", () => {
  it("sends the query, filters and key", async () => {
    const { fetch, calls } = fakeFetch([ok([PHOTO])]);
    await new PexafySearchPhotos({ apiKey: "k", fetch, maxResults: 4 }).invoke({
      query: "red bicycle",
      orientation: ["landscape", "square"],
    });
    const { url, headers } = calls[0];
    expect(url.href.startsWith("https://api.pexafy.com/api/v1/search/photos?")).toBe(true);
    expect(url.searchParams.get("q")).toBe("red bicycle");
    expect(url.searchParams.getAll("orientation")).toEqual(["landscape", "square"]);
    expect(url.searchParams.get("per_page")).toBe("4");
    expect(headers["x-api-key"]).toBe("k");
    expect(headers["user-agent"]).toBe(`langchain-pexafy-js/${VERSION}`);
  });

  it("a tool call gets a ToolMessage with compact content and the full records as artifact", async () => {
    const { fetch } = fakeFetch([ok([PHOTO])]);
    const tool = new PexafySearchPhotos({ apiKey: "k", fetch });
    const msg = (await tool.invoke(toolCall(tool.name, { query: "x" }))) as ToolMessage;
    expect(msg).toBeInstanceOf(ToolMessage);
    expect(msg.status).toBe("success");
    expect(JSON.parse(msg.content as string)).toEqual([
      {
        rank: 1,
        photo_id: PHOTO.photo_id,
        alt_text: "Red bicycle against a white wall",
        url: "r",
        thumbnail_url: "s",
        width: 4896,
        height: 3264,
        orientation: "landscape",
        dominant_color: "#CAB8B4",
        photographer: "nympha57",
        source: "Pexels",
        source_page_url: "https://www.pexels.com/photo/1",
        license: "free",
        credit: "Photo by nympha57 on Pexels (https://pexafy.com/legal/licenses/#pexels)",
      },
    ]);
    expect(msg.artifact[0].blur_hash).toBe("LZN^6#a}");
  });

  it("count overrides maxResults, default is 6", async () => {
    const { fetch, calls } = fakeFetch([ok([]), ok([])]);
    const tool = new PexafySearchPhotos({ apiKey: "k", fetch });
    await tool.invoke({ query: "x" });
    await tool.invoke({ query: "x", count: 12 });
    expect(calls.map((c) => c.url.searchParams.get("per_page"))).toEqual(["6", "12"]);
  });

  it("no result says so", async () => {
    const { fetch } = fakeFetch([ok([])]);
    const tool = new PexafySearchPhotos({ apiKey: "k", fetch });
    const msg = (await tool.invoke(toolCall(tool.name, { query: "x" }))) as ToolMessage;
    expect(msg.content).toBe("No photos matched. Try describing the scene differently.");
    expect(msg.artifact).toEqual([]);
  });

  it("similar and get hit their routes, photo_id is escaped", async () => {
    const { fetch, calls } = fakeFetch([ok([PHOTO]), ok(PHOTO)]);
    await new PexafyFindSimilarPhotos({ apiKey: "k", fetch }).invoke({ photo_id: "a/b", count: 3 });
    const tool = new PexafyGetPhoto({ apiKey: "k", fetch });
    const msg = (await tool.invoke(toolCall(tool.name, { photo_id: PHOTO.photo_id }))) as ToolMessage;
    expect(calls[0].url.pathname).toBe("/api/v1/photos/a%2Fb/similar");
    expect(calls[0].url.searchParams.get("per_page")).toBe("3");
    expect(calls[1].url.pathname).toBe(`/api/v1/photos/${PHOTO.photo_id}`);
    expect(JSON.parse(msg.content as string).photo_id).toBe(PHOTO.photo_id);
    expect(msg.artifact.photo_id).toBe(PHOTO.photo_id);
  });

  it("the model sees only the tool arguments", () => {
    const fn = convertToOpenAITool(new PexafySearchPhotos({ apiKey: "secret-key" })).function;
    expect(fn.name).toBe("pexafy_search_photos");
    expect(Object.keys((fn.parameters as any).properties)).toEqual(["query", "orientation", "count"]);
    expect((fn.parameters as any).required).toEqual(["query"]);
    for (const t of new PexafyToolkit({ apiKey: "secret-key" }).getTools()) {
      expect(JSON.stringify(convertToOpenAITool(t))).not.toContain("secret-key");
    }
  });

  it.each([
    [{ query: "" }],
    [{ query: "x".repeat(251) }],
    [{ query: "x", count: 0 }],
    [{ query: "x", count: 21 }],
    [{ query: "x", orientation: ["vertical"] }],
    [{ query: "x", orientation: "landscape" }],
  ])("refuses invalid input %j before any call", async (input) => {
    const { fetch, calls } = fakeFetch([]);
    await expect(new PexafySearchPhotos({ apiKey: "k", fetch }).invoke(input as any)).rejects.toThrow();
    expect(calls).toHaveLength(0);
  });

  it("in a LangGraph ToolNode, invalid input goes back to the model", async () => {
    const { ToolNode } = await import("@langchain/langgraph/prebuilt");
    const node = new ToolNode([new PexafySearchPhotos({ apiKey: "k" })]);
    const bad = new AIMessage({ content: "", tool_calls: [{ name: "pexafy_search_photos", id: "c1", args: { query: "x".repeat(300) } }] });
    const { messages } = await node.invoke({ messages: [bad] });
    expect(messages[0]).toBeInstanceOf(ToolMessage);
    expect(messages[0].status).toBe("error");
  });
});

describe("errors and retries", () => {
  const search = (fetch: typeof globalThis.fetch, extra = {}) =>
    new PexafySearchPhotos({ apiKey: "k", fetch, ...extra }).invoke({ query: "x" });

  it("retries a per-minute rate limit", async () => {
    const { fetch, calls } = fakeFetch([fail(429, "RATE_LIMITED", "Retry after 0s.", { "retry-after": "0" }), ok([])]);
    await search(fetch);
    expect(calls).toHaveLength(2);
  });

  it.each([
    ["DAILY_QUOTA_EXCEEDED", { "retry-after": "36000" }],
    ["QUOTA_EXCEEDED", undefined],
  ])("does not retry a spent quota (%s)", async (code, headers) => {
    const { fetch, calls } = fakeFetch([fail(429, code, "Quota spent.", headers)]);
    await expect(search(fetch)).rejects.toThrow("Pexafy rate limit or quota reached: Quota spent. (HTTP 429)");
    expect(calls).toHaveLength(1);
  });

  it("retries a 5xx, then reports it", async () => {
    const { fetch, calls } = fakeFetch([
      fail(503, "UNAVAILABLE", "busy", { "retry-after": "0" }),
      fail(500, "INTERNAL", "boom", { "retry-after": "0" }),
    ]);
    await expect(search(fetch, { maxRetries: 1 })).rejects.toThrow("Pexafy request failed: boom (HTTP 500)");
    expect(calls).toHaveLength(2);
  });

  it("says when the key is rejected, without retrying", async () => {
    const { fetch, calls } = fakeFetch([fail(401, "AUTH", "Invalid API Key.")]);
    await expect(search(fetch)).rejects.toThrow("Pexafy rejected the API key: Invalid API Key. (HTTP 401)");
    expect(calls).toHaveLength(1);
  });

  it("says when a photo does not exist", async () => {
    const { fetch } = fakeFetch([fail(404, "PHOTO_NOT_FOUND", "Photo 'x' not found")]);
    await expect(new PexafyGetPhoto({ apiKey: "k", fetch }).invoke({ photo_id: "x" })).rejects.toThrow(
      "No such Pexafy photo: Photo 'x' not found (HTTP 404)",
    );
  });

  it("retries a network failure, then reports it", async () => {
    const { fetch, calls } = fakeFetch(["network", ok([])]);
    await search(fetch, { maxRetries: 1 });
    expect(calls).toHaveLength(2);
    const again = fakeFetch(["network"]);
    await expect(search(again.fetch, { maxRetries: 0 })).rejects.toThrow("Could not reach Pexafy: fetch failed");
  });

  it("gives up after timeoutMs", async () => {
    const { fetch } = fakeFetch(["hang"]);
    await expect(search(fetch, { timeoutMs: 50, maxRetries: 0 })).rejects.toThrow("Pexafy did not answer within 50 ms");
  });

  it("stops at once when the caller aborts", async () => {
    const { fetch, calls } = fakeFetch(["hang", ok([])]);
    const controller = new AbortController();
    const pending = new PexafySearchPhotos({ apiKey: "k", fetch }).invoke({ query: "x" }, { signal: controller.signal });
    pending.catch(() => {});
    await vi.waitFor(() => expect(calls).toHaveLength(1)); // the request is in flight
    controller.abort(new Error("stopped by caller"));
    await expect(pending).rejects.toThrow();
    expect(calls).toHaveLength(1); // no retry after an abort
  });

  it("refuses a 200 that carries no data", async () => {
    const { fetch } = fakeFetch([{ status: 200, raw: "<html>maintenance</html>" }]);
    await expect(search(fetch)).rejects.toThrow("Unexpected response from Pexafy (HTTP 200, no data)");
  });
});

describe("construction", () => {
  it("fails without a key, before any call", () => {
    const saved = process.env.PEXAFY_API_KEY;
    delete process.env.PEXAFY_API_KEY;
    try {
      expect(() => new PexafySearchPhotos()).toThrow(/PEXAFY_API_KEY/);
    } finally {
      if (saved) process.env.PEXAFY_API_KEY = saved;
    }
  });

  it("reads the key from the environment; an explicit key wins", async () => {
    const saved = process.env.PEXAFY_API_KEY;
    process.env.PEXAFY_API_KEY = "from-env";
    try {
      const a = fakeFetch([ok([])]);
      await new PexafySearchPhotos({ fetch: a.fetch }).invoke({ query: "x" });
      expect(a.calls[0].headers["x-api-key"]).toBe("from-env");
      const b = fakeFetch([ok([])]);
      await new PexafySearchPhotos({ apiKey: "explicit", fetch: b.fetch }).invoke({ query: "x" });
      expect(b.calls[0].headers["x-api-key"]).toBe("explicit");
    } finally {
      if (saved) process.env.PEXAFY_API_KEY = saved;
      else delete process.env.PEXAFY_API_KEY;
    }
  });

  it("the toolkit shares one set of options", async () => {
    const { fetch, calls } = fakeFetch([ok([]), ok([])]);
    const tools = new PexafyToolkit({ apiKey: "shared", fetch, maxResults: 3 }).getTools();
    expect(tools.map((t) => t.name)).toEqual(["pexafy_search_photos", "pexafy_find_similar_photos", "pexafy_get_photo"]);
    await tools[0].invoke({ query: "x" });
    await tools[1].invoke({ photo_id: "p" });
    expect(calls.map((c) => [c.headers["x-api-key"], c.url.searchParams.get("per_page")])).toEqual([
      ["shared", "3"],
      ["shared", "3"],
    ]);
  });

  it("VERSION matches package.json", () => {
    expect(VERSION).toBe(pkg.version);
  });
});

describe("text cleaning", () => {
  it("flattens, strips invisible characters and caps", () => {
    expect(cleanText("a​ b\n\tc\u0007")).toBe("a b c");
    expect(cleanText("x".repeat(400))).toHaveLength(300);
  });
  it.each([
    ["nympha57 None", "nympha57"],
    ["Unknown photographer", ""],
    ["3345557", ""],
    ["Mitchel Lensink", "Mitchel Lensink"],
  ])("cleans the name %j", (raw, clean) => {
    expect(cleanName(raw)).toBe(clean);
  });
  it("repairs credit lines and falls back when fields are missing", () => {
    expect(cleanCredit("Photo by Unknown on Pixabay (u)")).toBe("Photo on Pixabay (u)");
    const s = summarize({ photo_id: "p", image_url: "full.jpg", description: "only", photographer_username: "u1" }, 1);
    expect([s.url, s.alt_text, s.photographer]).toEqual(["full.jpg", "only", "u1"]);
  });
});
