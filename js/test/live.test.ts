// What the tools return from the live API. Needs PEXAFY_API_KEY; skipped without it.
import { ToolMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";
import { PexafyFindSimilarPhotos, PexafyGetPhoto, PexafySearchPhotos } from "../src/index";

const FIELDS = ["rank", "photo_id", "alt_text", "url", "thumbnail_url", "width", "height", "orientation",
  "dominant_color", "photographer", "source", "source_page_url", "license", "credit"].sort();

const search = async (args: Record<string, unknown>) =>
  JSON.parse((await new PexafySearchPhotos().invoke(args as any)) as string);

describe.skipIf(!process.env.PEXAFY_API_KEY)("live API", () => {
  it("returns complete records", async () => {
    const photos = await search({ query: "a red bicycle leaning against a white wall", count: 3 });
    expect(photos).toHaveLength(3);
    for (const p of photos) {
      expect(Object.keys(p).sort()).toEqual(FIELDS);
      expect(p.url.startsWith("https://")).toBe(true);
      expect(p.credit).toBeTruthy();
    }
  });

  it.each(["portrait", "square"])("applies the %s orientation filter", async (shape) => {
    const photos = await search({ query: "a quiet street in the rain", orientation: [shape], count: 5 });
    expect(photos.length).toBeGreaterThan(0);
    expect(photos.every((p: any) => p.orientation === shape)).toBe(true);
  });

  it("understands a query in another language", async () => {
    expect(await search({ query: "une femme qui lit dans un café", count: 3 })).toHaveLength(3);
  });

  it("finds similar photos without the reference, and gets one photo back", async () => {
    const [ref] = await search({ query: "a lighthouse on a cliff at sunset", count: 1 });
    const similar = JSON.parse((await new PexafyFindSimilarPhotos().invoke({ photo_id: ref.photo_id, count: 4 })) as string);
    expect(similar).toHaveLength(4);
    expect(similar.some((p: any) => p.photo_id === ref.photo_id)).toBe(false);
    const one = JSON.parse((await new PexafyGetPhoto().invoke({ photo_id: ref.photo_id })) as string);
    expect([one.photo_id, one.url]).toEqual([ref.photo_id, ref.url]);
  });

  it("a tool call gets the full records as artifact", async () => {
    const tool = new PexafySearchPhotos({ maxResults: 2 });
    const msg = (await tool.invoke({ type: "tool_call", id: "1", name: tool.name, args: { query: "a cat on a sofa" } })) as ToolMessage;
    expect(msg.artifact).toHaveLength(2);
    expect(msg.artifact[0].urls.regular).toBeTruthy();
  });

  it("reports an unknown photo and an invalid key", async () => {
    await expect(new PexafyGetPhoto().invoke({ photo_id: "00000000-0000-7000-8000-000000000000" })).rejects.toThrow(/No such Pexafy photo/);
    await expect(new PexafySearchPhotos({ apiKey: "pexafy_api_invalid" }).invoke({ query: "x" })).rejects.toThrow(/rejected the API key/);
  });

  it("handles concurrent calls", async () => {
    const tool = new PexafySearchPhotos({ maxResults: 2 });
    const results = await Promise.all(["a cat", "a dog", "a bird", "a horse"].map((q) => tool.invoke({ query: q })));
    expect(results.every((r) => JSON.parse(r as string).length === 2)).toBe(true);
  });
});
