"""What the tools return from the live API. Needs PEXAFY_API_KEY; skipped without it."""

import asyncio
import json
import os

import pexafy
import pytest

from langchain_pexafy import PexafyFindSimilarPhotos, PexafyGetPhoto, PexafySearchPhotos

pytestmark = pytest.mark.skipif(
    not os.environ.get("PEXAFY_API_KEY"), reason="PEXAFY_API_KEY not set"
)

FIELDS = {"rank", "photo_id", "alt_text", "url", "thumbnail_url", "width", "height",
          "orientation", "dominant_color", "photographer", "source", "source_page_url",
          "license", "credit"}


def search(**args):
    return json.loads(PexafySearchPhotos().invoke(args))


def test_search_returns_complete_records():
    photos = search(query="a red bicycle leaning against a white wall", count=3)
    assert len(photos) == 3
    for p in photos:
        assert set(p) == FIELDS
        assert p["url"].startswith("https://") and p["alt_text"] and p["credit"]


@pytest.mark.parametrize("shape", ["portrait", "square"])
def test_orientation_filter_is_applied(shape):
    photos = search(query="a quiet street in the rain", orientation=[shape], count=5)
    assert photos and all(p["orientation"] == shape for p in photos)


def test_query_in_another_language():
    photos = search(query="une femme qui lit dans un café", count=3)
    assert len(photos) == 3


def test_similar_excludes_the_reference_and_get_photo_round_trips():
    ref = search(query="a lighthouse on a cliff at sunset", count=1)[0]
    args = {"photo_id": ref["photo_id"], "count": 4}
    similar = json.loads(PexafyFindSimilarPhotos().invoke(args))
    assert len(similar) == 4 and ref["photo_id"] not in {p["photo_id"] for p in similar}
    one = json.loads(PexafyGetPhoto().invoke({"photo_id": ref["photo_id"]}))
    assert one["photo_id"] == ref["photo_id"] and one["url"] == ref["url"]


def test_unknown_photo_goes_back_to_the_model():
    tool = PexafyGetPhoto()
    msg = tool.invoke({"type": "tool_call", "id": "1", "name": tool.name,
                       "args": {"photo_id": "00000000-0000-7000-8000-000000000000"}})
    assert msg.status == "error" and msg.content.startswith("No such Pexafy photo")


def test_invalid_key_raises():
    with pytest.raises(pexafy.AuthenticationError):
        PexafySearchPhotos(api_key="pexafy_api_invalid").invoke({"query": "x"})


async def test_concurrent_async_calls():
    tool = PexafySearchPhotos(max_results=2)
    queries = ["a cat on a sofa", "a dog on a beach", "a bird on a branch", "a horse in a field"]
    results = await asyncio.gather(*(tool.ainvoke({"query": q}) for q in queries))
    assert all(len(json.loads(r)) == 2 for r in results)
