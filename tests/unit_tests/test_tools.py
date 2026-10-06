"""Behaviour of the tools against a mocked API: no network, no key needed."""

import json
import re
from pathlib import Path

import httpx
import pexafy
import pytest
import respx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

import langchain_pexafy
from langchain_pexafy import (
    PexafyFindSimilarPhotos,
    PexafyGetPhoto,
    PexafySearchPhotos,
    PexafyToolkit,
)
from langchain_pexafy._format import clean_credit, clean_name, clean_text, summarize

API = "https://api.pexafy.com/api/v1"
PHOTO_ID = "019e1ea7-2e82-7a34-a451-4c6c7c8250f4"


def photo(photo_id=PHOTO_ID, **over):
    d = {
        "photo_id": photo_id,
        "image_url": "https://images.example.com/full.jpg",
        "urls": {"thumb": "t.jpg", "small": "s.jpg", "regular": "r.jpg", "large": "l.jpg"},
        "width": 4896,
        "height": 3264,
        "orientation": "landscape",
        "color_hex": "#CAB8B4",
        "photographer_username": "nympha57",
        "photographer_full_name": "nympha57 None",
        "source": "Pexels",
        "license_type": "free",
        "source_image_url": "https://www.pexels.com/photo/1/",
        "alt_description": "Red bicycle​ against\n a white wall",
        "description": "A long AI description.",
        "blur_hash": "LZN^6#a}tSx]",
        "attribution": {
            "html": "<span>Photo by nympha57 None on Pexels</span>",
            "plain": "Photo by nympha57 None on Pexels (https://pexafy.com/legal/licenses/#pexels)",
        },
    }
    d.update(over)
    return d


def page(*photos):
    return {"success": True, "data": list(photos), "meta": {"request_id": "r"},
            "pagination": {"next_cursor": None, "per_page": len(photos), "has_more": False}}


def error(status, code, message, headers=None):
    body = {"success": False, "data": None, "meta": {"request_id": "r"},
            "error": {"code": code, "message": message, "request_id": "r"}}
    return httpx.Response(status, json=body, headers=headers or {})


def call(tool, args):
    return tool.invoke({"type": "tool_call", "id": "call_1", "name": tool.name, "args": args})


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch):
    monkeypatch.delenv("PEXAFY_API_KEY", raising=False)


# -- what the model receives ----------------------------------------------------


@respx.mock
def test_search_sends_query_filters_and_key():
    ok = httpx.Response(200, json=page(photo()))
    route = respx.get(f"{API}/search/photos").mock(return_value=ok)
    tool = PexafySearchPhotos(api_key="k", max_results=4)
    tool.invoke({"query": "red bicycle", "orientation": ["landscape", "square"]})
    req = route.calls.last.request
    assert req.url.params["q"] == "red bicycle"
    assert req.url.params.get_list("orientation") == ["landscape", "square"]
    assert req.url.params["per_page"] == "4"
    assert req.headers["x-api-key"] == "k"
    assert req.headers["user-agent"].startswith(f"langchain-pexafy/{langchain_pexafy.__version__} ")


@respx.mock
def test_count_overrides_max_results():
    route = respx.get(f"{API}/search/photos").mock(return_value=httpx.Response(200, json=page()))
    PexafySearchPhotos(api_key="k").invoke({"query": "x", "count": 12})
    assert route.calls.last.request.url.params["per_page"] == "12"


@respx.mock
def test_tool_message_has_compact_content_and_full_artifact():
    respx.get(f"{API}/search/photos").mock(return_value=httpx.Response(200, json=page(photo())))
    msg = call(PexafySearchPhotos(api_key="k"), {"query": "red bicycle"})
    assert isinstance(msg, ToolMessage) and msg.status == "success"
    record = json.loads(msg.content)[0]
    assert record == {
        "rank": 1,
        "photo_id": PHOTO_ID,
        "alt_text": "Red bicycle against a white wall",
        "url": "r.jpg",
        "thumbnail_url": "s.jpg",
        "width": 4896,
        "height": 3264,
        "orientation": "landscape",
        "dominant_color": "#CAB8B4",
        "photographer": "nympha57",
        "source": "Pexels",
        "source_page_url": "https://www.pexels.com/photo/1/",
        "license": "free",
        "credit": "Photo by nympha57 on Pexels (https://pexafy.com/legal/licenses/#pexels)",
    }
    assert msg.artifact[0]["blur_hash"] == "LZN^6#a}tSx]"  # the full API record


@respx.mock
def test_no_result_says_so():
    respx.get(f"{API}/search/photos").mock(return_value=httpx.Response(200, json=page()))
    msg = call(PexafySearchPhotos(api_key="k"), {"query": "x"})
    assert msg.content.startswith("No photos matched")
    assert msg.artifact == []


@respx.mock
def test_similar_and_get_photo_hit_their_routes():
    sim = respx.get(f"{API}/photos/{PHOTO_ID}/similar").mock(
        return_value=httpx.Response(200, json=page(photo("other")))
    )
    one = respx.get(f"{API}/photos/{PHOTO_ID}").mock(
        return_value=httpx.Response(200, json={"success": True, "data": photo()})
    )
    out = PexafyFindSimilarPhotos(api_key="k").invoke({"photo_id": PHOTO_ID, "count": 3})
    assert json.loads(out)[0]["photo_id"] == "other"
    assert sim.calls.last.request.url.params["per_page"] == "3"
    msg = call(PexafyGetPhoto(api_key="k"), {"photo_id": PHOTO_ID})
    assert json.loads(msg.content)["photo_id"] == PHOTO_ID
    assert msg.artifact["photo_id"] == PHOTO_ID
    assert one.called


@respx.mock
async def test_async_path():
    respx.get(f"{API}/search/photos").mock(return_value=httpx.Response(200, json=page(photo())))
    out = await PexafySearchPhotos(api_key="k").ainvoke({"query": "x"})
    assert json.loads(out)[0]["photo_id"] == PHOTO_ID


# -- errors ---------------------------------------------------------------------


@pytest.mark.parametrize("response,expected", [
    (error(429, "DAILY_QUOTA_EXCEEDED", "Daily quota exceeded.", {"retry-after": "36000"}),
     "Pexafy rate limit or quota reached: Daily quota exceeded."),
    (error(429, "QUOTA_EXCEEDED", "Monthly API quota exceeded."),
     "Pexafy rate limit or quota reached: Monthly API quota exceeded."),
    (error(404, "PHOTO_NOT_FOUND", "Photo 'x' not found"),
     "No such Pexafy photo: Photo 'x' not found"),
    (error(500, "INTERNAL", "boom"), "Pexafy request failed: boom"),
])
@respx.mock
def test_actionable_errors_reach_the_model(response, expected):
    route = respx.get(url__regex=rf"{API}/.*").mock(return_value=response)
    tool = PexafyGetPhoto(api_key="k")
    tool.client.max_retries = 0  # the 500 would otherwise be retried
    msg = call(tool, {"photo_id": "x"})
    assert msg.status == "error"
    assert msg.content.startswith(expected)
    assert route.call_count == 1  # a spent quota is not retried


@pytest.mark.parametrize("status,exc", [
    (401, pexafy.AuthenticationError),
    (403, pexafy.PermissionError_),
])
@respx.mock
def test_bad_key_raises_instead_of_reaching_the_model(status, exc):
    respx.get(f"{API}/search/photos").mock(return_value=error(status, "AUTH", "Invalid API Key."))
    with pytest.raises(exc):
        call(PexafySearchPhotos(api_key="k"), {"query": "x"})


@respx.mock
async def test_bad_key_raises_async():
    respx.get(f"{API}/search/photos").mock(return_value=error(401, "AUTH", "Invalid API Key."))
    with pytest.raises(pexafy.AuthenticationError):
        await PexafySearchPhotos(api_key="k").ainvoke({"query": "x"})


@respx.mock
def test_per_minute_rate_limit_is_retried():
    route = respx.get(f"{API}/search/photos").mock(side_effect=[
        error(429, "RATE_LIMITED", "Retry after 0s.", {"retry-after": "0"}),
        httpx.Response(200, json=page(photo())),
    ])
    msg = call(PexafySearchPhotos(api_key="k"), {"query": "x"})
    assert msg.status == "success" and route.call_count == 2


@pytest.mark.parametrize("args", [
    {"query": ""},
    {"query": "x" * 251},
    {"query": "x", "count": 0},
    {"query": "x", "count": 21},
    {"query": "x", "orientation": ["vertical"]},
    {"query": "x", "orientation": "landscape"},
])
def test_invalid_arguments_are_refused_before_any_call(args):
    with pytest.raises(ValidationError):
        PexafySearchPhotos(api_key="k").invoke(args)


def test_invalid_arguments_go_back_to_the_model_in_langgraph():
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([PexafySearchPhotos(api_key="k")]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    bad = AIMessage("", tool_calls=[{"name": "pexafy_search_photos", "id": "c1",
                                     "args": {"query": "x" * 300}}])
    msg = graph.compile().invoke({"messages": [bad]})["messages"][-1]
    assert isinstance(msg, ToolMessage) and msg.status == "error" and "250" in msg.content


# -- construction -----------------------------------------------------------------


def test_missing_key_fails_at_construction():
    with pytest.raises(ValidationError, match="PEXAFY_API_KEY"):
        PexafySearchPhotos()


def test_key_from_env_and_explicit_key_wins(monkeypatch):
    monkeypatch.setenv("PEXAFY_API_KEY", "from-env")
    assert PexafySearchPhotos().api_key.get_secret_value() == "from-env"
    assert PexafySearchPhotos(api_key="explicit").api_key.get_secret_value() == "explicit"


def test_key_is_never_printed():
    tool = PexafySearchPhotos(api_key="pexafy_api_secret")
    assert "pexafy_api_secret" not in repr(tool) and "pexafy_api_secret" not in str(tool)


def test_model_sees_only_the_tool_arguments():
    fn = convert_to_openai_tool(PexafySearchPhotos(api_key="k"))["function"]
    assert fn["name"] == "pexafy_search_photos"
    assert set(fn["parameters"]["properties"]) == {"query", "orientation", "count"}
    assert fn["parameters"]["required"] == ["query"]
    for tool in PexafyToolkit(api_key="k").get_tools():
        params = convert_to_openai_tool(tool)["function"]["parameters"]
        assert "api_key" not in json.dumps(params)


def test_toolkit_shares_key_and_settings():
    tools = PexafyToolkit(api_key="shared", max_results=3).get_tools()
    assert [t.name for t in tools] == [
        "pexafy_search_photos", "pexafy_find_similar_photos", "pexafy_get_photo"]
    assert all(t.api_key.get_secret_value() == "shared" and t.max_results == 3 for t in tools)


def test_version_matches_package_metadata():
    pyproject = (Path(__file__).parents[2] / "pyproject.toml").read_text()
    assert langchain_pexafy.__version__ == re.search(r'^version = "(.+)"', pyproject, re.M).group(1)


# -- text cleaning ----------------------------------------------------------------


def test_clean_text():
    assert clean_text("a​ b\n\tc\u0007") == "a b c"
    assert len(clean_text("x" * 400)) == 300 and clean_text("x" * 400).endswith("…")
    assert clean_text(None) == ""


@pytest.mark.parametrize("raw,clean", [
    ("nympha57 None", "nympha57"),
    ("Unknown photographer", ""),
    ("3345557", ""),
    ("Mitchel Lensink", "Mitchel Lensink"),
])
def test_clean_name(raw, clean):
    assert clean_name(raw) == clean


def test_clean_credit():
    assert clean_credit("Photo by Unknown on Pixabay (u)") == "Photo on Pixabay (u)"
    assert clean_credit("Photo by Jane Doe on Unsplash (u)") == "Photo by Jane Doe on Unsplash (u)"


def test_summarize_falls_back_when_fields_are_missing():
    p = pexafy.Photo.from_dict({"photo_id": "p", "image_url": "full.jpg",
                                "description": "only a description",
                                "photographer_username": "user1"})
    s = summarize(p, 1)
    assert s["url"] == "full.jpg"
    assert s["alt_text"] == "only a description"
    assert s["photographer"] == "user1"
