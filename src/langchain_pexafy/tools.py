"""LangChain tools for the Pexafy photo search API."""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Optional

import pexafy
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, model_validator

from ._format import summarize

__all__ = [
    "PexafySearchPhotos",
    "PexafyFindSimilarPhotos",
    "PexafyGetPhoto",
]

Orientation = Literal["landscape", "portrait", "square"]

QUERY_MAX_LENGTH = 250
MAX_RESULTS = 20

_QUERY_DESCRIPTION = (
    "The photograph you want, as one concise English sentence about its visible "
    "subject and scene, e.g. 'two colleagues laughing in a bright open-plan office'. "
    "Full sentences rank better than keyword lists: the search matches meaning. "
    "Describe what should be in the picture, not what it is for."
)
_ORIENTATION_DESCRIPTION = (
    "Leave unset unless a shape is actually required (a wide banner: landscape; a "
    "phone wallpaper or a vertical story: portrait). It is a hard filter that drops "
    "every photo of another shape before ranking, so setting it without need loses "
    "the best matches. Several values may be combined."
)
_COUNT_DESCRIPTION = f"How many photos to return, 1 to {MAX_RESULTS}."
_PHOTO_ID_DESCRIPTION = (
    "A Pexafy photo_id, as returned by a previous Pexafy search "
    "(e.g. '019e1ecb-0039-7da6-b1ca-987ee4d337c0')."
)


class SearchPhotosInput(BaseModel):
    query: str = Field(min_length=1, max_length=QUERY_MAX_LENGTH, description=_QUERY_DESCRIPTION)
    orientation: Optional[list[Orientation]] = Field(
        default=None, description=_ORIENTATION_DESCRIPTION
    )
    count: Optional[int] = Field(default=None, ge=1, le=MAX_RESULTS, description=_COUNT_DESCRIPTION)


class FindSimilarPhotosInput(BaseModel):
    photo_id: str = Field(min_length=1, description=_PHOTO_ID_DESCRIPTION)
    count: Optional[int] = Field(default=None, ge=1, le=MAX_RESULTS, description=_COUNT_DESCRIPTION)


class GetPhotoInput(BaseModel):
    photo_id: str = Field(min_length=1, description=_PHOTO_ID_DESCRIPTION)


class _PexafyTool(BaseTool):
    """Holds the key and the clients; the subclasses only say what they call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: Optional[SecretStr] = None
    """Pexafy API key. Falls back to the `PEXAFY_API_KEY` environment variable."""

    base_url: str = pexafy.DEFAULT_BASE_URL
    max_results: int = Field(default=6, ge=1, le=MAX_RESULTS)
    """How many photos a search returns when the model does not ask for a number."""

    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    handle_tool_error: bool = True
    """Quota, rate-limit and not-found errors go back to the model as text it can act on."""

    _client: Optional[pexafy.Client] = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _key_from_env(cls, values: Any) -> Any:
        if isinstance(values, dict) and not values.get("api_key"):
            key = os.environ.get("PEXAFY_API_KEY")
            if key:
                values = {**values, "api_key": key}
        return values

    @model_validator(mode="after")
    def _require_key(self) -> _PexafyTool:
        if self.api_key is None or not self.api_key.get_secret_value():
            raise ValueError(
                "No Pexafy API key. Pass api_key= or set PEXAFY_API_KEY. "
                "Keys are created at https://pexafy.com/dashboard/api-keys/ "
                "(free plan, no card)."
            )
        return self

    def _client_kwargs(self) -> dict[str, Any]:
        assert self.api_key is not None
        return {"api_key": self.api_key.get_secret_value(), "base_url": self.base_url}

    def _tag(self, client: Any) -> Any:
        # Lets the API tell requests made through this package from direct SDK use.
        from . import __version__

        client._http.headers["user-agent"] = (
            f"langchain-pexafy/{__version__} pexafy-python/{pexafy.__version__}"
        )
        return client

    @property
    def client(self) -> pexafy.Client:
        if self._client is None:
            self._client = self._tag(pexafy.Client(**self._client_kwargs()))
        return self._client

    def _async_client(self) -> pexafy.AsyncClient:
        # One per call: an httpx.AsyncClient is bound to the event loop that made it.
        return self._tag(pexafy.AsyncClient(**self._client_kwargs()))

    @staticmethod
    def _error(exc: pexafy.PexafyError) -> ToolException:
        if isinstance(exc, pexafy.RateLimitError):
            return ToolException(f"Pexafy rate limit or quota reached: {exc}")
        if isinstance(exc, pexafy.NotFoundError):
            return ToolException(f"No such Pexafy photo: {exc}")
        return ToolException(f"Pexafy request failed: {exc}")

    @staticmethod
    def _photos(result: pexafy.SearchResult) -> tuple[str, list[dict[str, Any]]]:
        photos = list(result.photos)
        summary = [summarize(p, rank) for rank, p in enumerate(photos, start=1)]
        if not summary:
            return "No photos matched. Try describing the scene differently.", []
        return json.dumps(summary, ensure_ascii=False), [p.raw for p in photos]


class PexafySearchPhotos(_PexafyTool):
    """Search free-to-use stock photos by describing the scene.

    Setup:
        ```bash
        pip install -U langchain-pexafy
        export PEXAFY_API_KEY="pexafy_api_..."
        ```

    Instantiate:
        ```python
        from langchain_pexafy import PexafySearchPhotos

        tool = PexafySearchPhotos(max_results=5)
        ```

    Invoke directly with args:
        ```python
        tool.invoke({"query": "a red bicycle leaning against a white wall"})
        ```

    Invoke with a ToolCall (what an agent does): the message content is a JSON list
    of compact photo records; `artifact` holds the full API records.
    """

    name: str = "pexafy_search_photos"
    description: str = (
        "Find real, free-to-use stock photographs (Unsplash, Pexels, Pixabay and other "
        "libraries) by describing the scene in plain English. Use it whenever you need a "
        "photo: a blog or article header, a hero image, an illustration for a section, a "
        "slide or newsletter picture. It finds photographs that already exist; it does not "
        "generate or edit images, and does not find illustrations, logos, icons or named "
        "people. Each result has an image URL, alt text, its licence and the credit line "
        "to print next to the photo."
    )
    args_schema: type[BaseModel] = SearchPhotosInput

    def _filters(self, orientation: Optional[list[str]], count: Optional[int]) -> dict[str, Any]:
        filters: dict[str, Any] = {"per_page": count or self.max_results}
        if orientation:
            filters["orientation"] = list(orientation)
        return filters

    def _run(
        self,
        query: str,
        orientation: Optional[list[str]] = None,
        count: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        try:
            result = self.client.search(query, **self._filters(orientation, count))
        except pexafy.PexafyError as exc:
            raise self._error(exc) from exc
        return self._photos(result)

    async def _arun(
        self,
        query: str,
        orientation: Optional[list[str]] = None,
        count: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        try:
            async with self._async_client() as client:
                result = await client.search(query, **self._filters(orientation, count))
        except pexafy.PexafyError as exc:
            raise self._error(exc) from exc
        return self._photos(result)


class PexafyFindSimilarPhotos(_PexafyTool):
    """Find photos that look like one already found.

    ```python
    from langchain_pexafy import PexafyFindSimilarPhotos

    PexafyFindSimilarPhotos().invoke({"photo_id": "019e1ea7-2e82-7a34-a451-4c6c7c8250f4"})
    ```
    """

    name: str = "pexafy_find_similar_photos"
    description: str = (
        "Find stock photographs that look like a photo returned by an earlier Pexafy "
        "search: same subject, composition and mood. Use it to offer alternatives to a "
        "photo that is close but not quite right, or to build a consistent set. Takes the "
        "photo_id of that photo."
    )
    args_schema: type[BaseModel] = FindSimilarPhotosInput

    def _run(
        self,
        photo_id: str,
        count: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        try:
            result = self.client.similar(photo_id, per_page=count or self.max_results)
        except pexafy.PexafyError as exc:
            raise self._error(exc) from exc
        return self._photos(result)

    async def _arun(
        self,
        photo_id: str,
        count: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        try:
            async with self._async_client() as client:
                result = await client.similar(photo_id, per_page=count or self.max_results)
        except pexafy.PexafyError as exc:
            raise self._error(exc) from exc
        return self._photos(result)


class PexafyGetPhoto(_PexafyTool):
    """Fetch one photo's details and credit line by its photo_id.

    ```python
    from langchain_pexafy import PexafyGetPhoto

    PexafyGetPhoto().invoke({"photo_id": "019e1ea7-2e82-7a34-a451-4c6c7c8250f4"})
    ```
    """

    name: str = "pexafy_get_photo"
    description: str = (
        "Get the details of one Pexafy photo by its photo_id: image URL, size, alt text, "
        "licence and the credit line to print next to it."
    )
    args_schema: type[BaseModel] = GetPhotoInput

    @staticmethod
    def _one(photo: pexafy.Photo) -> tuple[str, dict[str, Any]]:
        return json.dumps(summarize(photo, 1), ensure_ascii=False), photo.raw

    def _run(
        self, photo_id: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> tuple[str, dict[str, Any]]:
        try:
            return self._one(self.client.get_photo(photo_id))
        except pexafy.PexafyError as exc:
            raise self._error(exc) from exc

    async def _arun(
        self, photo_id: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> tuple[str, dict[str, Any]]:
        try:
            async with self._async_client() as client:
                return self._one(await client.get_photo(photo_id))
        except pexafy.PexafyError as exc:
            raise self._error(exc) from exc
