"""All Pexafy tools at once, sharing one key."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import SecretStr

from .tools import PexafyFindSimilarPhotos, PexafyGetPhoto, PexafySearchPhotos


class PexafyToolkit(BaseToolkit):
    """Search, find similar and get one photo.

    ```python
    from langchain_pexafy import PexafyToolkit

    tools = PexafyToolkit().get_tools()  # reads PEXAFY_API_KEY
    ```
    """

    api_key: Optional[SecretStr] = None
    max_results: int = 6

    def get_tools(self) -> list[BaseTool]:
        kwargs = {"max_results": self.max_results}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        return [
            PexafySearchPhotos(**kwargs),
            PexafyFindSimilarPhotos(**kwargs),
            PexafyGetPhoto(**kwargs),
        ]
