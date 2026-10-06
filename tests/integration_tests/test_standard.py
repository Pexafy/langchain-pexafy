"""LangChain's standard integration tests, against the live API.

Needs PEXAFY_API_KEY; skipped without it.
"""

import os

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_pexafy import PexafyFindSimilarPhotos, PexafyGetPhoto, PexafySearchPhotos

pytestmark = pytest.mark.skipif(
    not os.environ.get("PEXAFY_API_KEY"), reason="PEXAFY_API_KEY not set"
)

PHOTO_ID = "019e1ea7-2e82-7a34-a451-4c6c7c8250f4"


class TestSearchPhotosIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self):
        return PexafySearchPhotos

    @property
    def tool_invoke_params_example(self):
        return {"query": "a red bicycle leaning against a white wall", "count": 2}


class TestFindSimilarPhotosIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self):
        return PexafyFindSimilarPhotos

    @property
    def tool_invoke_params_example(self):
        return {"photo_id": PHOTO_ID, "count": 2}


class TestGetPhotoIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self):
        return PexafyGetPhoto

    @property
    def tool_invoke_params_example(self):
        return {"photo_id": PHOTO_ID}
