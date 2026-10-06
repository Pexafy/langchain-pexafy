"""LangChain's standard unit tests, for each tool."""

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_pexafy import PexafyFindSimilarPhotos, PexafyGetPhoto, PexafySearchPhotos

PHOTO_ID = "019e1ea7-2e82-7a34-a451-4c6c7c8250f4"


class TestSearchPhotosUnit(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return PexafySearchPhotos

    @property
    def tool_constructor_params(self):
        return {"api_key": "pexafy_api_test"}

    @property
    def tool_invoke_params_example(self):
        return {"query": "a red bicycle leaning against a white wall", "count": 3}

    @property
    def init_from_env_params(self):
        return {"PEXAFY_API_KEY": "pexafy_api_env"}, {}, {"api_key": "pexafy_api_env"}


class TestFindSimilarPhotosUnit(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return PexafyFindSimilarPhotos

    @property
    def tool_constructor_params(self):
        return {"api_key": "pexafy_api_test"}

    @property
    def tool_invoke_params_example(self):
        return {"photo_id": PHOTO_ID, "count": 3}


class TestGetPhotoUnit(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return PexafyGetPhoto

    @property
    def tool_constructor_params(self):
        return {"api_key": "pexafy_api_test"}

    @property
    def tool_invoke_params_example(self):
        return {"photo_id": PHOTO_ID}
