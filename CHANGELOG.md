# Changelog

## 0.1.2 — 2026-10-07

README only: the agent example now says what else it needs
(`pip install langchain langchain-anthropic`), so it runs as written.

## 0.1.1 — 2026-10-07

- An invalid or revoked key raises `pexafy.AuthenticationError` instead of reaching
  the model as a tool error.
- A spent daily or monthly quota reaches the model at once; it used to be retried for
  up to two minutes first (fixed in `pexafy` 0.1.2, now required).
- Shorter README.

## 0.1.0 — 2026-10-06

First release: `PexafySearchPhotos`, `PexafyFindSimilarPhotos`, `PexafyGetPhoto`,
`PexafyToolkit`. Sync and async, compact JSON content with the full records as the
tool artifact, LangChain standard unit and integration tests.
