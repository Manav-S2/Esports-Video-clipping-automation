# Contributing

This is a proprietary personal project (see [LICENSE](LICENSE)); external contributions
are not accepted at this time. The notes below document the working conventions used in
the codebase.

## Conventions

* **Python**: type hints throughout, `from __future__ import annotations`, module-private
  helpers prefixed `_`. Heavy dependencies (demoparser2, numpy, SDKs) are imported lazily
  or guarded so utility functions stay importable and testable.
* **Pipelines write files**: each stage's contract is its output folder
  (`round_raw/`, `round_edited/`, `round_final/`, `meta/`). Prefer adding a new artifact
  over passing more in-memory state.
* **LLM responses are hostile input**: never `json.loads` a raw model reply — route it
  through `_extract_json` / `_extract_json_block` so fences, bare keys, trailing commas,
  and truncation are handled.
* **Cross-platform**: changes must work from PowerShell on Windows and from a shell on
  Linux/macOS (no hardcoded drive letters outside local config).

## Tests

```powershell
python -m pytest            # or: python -m unittest discover tests -v
```

Unit tests cover pure logic only (scoring, parsing, formatting). Anything that needs
ffmpeg, a network, or a model belongs in a manual dry-run, not the unit suite.
