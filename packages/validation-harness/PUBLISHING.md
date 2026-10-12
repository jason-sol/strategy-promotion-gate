# Publishing

Nothing here has been published. `validation-harness` is unclaimed on PyPI as of
2026-09-07 and the name is not reserved — a claim only happens on first upload.

The bot is the source of truth. `bot/validation.py` and `bot/metrics.py` are the
live implementations; this package is an extraction that must not drift from
them. `tests/test_harness_parity.py` in the parent repo is the release gate.

## Pre-flight

- [ ] `uv run pytest packages/validation-harness/tests -q` — 75 tests green.
- [ ] `uv run pytest tests/test_harness_parity.py -q` — the drift guard. A
      failure here means the bot and the package disagree; fix the drift, do not
      publish around it.
- [ ] `uv run ruff check packages/validation-harness` — clean.
- [ ] Version bumped in `pyproject.toml` (`[project] version`). PyPI refuses a
      re-upload of a version that already exists, and yanking does not free it.
- [ ] Decide on a changelog. There is none today, which is defensible for
      `0.1.0` but not for a second release — add `CHANGELOG.md` (Keep a
      Changelog, one section per version) before cutting `0.2.0`, and add it to
      the sdist check below.
- [ ] `[project.urls]` in `pyproject.toml` resolve for an anonymous visitor —
      PyPI renders them on the project page, so a dead link is public.

## Build and check

```sh
cd packages/validation-harness
rm -rf dist
uv build
uvx twine check dist/*
```

Confirm the artifacts:

```sh
tar tzf dist/validation_harness-*.tar.gz   # LICENSE, README.md, py.typed, pyproject.toml, PKG-INFO
unzip -l dist/validation_harness-*.whl     # py.typed present, LICENSE under dist-info/licenses/
```

Tests ship in the sdist and the parent repo's `.gitignore` is vendored into the
sdist root — both are hatchling defaults and neither is worth fighting.

## TestPyPI dry run

Do this first, every time. TestPyPI is a separate account and a separate token.

```sh
uvx twine upload --repository testpypi dist/*
```

Then install from it, pulling real dependencies from PyPI:

```sh
uv run --no-project --python 3.12 \
  --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match \
  --with validation-harness python -c "import validation_harness as v; print(len(v.__all__))"
```

Expect `16`.

## PyPI publish

Prefer trusted publishing (OIDC) over a long-lived token — no secret to store or
rotate. Configure it once at
`pypi.org/manage/account/publishing/` against the repo, workflow filename, and
optionally a GitHub environment; the workflow then needs `permissions:
id-token: write` and no password at all. Trusted publishing requires the
repository to be reachable by PyPI.

Token fallback, if publishing from a laptop:

```sh
uvx twine upload dist/*
```

Username `__token__`, password the API token — enter it at the prompt or put it
in `~/.pypirc`. Do not commit it and do not paste it into this repo. Scope the
token to this project after the first upload.

## Post-publish smoke

```sh
uv run --no-project --python 3.12 --with validation-harness \
  python -c "import validation_harness as v; print(v.__all__)"
uv run --no-project --python 3.13 --with validation-harness \
  python -c "import validation_harness as v; print(len(v.__all__))"
```

Then check the project page renders the README and shows the MIT licence.

## Notes

- Metadata is PEP 639: `license = "MIT"` as an SPDX expression plus
  `license-files`. The legacy `License :: OSI Approved :: MIT License`
  classifier is deliberately absent — PyPI rejects an upload that carries both a
  `License-Expression` and a `License ::` classifier, and `twine check` does not
  catch it locally.
- `requires-python = ">=3.12"`. 3.12 and 3.13 are exercised; the classifiers
  claim only those.
