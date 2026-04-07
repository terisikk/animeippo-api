# Project: Anime Recommendation System

Personalized anime recommendation engine backed by AniList data, with a FastAPI web API and Jupyter notebooks for analysis.

## Code conventions

- Always add imports to the top of the file, not in the middle
- Keep the existing thresholds for coverage requirements, line-lengths, ignore rules etc. unless explicitly told to change them
- `make lint` is the single source of truth for linting errors, not IDE
- The test suite is fast enough that it can be executed fully always unless fixing some very specific issue
- When leaving comments to code, prefer *why* something is done instead of explaining *what* something does, especially if it's evident from the code. If leaving a *what* comment, consider if it should be a small function instead.
- Use `make format` to format code, not manual ruff commands.
- Avoid using getattr, hasattr, type checking etc. Use known architectural patterns to avoid these.

## Workflow

- When asked to create a todo item, write it to `.claude/todos/` as a markdown file and update `.claude/todos/index.md`. Move completed items from Active to Completed in the index.
- Always use redis cache when debugging, unless testing direct provider network code

## Running

- `make serve` — start FastAPI on port 5000
- `make test` — run full test suite with 100% coverage check
- `make lint` / `make format` — check / fix code style
- `make profile` — run `__main__.py` with CPU profiling

## Verifying against real data

Use `/verify` to generate an inline script that fetches real recommendations with cache and inspects results.
