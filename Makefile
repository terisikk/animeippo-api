.PHONY: install lint format test coverage-lcov profile pydeps clear-cache serve audit

install:
	pip install uv==0.13.6
	uv sync && uv pip install -e .

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check . --select I --fix
	uv run ruff format .

# Lcov report included for vscode coverage
test:
	uv run coverage run
	uv run coverage report

coverage-lcov:
	uv run coverage lcov

# Requires @profile decorator with filename=".profiling/cprofile.pstats" 
# in function from profilehooks
profile:
	uv run python -m animeippo
	uv run python -m gprof2dot -f pstats .profiling/cprofile.pstats > .profiling/cprofile.dot

# For some reason it does not read pyproject.toml, even though it should
pydeps:
	uv run pydeps --only animeippo --cluster --max-bacon 3 animeippo

clear-cache:
	uv run animeippo-clear-cache

serve:
	uv run uvicorn app:app --host 0.0.0.0 --port 5000 --reload

audit:
	uv audit --preview-features audit-command