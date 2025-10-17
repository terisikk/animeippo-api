.PHONY: install
install:
	pip install uv==0.7.2
	uv sync && uv pip install -e .

.PHONY: lint
lint:
	uv run ruff check .
	uv run ruff format --check .

.PHONY: format
format:
	uv run ruff format .

# Lcov report included for vscode coverage
.PHONY: test
test:
	uv run coverage run
	uv run coverage report

.PHONY: coverage-lcov
coverage-lcov:
	uv run coverage lcov

# Requires @profile decorator with filename=".profiling/cprofile.pstats" 
# in function from profilehooks
.PHOY: profile 
profile:
	uv run python -m animeippo
	uv run python -m gprof2dot -f pstats .profiling/cprofile.pstats > .profiling/cprofile.dot

# For some reason it does not read pyproject.toml, even though it should
.PHONY: pydeps
pydeps:
	uv run pydeps --only animeippo --cluster --max-bacon 3 animeippo

.PHONY: serve
serve:
	uv run flask run --host=0.0.0.0
