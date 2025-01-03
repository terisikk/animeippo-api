.PHONY: install
install:
	pipx install poetry==1.7.1
	poetry install && poetry build

.PHONY: lint
lint:
	poetry run bandit -c pyproject.toml -q -r .
	poetry run ruff check .
	poetry run ruff format --check .

.PHONY: format
format:
	poetry run ruff format .

# Lcov report included for vscode coverage
.PHONY: test
test:
	poetry run coverage run
	poetry run coverage report

.PHONY: coverage-lcov
coverage-lcov:
	poetry run coverage lcov

# Requires @profile decorator with filename=".profiling/cprofile.pstats" 
# in function from profilehooks
.PHOY: profile 
profile:
	poetry run python -m animeippo
	poetry run python -m gprof2dot -f pstats .profiling/cprofile.pstats > .profiling/cprofile.dot

# For some reason it does not read pyproject.toml, even though it should
.PHONY: pydeps
pydeps:
	poetry run pydeps --only animeippo --cluster --max-bacon 3 animeippo

.PHONY: serve
serve:
	poetry run flask run --host=0.0.0.0
