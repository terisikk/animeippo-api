.PHONY: install
install:
	pipx install poetry==1.4.0
	poetry install && poetry build

.PHONY: lint
lint:
	poetry run bandit -c pyproject.toml -q -r .
	poetry run ruff check .
	poetry run black -q --check .

.PHONY: format
format:
	poetry run black .

.PHONY: test
test:
	poetry run coverage run
	poetry run coverage report

# Requires @profile decorator with filename=".profiling/cprofile.pstats" 
# in function from profilehooks
.PHOY: profile 
profile:
	poetry run python animeippo/main.py
	poetry run python -m gprof2dot -f pstats .profiling/cprofile.pstats > .profiling/cprofile.dot

# For some reason it does not read pyproject.toml, even though it should
.PHONY: pydeps
pydeps:
	poetry run pydeps --only animeippo --max-bacon 0 animeippo/main.py

.PHONY: serve
serve:
	poetry run flask run --host=0.0.0.0
