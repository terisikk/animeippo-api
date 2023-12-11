# This is a production Dockerfile for hosting the application

FROM python:3.12-slim-bullseye as base

WORKDIR /app

FROM base as builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.4.0

RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.in-project true && \
    poetry install --only=main --no-root

COPY . .

RUN poetry build && ./.venv/bin/pip install dist/*.whl

FROM base as final

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/dist .
COPY docker-entrypoint.sh wsgi.py app.py ./
COPY conf ./conf

CMD ["./docker-entrypoint.sh"]
