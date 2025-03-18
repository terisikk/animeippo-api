# This is a production Dockerfile for hosting the application

FROM python:3.12-slim-bullseye AS base

WORKDIR /app

FROM base AS builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_VERSION=0.6.7

RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "uv==$UV_VERSION"

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

COPY . .

RUN uv build --wheel

FROM base AS final

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/dist .
COPY docker-entrypoint.sh wsgi.py app.py ./
COPY conf ./conf

CMD ["./docker-entrypoint.sh"]
