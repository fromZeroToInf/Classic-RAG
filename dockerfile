FROM ghcr.io/astral-sh/uv:0.12.3-python3.12-trixie-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen  --no-install-project  --no-dev

COPY ./src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
