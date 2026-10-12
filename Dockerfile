FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY config.py ./
COPY bot ./bot

# All runtime state (trades.csv, daily_pnl.csv, state.json, HALT) lands in the
# working directory: mount a persistent volume at /data so it survives deploys.
RUN mkdir /data
WORKDIR /data

CMD ["uv", "run", "--project", "/app", "--no-sync", "python", "-m", "bot.main"]
