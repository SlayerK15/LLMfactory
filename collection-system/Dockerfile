FROM python:3.13.13-slim

WORKDIR /app

RUN pip install uv==0.11.7

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

RUN uv run playwright install chromium --with-deps

COPY . .

RUN uv run alembic upgrade head || true

EXPOSE 8000 8001

CMD ["uv", "run", "python", "-m", "collection_system.cli", "serve"]
