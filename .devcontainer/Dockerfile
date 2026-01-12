FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# RUN uv run dvc init --no-scm
# COPY ../.dvc/config .dvc/config
# COPY ../*.dvc .dvc/
# RUN uv run dvc config core.no_scm true
# RUN uv run dvc pull
