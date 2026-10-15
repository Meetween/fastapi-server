# Production image for the FastAPI server.
#
#   docker build \
#     --build-arg VERSION=$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2) \
#     --build-arg REVISION=$(git rev-parse HEAD) \
#     -t fastapi-server .
#
# VERSION and REVISION only feed the OCI labels. The deploy script compares the version label
# with what /health reports, so CI must pass the pyproject version; a local build without them
# still works and is labelled "dev".
#
# The container is configured only through environment variables, see the table in README.md.
# There is no .env in the image: .dockerignore keeps local env files out of the context.

# Same uv version as CI, so the lock file is read by the tool that wrote it.
FROM ghcr.io/astral-sh/uv:0.9.13@sha256:f07d1bf7b1fb4b983eed2b31320e25a2a76625bdf83d5ff0208fe105d4d8d2f5 AS uv


# python:3.12-slim by multi-arch index digest, so the same line builds linux/amd64 on CI and
# linux/arm64 on a Mac. The runtime stage uses the same digest: the venv links to this
# interpreter. Written out twice rather than through an ARG so that Dependabot can bump both.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS builder

COPY --from=uv /uv /usr/local/bin/uv

# Bytecode is compiled once here instead of on first import as a user who cannot write it.
# Copy, not hardlink, because the cache is a mount and does not end up in the image.
# The interpreter is the image's own, uv must not download another one.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first, in their own layer, so a code change does not reinstall torch.
# On Linux the lock resolves torch to the CPU wheel index.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-dev --no-install-project

COPY . /app

# The project is not packaged today, so this is a check that the lock still matches. It
# installs the project if a build system is ever added to pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ARG VERSION=dev
ARG REVISION=unknown

LABEL org.opencontainers.image.title="fastapi-server" \
      org.opencontainers.image.description="Meeting intelligence API for Meetween, backed by Triton" \
      org.opencontainers.image.source="https://github.com/Meetween/fastapi-server" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

# pydub decodes audio through the ffmpeg binary.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uid and gid 1000 match the ubuntu user on the VM, so the recordings bind mount is writable
# from both sides. The default RECORDINGS_BASE_PATH is created here and owned by that user,
# because the service creates the directory at startup and would fail on a root-owned /mnt.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /mnt/recordings \
    && chown app:app /mnt/recordings

WORKDIR /app

# Code and dependencies stay owned by root, so the running service cannot rewrite itself.
# The venv and the code are separate layers: a code change ships kilobytes, not torch.
COPY --from=builder /app/.venv /app/.venv
COPY . /app

# librosa compiles some functions with numba, which caches them next to the source, else in
# the home directory. site-packages is read-only for the app user and a home directory is not
# guaranteed under every run option, so the cache goes to /tmp.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba

USER 1000:1000

EXPOSE 8000

# The shell is only there to expand HOST, PORT and LOG_LEVEL at start; exec hands PID 1 to
# uvicorn so it receives SIGTERM from docker stop. The VM runs with --network host and sets
# PORT per environment. No HEALTHCHECK: the deploy script polls /health itself.
CMD ["sh", "-c", "exec uvicorn main:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8000}\" --log-level \"$(echo \"${LOG_LEVEL:-info}\" | tr '[:upper:]' '[:lower:]')\""]
