# 1. Provide CUDA runtime libraries; the GPU driver stays on the host.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# 2. Add uv to install Python and the project's locked dependencies.
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /usr/local/bin/

# 3. Enable HTTPS downloads and Git metadata, then discard package-manager caches.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# 4. Keep the container environment separate from the mounted host environment.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# 5. Work in the mounted repository and open a shell for uv commands.
WORKDIR /workspace
CMD ["bash"]
