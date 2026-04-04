# Builder stage
FROM ghcr.io/astral-sh/uv:debian AS builder
WORKDIR /app

# Install Python 3.12
RUN uv python install 3.12

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./

# Install dependencies into a virtual environment
RUN uv sync --locked --no-install-project

# Copy source code
COPY src/ ./src/

# Install the project
RUN uv sync

# Runtime stage
FROM debian:bookworm-slim AS runtime
WORKDIR /app

# Copy the virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy project files needed for runtime
COPY --from=builder /app/pyproject.toml ./

# Make sure we use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Run the application
CMD ["python", "-m", "continuous_calibration"]
