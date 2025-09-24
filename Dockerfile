# Stage 1: Build the environment
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_NO_INTERACTION=1 \
    PATH="/root/.local/bin:$PATH" \
    PYTHONPATH="/app/CardiacDiffAE_GWAS:/app/ImLatent"

# Install system dependencies, clone repos, and clean up in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && cd / \
    && git clone https://github.com/GlastonburyGroup/CardiacDiffAE_GWAS.git /app/CardiacDiffAE_GWAS \
    && git clone https://github.com/GlastonburyGroup/ImLatent.git /app/ImLatent

# Install Python dependencies for BOTH projects and clean up poetry's cache
RUN cd /app/CardiacDiffAE_GWAS && poetry config virtualenvs.in-project true && poetry install --no-root \
    && cd /app/ImLatent && poetry config virtualenvs.in-project true && poetry install \
    && poetry cache clear . --all

# Set final working directory and default command
WORKDIR /app
CMD ["/bin/bash"]
