# Stage 1: Build the environment
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Set environment variables for non-interactive setup and Poetry configuration
ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_NO_INTERACTION=1 \
    PATH="/root/.local/bin:$PATH"

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# 3. Clone your specific repositories into the image
WORKDIR /app
RUN git clone https://github.com/GlastonburyGroup/CardiacDiffAE_GWAS.git
RUN git clone https://github.com/GlastonburyGroup/ImLatent.git

# 4. Install environment for the FIRST repository
WORKDIR /app/CardiacDiffAE_GWAS
RUN poetry config virtualenvs.in-project true
RUN poetry install --no-root

# 5. Install environment for the SECOND repository
WORKDIR /app/ImLatent
RUN poetry config virtualenvs.in-project true
RUN poetry install --no-root

# 6. Final configuration
WORKDIR /app
ENV PYTHONPATH="${PYTHONPATH}:/app/CardiacDiffAE_GWAS:/app/ImLatent"

# Set a default command to show that the container is ready
CMD ["/bin/bash"]
