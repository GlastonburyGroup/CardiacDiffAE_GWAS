#-------------------------------------------------------------------
# Stage 1: Dependency Builder
# This stage's only job is to create a combined virtual environment.
#-------------------------------------------------------------------
FROM python:3.11-slim as builder

# 1. Install necessary tools (git and poetry)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install poetry

# 2. Clone the source repositories into a temporary location
WORKDIR /tmp
RUN git clone https://github.com/GlastonburyGroup/CardiacDiffAE_GWAS.git
RUN git clone https://github.com/GlastonburyGroup/ImLatent.git

# 3. Prepare the main app directory and create a template pyproject.toml
WORKDIR /app
RUN cat <<EOF > pyproject.toml
[tool.poetry]
name = "combined-env"
version = "0.1.0"
description = "Dynamically combined environment"
authors = ["Dockerfile"]

[tool.poetry.dependencies]
python = "^3.11"
EOF

# 4. Use a shell command (awk) to find, extract, and append the dependencies
#    from each repository's pyproject.toml into our new combined file.
RUN awk '/^\[tool\.poetry\.dependencies\]/{p=1;next} /^\[/{p=0} p' /tmp/CardiacDiffAE_GWAS/pyproject.toml >> pyproject.toml
RUN awk '/^\[tool\.poetry\.dependencies\]/{p=1;next} /^\[/{p=0} p' /tmp/ImLatent/pyproject.toml >> pyproject.toml

# 5. Install all combined dependencies into a local .venv folder.
#    Poetry will automatically resolve any duplicates or version conflicts.
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
RUN poetry install --no-root


#-------------------------------------------------------------------
# Stage 2: Final Production Image
# This stage builds the final, clean image.
#-------------------------------------------------------------------
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# 6. Copy the pre-built virtual environment from the 'builder' stage.
#    This is the key step that brings in all the Python packages at once.
COPY --from=builder /app/.venv /.venv

# 7. Set the PATH to use the packages from the virtual environment
ENV PATH="/.venv/bin:$PATH"

# 8. Install git for cloning the final app code
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# 9. Clone the repositories into their final location in the image
WORKDIR /app
RUN git clone https://github.com/GlastonburyGroup/CardiacDiffAE_GWAS.git
RUN git clone https://github.com/GlastonburyGroup/ImLatent.git

# 10. Set the PYTHONPATH so the two projects can be imported by scripts
ENV PYTHONPATH="/app/CardiacDiffAE_GWAS:/app/ImLatent"

CMD ["/bin/bash"]
