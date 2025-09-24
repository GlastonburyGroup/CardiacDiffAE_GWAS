#-------------------------------------------------------------------
# Stage 1: Dependency Builder
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

# ADD THIS SECTION to define the custom PyTorch repository
[[tool.poetry.source]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu118"
priority = "explicit"

[tool.poetry.dependencies]
python = "^3.10"
EOF

# 4. Extract all dependencies from both repos into a single temporary file.
RUN awk '/^\[tool\.poetry\.dependencies\]/{p=1;next} /^\[/{p=0} p && !/python =/' /tmp/CardiacDiffAE_GWAS/pyproject.toml > /tmp/combined_deps.txt
RUN awk '/^\[tool\.poetry\.dependencies\]/{p=1;next} /^\[/{p=0} p && !/python =/' /tmp/ImLatent/pyproject.toml >> /tmp/combined_deps.txt

# 5. De-duplicate the combined list and append it to our pyproject.toml.
RUN awk -F ' = ' '!seen[$1]++' /tmp/combined_deps.txt >> pyproject.toml

# 6. Install all combined and de-duplicated dependencies.
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
RUN poetry install --no-root


#-------------------------------------------------------------------
# Stage 2: Final Production Image
#-------------------------------------------------------------------
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# 7. Copy the pre-built virtual environment from the 'builder' stage.
COPY --from=builder /app/.venv /.venv

# 8. Set the PATH to use the packages from the virtual environment
ENV PATH="/.venv/bin:$PATH"

# 9. Install git for cloning the final app code
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs \
    && rm -rf /var/lib/apt/lists/*

# 10. Clone the repositories into their final location in the image
WORKDIR /app
RUN git clone https://github.com/GlastonburyGroup/CardiacDiffAE_GWAS.git
RUN git clone https://github.com/GlastonburyGroup/ImLatent.git

# 11. Set the PYTHONPATH so the two projects can be imported by scripts
ENV PYTHONPATH="/app/CardiacDiffAE_GWAS:/app/ImLatent"

CMD ["/bin/bash"]
