FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs \
    && rm -rf /var/lib/apt/lists/*
RUN pip install poetry

# 10. Clone the repositories into their final location in the image
WORKDIR /app
RUN git clone https://github.com/GlastonburyGroup/ImLatent.git
RUN git clone https://github.com/GlastonburyGroup/CardiacDiffAE_GWAS.git

RUN poetry init
RUN poetry source add --priority=explicit pytorch https://download.pytorch.org/whl/cu118

RUN cat /app/ImLatent/pyproject.toml | python3 -c "
        import toml, sys
        data = toml.load(sys.stdin)
        deps = data['tool']['poetry']['dependencies']
        for name, version in deps.items():
            if name != 'python':
                if isinstance(version, dict) and 'source' in version:
                    print(f'{name}@{version[\"version\"]} --source {version[\"source\"]}')
                else:
                    print(f'{name}@{version}')
        " | xargs -I {} poetry add {}

RUN cat /app/CardiacDiffAE_GWAS/pyproject.toml | python3 -c "
        import toml, sys
        data = toml.load(sys.stdin)
        deps = data['tool']['poetry']['dependencies']
        for name, version in deps.items():
            if name != 'python':
                if isinstance(version, dict) and 'source' in version:
                    print(f'{name}@{version[\"version\"]} --source {version[\"source\"]}')
                else:
                    print(f'{name}@{version}')
        " | xargs -I {} poetry add {}

RUN poetry install

# 11. Set the PYTHONPATH so the two projects can be imported by scripts
ENV PYTHONPATH="/app/CardiacDiffAE_GWAS:/app/ImLatent"

CMD ["/bin/bash"]
