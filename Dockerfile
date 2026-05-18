FROM  pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

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

RUN pip install toml

RUN cp ImLatent/pyproject.toml ./pyproject.toml
RUN cp CardiacDiffAE_GWAS/pyproject.toml ./pyproject.add.toml

RUN echo "import toml\n\
f_base = 'pyproject.toml'\n\
f_add = 'pyproject.add.toml'\n\
with open(f_base, 'r') as f:\n\
    base_data = toml.load(f)\n\
with open(f_add, 'r') as f:\n\
    add_data = toml.load(f)\n\
base_data['tool']['poetry']['dependencies'].update(add_data['tool']['poetry']['dependencies'])\n\
with open(f_base, 'w') as f:\n\
    toml.dump(base_data, f)" > merge_deps.py

# 4. Execute the newly created script.
RUN python3 merge_deps.py

# 5. Install dependencies from the now-merged pyproject.toml.
RUN poetry install --no-root

RUN rm merge_deps.py pyproject.add.toml

# 11. Set the PYTHONPATH so the two projects can be imported by scripts
ENV PYTHONPATH="/app/CardiacDiffAE_GWAS:/app/ImLatent"

CMD ["/bin/bash"]
