FROM python:3.11-slim

ARG USER_HOME=/home/codespace
ARG UTIL_PATH=utils
ARG API_PATH=api

RUN groupadd --gid 1000 codespace \
    && useradd --uid 1000 --gid codespace --shell /bin/bash --create-home codespace

COPY --chown=codespace:codespace ${UTIL_PATH}/ ${USER_HOME}/utils
COPY --chown=codespace:codespace ${API_PATH}/bentofile.yaml ${USER_HOME}/
COPY --chown=codespace:codespace ${API_PATH}/requirements.txt ${USER_HOME}/
COPY --chown=codespace:codespace ${API_PATH}/services.py ${USER_HOME}/
COPY --chown=codespace:codespace ${API_PATH}/src/ ${USER_HOME}/${API_PATH}/src

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

USER codespace

RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r ${USER_HOME}/requirements.txt

ENV PATH="${USER_HOME}/.local/bin:${PATH}"

WORKDIR ${USER_HOME}/${API_PATH}
