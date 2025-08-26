FROM python:3.12-slim-bookworm
LABEL maintainer="otzslayer@gmail.com"
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG USER_HOME=/home/codespace

# TODO: 인자 정보 추가
# 모든 경로는 프로젝트 루트 폴더를 기준으로 생각
ARG UTIL_PATH=
ARG PREPROCESSING_PATH=
ARG REQUIREMENTS_PATH=

RUN groupadd --gid 1000 codespace \
    && useradd --uid 1000 --gid codespace --shell /bin/bash --create-home codespace

COPY --chown=codespace:codespace ${UTIL_PATH}/ ${USER_HOME}/utils
COPY --chown=codespace:codespace ${PREPROCESSING_PATH}/preprocessor.py \
    ${USER_HOME}/data_preprocessing/
COPY --chown=codespace:codespace ${REQUIREMENTS_PATH}/requirements.txt \
    ${USER_HOME}/

USER codespace
WORKDIR ${USER_HOME}

ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV="${USER_HOME}/.venv"

ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

RUN mkdir -p ${USER_HOME}/artifacts \
    && uv init --python 3.12 \
    && uv venv --python 3.12 --seed \
    && uv pip install -r ${USER_HOME}/requirements.txt