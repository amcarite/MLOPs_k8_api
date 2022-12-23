#BaseImage#
FROM python:3.10-slim-buster  as build

RUN apt-get update \
    && apt-get install -y \
         curl \
         build-essential \
         libffi-dev \
    && rm -rf /var/lib/apt/lists/*


RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=/root/.local/bin:${PATH}

WORKDIR /usr/app

RUN python -m venv /usr/app/project
COPY pyproject.toml ./
RUN . /usr/app/project/bin/activate && poetry install

FROM python:3.10-slim-buster as prod



WORKDIR /usr/app
COPY --from=build /usr/app/project /usr/app/project/


ENV PATH /usr/app/project/bin:${PATH}

COPY src/main.py ./

RUN mkdir -p /usr/app/distilbert-base-uncased-finetuned-sst2

# COPY src/models/DataModel.py /usr/app/models

COPY distilbert-base-uncased-finetuned-sst2/ /usr/app/distilbert-base-uncased-finetuned-sst2

# COPY src/models/predict.py /usr/app/models

# HEALTHCHECK --interval=5m --timeout=3s \
#     CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

HEALTHCHECK --start-period=30s CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



#   CMD curl -f http://localhost:8000/health || exit 1
