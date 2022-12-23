import logging
import os
from typing import List

from fastapi import FastAPI, Request, Response
from fastapi_cache.decorator import cache
from pydantic import BaseModel
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache import FastAPICache
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_path = "../distilbert-base-uncased-finetuned-sst2"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    return_all_scores=True,
)

logger = logging.getLogger(__name__)
LOCAL_REDIS_URL = "redis://redis:6379"
app = FastAPI()


@app.on_event("startup")
async def startup():
    # redis = aioredis.from_url(redis_url, encoding = 'utf8', decode_responses = True)
    FastAPICache.init(InMemoryBackend())


class SentimentRequest(BaseModel):
    text: list[str]


class Sentiment(BaseModel):
    label: str
    score: float

class SentimentResponse(BaseModel):
    predictions: List[List[Sentiment]]


@app.post("/predict", response_model=SentimentResponse)
@cache(expire=600)
async def predict(sentiments: SentimentRequest):
    return {"predictions": classifier(sentiments.text)}


@app.get("/health")
async def health():
    return {"status": "healthy"}
