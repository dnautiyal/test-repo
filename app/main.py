from fastapi import FastAPI
import json
from pydantic import BaseModel
from transformers import pipeline
from starlette.responses import RedirectResponse

class PredictionRequest(BaseModel):
  query_string: str

app = FastAPI()

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return "Service is online"

@app.post("/my-endpoint/")
def my_endpoint(request: PredictionRequest):
  # YOUR CODE GOES HERE
  sentiment_model = pipeline("sentiment-analysis")
  sentiment_query_sentence = request.query_string
  sentiment = sentiment_model(sentiment_query_sentence)
  print(f"Sentiment test: {sentiment_query_sentence} == {sentiment}")
  return_dictionary = {
      "query_sentence" : sentiment_query_sentence,
      "sentiment_label":sentiment[0]['label'],
      "sentiment_score":sentiment[0]['score']
  }
  return return_dictionary