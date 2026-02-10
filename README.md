# sentiment-analysis-ml-app
Data → Train Model → Save Model → FastAPI → Web UI → Deploy

# Sentiment Analysis ML App

Production-style machine learning project that predicts sentiment from user text using a trained model exposed through a FastAPI endpoint.

## Project Goal

Build an end-to-end ML system — from training to inference — using clean engineering practices.

## Planned Features

* Train a sentiment classification model
* Serve predictions via FastAPI
* Containerize the application
* Deploy for public access


## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# train (creates models/)
python src/train.py --data data/sample.csv

# serve
uvicorn app.main:app --reload
## 🚀 API Endpoints

### Interactive Docs

FastAPI automatically generates interactive API documentation:

👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Health Check

Verify the service is running:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

---

### Predict Sentiment

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this product!"}'
```

Example response:

```json
{
  "label": "positive",
  "score": 0.94
}
```
### Tech stack
