from fastapi import FastAPI

app = FastAPI(
    title="Démarches API",
    description="Fransa'daki idari işlemler için RAG tabanlı asistan",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
