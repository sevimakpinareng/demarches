"""
embed.py — Metin → embedding vektörü dönüşümü

Modül olarak kullanım (index.py ve retrieve.py tarafından import edilir):
    from rag.embed import embed_passages, embed_query, embedding_dim

Standalone kullanım (chunks.json → embeddings.npy):
    python -m rag.embed

Config (.env'den):
    EMBEDDING_MODEL        HuggingFace model ID (varsayılan: intfloat/multilingual-e5-base)
    EMBEDDING_BATCH_SIZE   Batch boyutu (varsayılan: 32)

e5 modellerinde iki farklı ön ek kullanılır:
    "passage: " → belgeler için  (indeksleme aşaması)
    "query: "   → sorgular için  (arama aşaması)
Bu asimetri retrieval kalitesini artırır: model, belge ile sorgunun
farklı "niyetler" taşıdığını öğrenmiştir.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent.parent.parent / ".env")

log = logging.getLogger(__name__)

# --- Config ------------------------------------------------------------

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# e5 ailesi için standart ön ekler
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

# Dosya yolları
_CHUNKS_FILE = Path(__file__).parent.parent / "ingestion" / "chunks.json"
_EMBEDDINGS_FILE = Path(__file__).parent.parent / "ingestion" / "data" / "embeddings.npy"

# -----------------------------------------------------------------------

# Lazy singleton: model ilk çağrıda yüklenir, sonra bellekte kalır
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Modeli ilk kullanımda yükle, sonraki çağrılarda aynı örneği döndür."""
    global _model
    if _model is None:
        log.info("Model yükleniyor: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        log.info("Model hazır. Vektör boyutu: %d", _model.get_sentence_embedding_dimension())
    return _model


def embedding_dim() -> int:
    """Seçili modelin vektör boyutunu döndür (Qdrant koleksiyonu için gerekli)."""
    return get_model().get_sentence_embedding_dimension()


def embed_passages(texts: list[str]) -> np.ndarray:
    """
    Belge/chunk metinleri için embedding üret.

    normalize_embeddings=True → vektörler birim uzunluğa çekilir.
    Bu sayede Qdrant'ta cosine_similarity = dot_product olur → daha hızlı arama.

    Dönüş: shape (N, dim) float32 numpy dizisi
    """
    prefixed = [PASSAGE_PREFIX + t for t in texts]
    return get_model().encode(
        prefixed,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_query(text: str) -> list[float]:
    """
    Tek bir arama sorgusu için embedding üret.
    Dönüş: Python list[float] (Qdrant client'ın beklediği format)
    """
    vec = get_model().encode(
        QUERY_PREFIX + text,
        normalize_embeddings=True,
    )
    return vec.tolist()


# --- Standalone: chunks.json → embeddings.npy -------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not _CHUNKS_FILE.exists():
        log.error("chunks.json bulunamadı: %s", _CHUNKS_FILE)
        log.error("Önce 'python ingestion/chunk.py' çalıştır.")
        return

    chunks = json.loads(_CHUNKS_FILE.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]

    log.info("%d chunk için embedding üretiliyor...", len(texts))
    log.info("Model: %s  |  Boyut: %d", MODEL_NAME, embedding_dim())

    embeddings = embed_passages(texts)  # (N, dim) float32

    _EMBEDDINGS_FILE.parent.mkdir(exist_ok=True)
    np.save(_EMBEDDINGS_FILE, embeddings)

    log.info("Kaydedildi: %s", _EMBEDDINGS_FILE)
    log.info("Shape: %s  |  dtype: %s", embeddings.shape, embeddings.dtype)
    log.info(
        "Örnek vektör (ilk 5 değer): %s",
        embeddings[0, :5].round(4),
    )


if __name__ == "__main__":
    main()
