# Démarches — Fransa İdari İşlemler Asistanı

Fransa'ya gelen kişilere titre de séjour, sécurité sociale, CAF ve diğer
idari konularda resmi kaynaklara dayalı, kaynak gösteren bir RAG asistanı.

## Teknoloji Yığını

- **Backend:** Python 3.11 + FastAPI
- **Vektör Veritabanı:** Qdrant
- **İlişkisel Veritabanı:** PostgreSQL
- **LLM:** Mistral AI *(sonraki haftalarda)*
- **Frontend:** *(sonraki haftalarda)*

## Proje Yapısı

```
backend/
  api/         → FastAPI endpoint'leri
  ingestion/   → Belge çekme, temizleme ve chunking
  rag/         → Retrieval ve generation (sonraki hafta)
  eval/        → Değerlendirme (sonraki hafta)
  tests/       → Testler
frontend/      → Kullanıcı arayüzü (sonraki hafta)
```

## Kurulum

### 1. Ortam Değişkenleri

```bash
cp .env.example .env
# .env dosyasını düzenle
```

### 2. Python Sanal Ortamı

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Docker Servisleri

```bash
docker-compose up -d
```

## Haftalık Plan

- [x] Hafta 1: Proje iskeleti, Docker, veri toplama ve chunking
- [ ] Hafta 2: Embedding + Qdrant ingestion + retrieval
- [ ] Hafta 3: Mistral entegrasyonu + API endpoint'leri
- [ ] Hafta 4: Frontend + değerlendirme
