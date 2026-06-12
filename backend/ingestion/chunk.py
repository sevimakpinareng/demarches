"""
chunk.py — Temizlenmiş belgeleri anlamlı parçalara böler

Strateji:
  - Paragraf/satır sınırlarına saygı gösterir (rastgele kesmez)
  - Hedef: ~300-500 token per chunk (≈ 1350-2250 karakter)
  - Başlık bağlamı bir sonraki chunk'a taşınır
  - Her chunk'a metadata eklenir: kaynak URL, başlık, kategori

Kullanım:
  python chunk.py                  # data/ altındaki tüm JSON'ları işle
  python chunk.py --topic cvec     # tek konu
  python chunk.py --show 5         # N örnek chunk göster (varsayılan 3)
"""

import argparse
import json
from pathlib import Path

# --- Ayarlar -----------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_FILE = BASE_DIR / "chunks.json"

# Fransız metni için 1 token ≈ 4.5 karakter (GPT tokenizer'a yakın tahmin)
CHARS_PER_TOKEN = 4.5

MIN_TOKENS = 80     # bu altındaki chunk'lar öncekiyle birleştirilir
TARGET_TOKENS = 400  # ideal boyut (300-500 aralığının ortası)
MAX_TOKENS = 520    # bu üstüne çıkmadan chunk'ı kapat

MIN_CHARS = int(MIN_TOKENS * CHARS_PER_TOKEN)   # ~360
MAX_CHARS = int(MAX_TOKENS * CHARS_PER_TOKEN)   # ~2340

# -----------------------------------------------------------------------


def approx_tokens(text: str) -> int:
    """Karakter sayısından yaklaşık token sayısı hesapla."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def is_heading(line: str) -> bool:
    """
    Satırın başlık olup olmadığını sezgisel olarak tespit et.
    Kriter: 15-90 karakter, büyük harfle başlıyor, cümle noktalama işareti yok.
    Minimum 15 karakter: "Facebook", "Linkedin" gibi kısa UI kalıntılarını eliyor.
    Fransız idari metinlerde başlıklar genellikle bu kalıba uyar.
    """
    line = line.strip()
    if not (15 <= len(line) <= 90):
        return False
    # Cümle biten satırlar başlık değildir
    if line[-1] in ".!?,;…":
        return False
    # Büyük harfle başlamalı veya rakamla (madde numarası: "1. Conditions...")
    return line[0].isupper() or line[0].isdigit()


def make_chunk(doc: dict, index: int, heading: str, lines: list[str]) -> dict:
    """Satır listesinden chunk nesnesi oluştur."""
    text = "\n".join(lines)
    return {
        "chunk_id": f"{doc['topic_id']}_{index}",
        "topic_id": doc["topic_id"],
        "name": doc["name"],
        "title": doc["title"],
        "category": doc["category"],
        "source_url": doc["source_url"],
        "heading": heading,
        "text": text,
        "char_count": len(text),
        "approx_tokens": approx_tokens(text),
        # chunk_index ve total_chunks son adımda doldurulur
        "chunk_index": index,
        "total_chunks": 0,
    }


def split_into_chunks(doc: dict) -> list[dict]:
    """
    Bir belgeyi chunk listesine dönüştür.

    Algoritma:
    1. Metni satırlara ayır.
    2. Her satır için başlık mı yoksa içerik mi olduğuna karar ver.
    3. Satırları bir tamponda topla; boyut MAX_CHARS'ı aşınca chunk'ı kapat.
    4. Başlık geldiğinde mevcut tampon yeterliyse chunk'ı kapat ve
       başlığı bir sonraki chunk'ın bağlamı olarak sakla.
    5. Çok küçük son chunk'ı öncekiyle birleştir.
    """
    lines = [ln.strip() for ln in doc["text"].splitlines() if ln.strip()]

    chunks: list[dict] = []
    buffer: list[str] = []
    buf_chars: int = 0
    current_heading: str = ""

    def flush() -> None:
        """Tamponu kaydedip temizle."""
        nonlocal buffer, buf_chars
        if not buffer:
            return
        text = "\n".join(buffer)
        # Çok küçük chunk: bir öncekiyle birleştir
        if chunks and approx_tokens(text) < MIN_TOKENS // 2:
            prev = chunks[-1]
            prev["text"] += "\n" + text
            prev["char_count"] = len(prev["text"])
            prev["approx_tokens"] = approx_tokens(prev["text"])
        else:
            chunks.append(make_chunk(doc, len(chunks), current_heading, buffer))
        buffer = []
        buf_chars = 0

    for line in lines:
        line_chars = len(line) + 1  # +1 newline payı

        if is_heading(line):
            # Tampon yeterliyse kapat, ardından başlığı güncelle
            if buf_chars >= MIN_CHARS:
                flush()
            current_heading = line
            # Başlığı tampona ekle (içeriğiyle birlikte chunk'ta görünsün)
            buffer.append(line)
            buf_chars += line_chars
            continue

        # Boyut limitine ulaştık: chunk'ı kapat
        if buf_chars + line_chars > MAX_CHARS and buf_chars >= MIN_CHARS:
            flush()
            # Başlık bağlamını yeni chunk'a taşı
            if current_heading:
                buffer = [current_heading]
                buf_chars = len(current_heading) + 1

        buffer.append(line)
        buf_chars += line_chars

    flush()  # son tampon

    # chunk_index ve total_chunks'ı güncelle
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i
        chunk["total_chunks"] = total

    return chunks


# --- Çıktı ve gösterim -------------------------------------------------


def print_chunk(chunk: dict) -> None:
    """Tek chunk'ı okunabilir biçimde ekrana yaz."""
    sep = "─" * 60
    preview = chunk["text"][:450]
    if len(chunk["text"]) > 450:
        preview += "…"
    print(f"\n{sep}")
    print(f"  chunk_id  : {chunk['chunk_id']}")
    print(f"  kategori  : {chunk['category']}")
    print(f"  başlık    : {chunk['heading'] or '(yok)'}")
    print(f"  token     : ~{chunk['approx_tokens']}  ({chunk['char_count']} karakter)")
    print(f"  kaynak    : {chunk['source_url']}")
    print(f"  {chunk['chunk_index'] + 1}/{chunk['total_chunks']}. chunk")
    print(f"{sep}")
    print(preview)


def main(topic_ids: list[str] | None, show_n: int) -> None:
    data_files = sorted(DATA_DIR.glob("*.json"))

    if not data_files:
        print("Veri yok: önce 'python fetch.py' çalıştır.")
        return

    if topic_ids:
        data_files = [f for f in data_files if f.stem in topic_ids]
        if not data_files:
            print(f"Konu bulunamadı: {topic_ids}")
            return

    all_chunks: list[dict] = []

    for path in data_files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        chunks = split_into_chunks(doc)
        all_chunks.extend(chunks)

        avg_tok = (
            sum(c["approx_tokens"] for c in chunks) // len(chunks)
            if chunks else 0
        )
        print(
            f"{doc['topic_id']:40s}  "
            f"{len(chunks):3d} chunk  "
            f"ort. {avg_tok} token/chunk"
        )

    CHUNKS_FILE.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nToplam {len(all_chunks)} chunk → {CHUNKS_FILE.name}")

    if show_n > 0 and all_chunks:
        print(f"\n{'='*60}")
        print(f"ÖRNEK {min(show_n, len(all_chunks))} CHUNK")
        print("=" * 60)
        for chunk in all_chunks[:show_n]:
            print_chunk(chunk)
        print()


# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Temizlenmiş belgeleri ~300-500 tokenlik chunk'lara böl"
    )
    parser.add_argument(
        "--topic", nargs="+", metavar="ID",
        help="İşlenecek konu ID'leri (boşsa tümü)"
    )
    parser.add_argument(
        "--show", type=int, default=3, metavar="N",
        help="Ekranda gösterilecek örnek chunk sayısı (varsayılan: 3)"
    )
    args = parser.parse_args()

    main(topic_ids=args.topic, show_n=args.show)
