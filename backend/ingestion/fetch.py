"""
fetch.py — Belge çekme ve temizleme scripti

Kullanım:
  python fetch.py                          # tüm konuları işle
  python fetch.py --topic titre-sejour-etudiant cvec   # belirli konular
  python fetch.py --manual                 # sadece manual/ klasöründeki dosyalar
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

# --- Ayarlar -----------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MANUAL_DIR = BASE_DIR / "manual"
TOPICS_FILE = BASE_DIR / "topics.json"

REQUEST_DELAY = 2.0        # istekler arası bekleme süresi (saniye)
REQUEST_TIMEOUT = 30       # tek istek için zaman aşımı (saniye)
USER_AGENT = (
    "DemarchesBot/1.0 (educational RAG project; "
    "github.com/sevimakpinareng/demarches)"
)

# Ana içerik için denenen CSS/ID seçiciler (sırayla)
CONTENT_SELECTORS = [
    {"name": "main"},
    {"name": "article"},
    {"id": "content"},
    {"id": "main-content"},
    {"class_": "content"},
    {"class_": "main-content"},
    {"role": "main"},
]

# -----------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# --- robots.txt kontrolü -----------------------------------------------

def is_crawl_allowed(url: str) -> bool:
    """robots.txt'e göre URL'nin taranmasına izin verilip verilmediğini kontrol et."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        # robots.txt okunamazsa: taramaya izin var say (güvenli taraf)
        return True


# --- HTML temizleme ----------------------------------------------------

def clean_html(html: str) -> tuple[str, str]:
    """
    HTML metnini düz metne çevirir.
    Döndürür: (temiz_metin, sayfa_başlığı)
    """
    soup = BeautifulSoup(html, "lxml")

    # Sayfa başlığı
    title = soup.title.get_text(strip=True) if soup.title else ""

    # Yapısal gürültüyü kaldır (etiket bazlı)
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "button", "iframe", "noscript",
                     "svg", "img", "picture"]):
        tag.decompose()

    # UI widget'larını class/role bazlı kaldır (paylaşım, abonelik, çerez vb.)
    _NOISE_PATTERNS = [
        {"role": "complementary"}, {"role": "navigation"}, {"role": "search"},
        {"aria-hidden": "true"},
    ]
    _NOISE_CLASSES = ("share", "social", "favorite", "alert", "cookie",
                      "breadcrumb", "pagination", "sidebar", "banner")
    for pattern in _NOISE_PATTERNS:
        for tag in soup.find_all(**pattern):
            tag.decompose()
    for cls in _NOISE_CLASSES:
        for tag in soup.find_all(class_=lambda c: c and cls in c.lower()):
            tag.decompose()

    # Ana içerik alanını bul (sırayla dene)
    content_node = None
    for selector in CONTENT_SELECTORS:
        content_node = soup.find(**selector)
        if content_node:
            break
    if content_node is None:
        content_node = soup.body or soup

    # Düz metni çıkar; 3 karakterden kısa satırlar UI kalıntısıdır, atla
    raw_text = content_node.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.splitlines()]
    text = "\n".join(line for line in lines if len(line) >= 3)

    return text, title


# --- Otomatik çekme ----------------------------------------------------

def fetch_auto(topic: dict, client: httpx.Client) -> dict | None:
    """URL'den içerik çeker, HTML'i temizler, None döndürmesi manuel moda yönlendirir."""
    url = topic["source_url"]

    if not is_crawl_allowed(url):
        log.warning(f"robots.txt izin vermiyor → manuel moda yönleniyor: {url}")
        return None

    try:
        response = client.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        log.error(f"HTTP hatası {e.response.status_code}: {url}")
        return None
    except Exception as e:
        log.error(f"Bağlantı hatası ({url}): {e}")
        return None

    text, page_title = clean_html(response.text)

    return _build_result(
        topic=topic,
        text=text,
        title=page_title or topic["name"],
        fetch_mode="auto",
    )


# --- Manuel okuma ------------------------------------------------------

def fetch_manual(topic: dict) -> dict | None:
    """manual/{topic_id}.txt dosyasından içerik okur."""
    txt_file = MANUAL_DIR / f"{topic['id']}.txt"

    if not txt_file.exists():
        log.warning(f"Manuel dosya bulunamadı: {txt_file}")
        log.warning(
            f"  → {txt_file.name} dosyasını oluşturup içine metni yapıştır."
        )
        return None

    text = txt_file.read_text(encoding="utf-8").strip()
    if not text:
        log.warning(f"Manuel dosya boş: {txt_file}")
        return None

    return _build_result(
        topic=topic,
        text=text,
        title=topic["name"],
        fetch_mode="manual",
    )


# --- Yardımcı fonksiyonlar --------------------------------------------

def _build_result(
    topic: dict, text: str, title: str, fetch_mode: str
) -> dict:
    """Tüm çıktı formatını tek yerden üret (auto ve manual aynı şema)."""
    return {
        "topic_id": topic["id"],
        "name": topic["name"],
        "title": title,
        "category": topic["category"],
        "source_url": topic["source_url"],
        "fetch_mode": fetch_mode,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
    }


def process_topic(topic: dict, client: httpx.Client, force_manual: bool) -> dict | None:
    """Konuyu işler: önce auto dene, başarısız olursa manual'a düş."""
    if force_manual or topic.get("fetch_mode") == "manual":
        return fetch_manual(topic)

    result = fetch_auto(topic, client)
    if result is None:
        log.info(f"Auto başarısız → manuel deneniyor: {topic['id']}")
        return fetch_manual(topic)

    return result


def save_result(result: dict) -> None:
    """Temizlenmiş belgeyi data/{topic_id}.json olarak kaydet."""
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / f"{result['topic_id']}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"Kaydedildi → {output_path.name}  ({len(result['text'])} karakter)")


# --- Ana akış ----------------------------------------------------------

def main(topic_ids: list[str] | None, manual_only: bool) -> None:
    topics_data = json.loads(TOPICS_FILE.read_text(encoding="utf-8"))
    topics = topics_data["topics"]

    if topic_ids:
        topics = [t for t in topics if t["id"] in topic_ids]
        if not topics:
            log.error(f"Konu bulunamadı: {topic_ids}")
            return

    MANUAL_DIR.mkdir(exist_ok=True)

    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for i, topic in enumerate(topics):
            log.info(f"[{i + 1}/{len(topics)}] {topic['name']}")

            result = process_topic(topic, client, force_manual=manual_only)

            if result:
                save_result(result)
            else:
                log.warning(f"Atlandı (ne auto ne manual): {topic['id']}")

            # Son konu değilse sunucuya saygı için bekle
            if i < len(topics) - 1:
                time.sleep(REQUEST_DELAY)

    log.info("Bitti.")


# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="topics.json'dan belge çekip temizler"
    )
    parser.add_argument(
        "--topic", nargs="+", metavar="ID",
        help="İşlenecek konu ID'leri (boşsa tümü)"
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Sadece manual/ klasöründeki .txt dosyalarını oku"
    )
    args = parser.parse_args()

    main(topic_ids=args.topic, manual_only=args.manual)
