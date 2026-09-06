"""IslamQA.info English corpus harvester (owner-approved source).

Harvests the public English fatwa collection of Sheikh Muhammad Saalih
al-Munajjid (islamqa.info — reputable, used worldwide) for the offline
knowledge base:

  Stage 1 — index: topic-category listings (~11 topics, paginated, 10/page)
  Stage 2 — answers: each /en/answers/<id> page; the Q/A content is in the
             Next.js RSC payload (self.__next_f.push flight data)

Polite crawl: serial requests, short delay, standard UA, resumable via
checkpoint JSON. Output: knowledge/web/raw/islamqa_en/<id>.json + index.json
"""

from __future__ import annotations

import functools
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "knowledge" / "web" / "raw" / "islamqa_en"
INDEX_PATH = OUT_DIR / "index.json"

BASE = "https://islamqa.info/en"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) IlmanKB/1.0 (local study)"}
TOPIC_IDS = ["3", "25", "30", "34", "55", "210", "247", "258", "261", "266", "269"]
DELAY_S = 1.0

print = functools.partial(print, flush=True)  # noqa: F811 - unbuffered for nohup


def _get(url: str) -> str | None:
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200:
            return None
        return r.text
    except requests.RequestException:
        return None


def harvest_index() -> list[dict]:
    """Stage 1: collect (id, title, topic) from every topic listing page."""
    seen: dict[int, dict] = {}
    for topic in TOPIC_IDS:
        page = 1
        while True:
            url = f"{BASE}/categories/topics/{topic}" + (f"?page={page}" if page > 1 else "")
            html = _get(url)
            if not html:
                break
            pairs = re.findall(r'href="/en/answers/(\d+)"[^>]*>([^<]+)<', html)
            if not pairs:
                # fall back to id-only links (title inside the link markup)
                ids = re.findall(r'href="/en/answers/(\d+)"', html)
                pairs = [(i, "") for i in ids]
            new_on_page = 0
            for aid, title in pairs:
                aid = int(aid)
                if aid not in seen:
                    seen[aid] = {"id": aid, "title": title.strip(), "topic": topic}
                    new_on_page += 1
            print(f"  topic {topic} page {page}: {len(pairs)} links ({new_on_page} new)")
            if new_on_page == 0:
                break
            pages = [int(p) for p in re.findall(r"page=(\d+)", html)]
            page += 1
            if pages and page > max(pages):
                break
            time.sleep(DELAY_S)
    entries = sorted(seen.values(), key=lambda e: e["id"])
    return entries


def _decode_flight(blob: str) -> str:
    r"""Unescape the concatenated RSC push-payload strings.

    Only \uXXXX / \n escapes are processed; the surrounding text is UTF-8
    and must pass through byte-identical (unicode_escape would mangle it).
    """
    blob = blob.replace('\\"', '"').replace("\\\\", "\\")

    def _sub(m: re.Match) -> str:
        if m.group(1):
            return chr(int(m.group(1), 16))
        return "\n" if m.group(0) == "\\n" else m.group(2) or m.group(0)

    return re.sub(r"\\u([0-9a-fA-F]{4})|(\\n)|(\\\\)", _sub, blob)


def parse_answer(html: str) -> dict | None:
    """Extract the Q/A from the Next.js RSC flight payload.

    Two body shapes exist:
      - long answers: "body":"$56" references a text chunk "56:T...,<html>"
      - short answers: "body":"<inline html>"
    Rich metadata lives on the answer object: reference id, summary, title, url.
    """
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', html)
    if not chunks:
        return None
    blob = _decode_flight("".join(chunks))

    out: dict = {}
    m = re.search(r'\{"type":"answer","reference":(\d+),"summary":"((?:[^"\\]|\\.)*)"', blob)
    if m:
        out["id"] = int(m.group(1))
        out["summary"] = "" if "$" in m.group(2) else m.group(2)
    mt = re.search(r'"summary":"(?:[^"\\]|\\.)*?","title":"((?:[^"\\]|\\.)*)"', blob)
    if mt:
        out["title"] = mt.group(1)

    body = None
    mb = re.search(r'"body":"(\$[0-9a-z]+)"', blob)
    if mb:
        ref = mb.group(1)[1:]
        mc = re.search(rf"(?:^|\n){ref}:T[0-9a-z]*,(.*?)(?:\n\d+:|$)", blob, re.S)
        if mc:
            body = mc.group(1)
    if body is None:
        m2 = re.search(r'"body":"((?:[^"\\]|\\.)+)"', blob)
        if m2 and not m2.group(1).startswith("$"):
            body = m2.group(1)
    out["body_html"] = body or ""

    if not out.get("title") and not out.get("body_html"):
        return None
    return out


def harvest_answers(entries: list[dict], resume: bool = True) -> None:
    """Stage 2: fetch + parse each answer page into raw JSON."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, entry in enumerate(entries, 1):
        path = OUT_DIR / f"{entry['id']}.json"
        if resume and path.exists():
            done += 1
            continue
        html = _get(f"{BASE}/answers/{entry['id']}")
        if not html:
            print(f"  [{i}/{len(entries)}] {entry['id']}: fetch failed, skipped")
            continue
        parsed = parse_answer(html)
        if parsed is None:
            print(f"  [{i}/{len(entries)}] {entry['id']}: no RSC payload, skipped")
            continue
        record = {
            "id": entry["id"],
            "url": f"{BASE}/answers/{entry['id']}",
            "title": parsed.get("title") or entry.get("title", ""),
            "summary": parsed.get("summary", ""),
            "body_html": parsed["body_html"],
            "topic": entry.get("topic", ""),
            "scholar": "Muhammad Saalih al-Munajjid",
            "harvested_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        done += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(entries)}] harvested ({done} ok)")
        time.sleep(DELAY_S)
    print(f"answers harvested: {done}/{len(entries)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("stage 1: harvesting topic index…")
    entries = harvest_index()
    print(f"index: {len(entries)} unique answers")
    INDEX_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    print("stage 2: harvesting answer pages…")
    harvest_answers(entries)
    print(f"done. index at {INDEX_PATH}")


if __name__ == "__main__":
    main()
