"""Hybrid retrieval orchestrator (agentodo.md §8).

FTS/BM25 leg + reference-exact leg fused with reciprocal-rank fusion, then the
MANDATORY source-policy filter applied to every result. Vector search joins
here later (embeddings via LM Studio); the interface already accepts it.

Do NOT use vector search alone. Never retrieve excluded material.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.policy.source_policy import SourcePolicy, SourceRegistry
from agent.tools.quran_refs import extract_references, normalize_reference
from ingestion.quran_ingest import QuranStore

SURAH_ONLY_RE = re.compile(
    r"\bsurah\s+(\d{1,3})\b|\bchapter\s+(\d{1,3})\b", re.IGNORECASE
)

# Which leg's passage object to keep when a citation appears in several legs.
# Lower = richer/more authoritative: reference > translation/hadith > tafsir > fts.
_LEG_PRIORITY = {"reference": 0, "translation": 1, "hadith": 1, "tafsir": 2,
                 "vector": 2, "fts": 2, "fusion": 3}


@dataclass
class RetrievedPassage:
    citation_id: str
    surah: int
    ayah: int
    arabic: str
    source_id: str
    tier: int
    leg: str  # which retrieval leg found it ("fts", "reference", "translation", "hadith", "fusion")
    score: float
    translation: str = ""  # English rendering (interpretation, NOT Qur'an text)
    collection: str = ""  # hadith collections only
    hadithnumber: int | None = None
    grades: list | None = None  # hadith grading metadata (§6), kept verbatim
    scholar: str = ""  # classic tafsir author (chunked corpora)


class RetrievalOrchestrator:
    def __init__(self, store: QuranStore, policy: SourcePolicy | None = None,
                 hadith_store=None, tafsir_store=None, tafsir_en_store=None,
                 vector_store=None):
        self.store = store
        self.policy = policy or SourcePolicy(SourceRegistry.load())
        self.hadith_store = hadith_store
        self.tafsir_store = tafsir_store
        self.tafsir_en_store = tafsir_en_store
        self.vector_store = vector_store

    def _has_translation(self, lang: str = "en") -> bool:
        try:
            return self.store.translation_count(lang) > 0
        except Exception:
            return False

    def _is_latin(self, query: str) -> bool:
        letters = [c for c in query if c.isalpha()]
        if not letters:
            return False
        latin = sum(1 for c in letters if ord(c) < 0x0590)
        return latin / len(letters) > 0.7

    def search(self, query: str, limit: int = 8,
               concept_expansions: list[str] | None = None,
               semantic_only: bool = False) -> list[RetrievedPassage]:
        """Query -> evidence candidates. Source filter applied always.

        concept_expansions: emotional-register expansions from the intent
        router — plain-vector retrieval cannot bridge 'I am lonely' to 'indeed
        I am near', explicit concepts do (they run as extra vector queries).
        semantic_only: for emotional statements, lexical matching is pure
        noise ('I am lonely' matches 'I am only a warner'); rely on the
        vector leg + expansions alone.
        """
        # Emotional statements lexically false-match and even semantically
        # rank literal-narration hits above comfort material; the concept
        # expansions are the real queries. Raw query participates only when
        # no expansions were provided.
        if concept_expansions:
            vector_queries = list(concept_expansions)
        else:
            vector_queries = [query]
        # Leg 1: exact Qur'an references in the query, deterministic (§14).
        # The paired TIER 2 tafsir for the same ayah is seeded alongside: tafsir
        # explains the ayah (§6), so a referenced-ayah lookup naturally carries
        # its tafsir.
        ref_hits: list[RetrievedPassage] = []
        seeded_tafsir: set[tuple[int, int]] = set()
        for ref in self._references_in(query):
            row = self.store.get_ayah(ref["surah"], ref["ayah"])
            if row:
                ref_hits.append(self._to_passage(row, "reference", 1.0))
                seeded_tafsir.add((ref["surah"], ref["ayah"]))
        # Leg 2: FTS/BM25 — Arabic corpus for Arabic queries. Skipped for
        # semantic_only (emotional) queries: lexical matches are noise there.
        fts_hits: list[RetrievedPassage] = []
        if not semantic_only and (not self._is_latin(query) or not self._has_translation()):
            fts_hits = [
                self._to_passage(h, "fts", h["rank"]) for h in self.store.search_fts(query, limit=limit)
            ]
        # Leg 3: translation FTS for Latin-script queries (en by default);
        # skipped for semantic_only queries (see leg 2 note).
        translation_hits: list[RetrievedPassage] = []
        if not semantic_only and self._is_latin(query) and self._has_translation():
            legs_lang = ["en"]
            if self._has_translation("id"):
                legs_lang.append("id")  # Indonesian queries hit the Kemenag translation
            for lang in legs_lang:
                translation_hits.extend(
                    self._to_passage(h, "translation", h["rank"])
                    for h in self.store.search_translation_fts(query, lang=lang, limit=limit)
                )
        # Leg 4: hadith corpus (TIER 1, §6) — Arabic or English by script.
        # Skipped for semantic_only (emotional) queries: lexical matches are noise.
        hadith_hits: list[RetrievedPassage] = []
        if not semantic_only and self.hadith_store is not None and self._hadith_corpus_present():
            for h in self.hadith_store.search_fts(query, limit=limit):
                hadith_hits.append(
                    RetrievedPassage(
                        citation_id=h["citation_id"], surah=0, ayah=0,
                        arabic=h["arabic"], source_id=h["source_id"], tier=1,
                        leg="hadith", score=h["rank"],
                        translation=h.get("english") or "",
                        collection=h["source_id"],
                        hadithnumber=h["hadithnumber"],
                        grades=h.get("grades"),
                    )
                )
        # Leg 5: tafsir corpora (TIER 2, §6) — interpretation, ranks below Qur'an/hadith.
        # (a) Kemenag per-ayah tafsir; reference-seeded first so RRF ranks it
        #     alongside the ayah it explains (§6 tiering: tafsir explains the ayah).
        tafsir_hits: list[RetrievedPassage] = []
        if not semantic_only and self.tafsir_store is not None and self._tafsir_corpus_present():
            for (surah, ayah) in sorted(seeded_tafsir):
                t = self.tafsir_store.get_tafsir(surah, ayah)
                if t:
                    tafsir_hits.append(
                        RetrievedPassage(
                            citation_id=t["citation_id"], surah=surah, ayah=ayah,
                            arabic="", source_id=t["source_id"], tier=2,
                            leg="tafsir", score=1.0, translation=t["tafsir"],
                        )
                    )
            for t in self.tafsir_store.search_fts(query, limit=limit):
                if t["citation_id"] in {h.citation_id for h in tafsir_hits}:
                    continue
                tafsir_hits.append(
                    RetrievedPassage(
                        citation_id=t["citation_id"], surah=t["surah"], ayah=t["ayah"],
                        arabic="", source_id=t["source_id"], tier=2,
                        leg="tafsir", score=t["rank"],
                        translation=t["tafsir"],
                    )
                )
        # (b) classic English tafsirs (chunked): ayah-anchored for referenced
        #     ayahs (all three scholars' commentary on that ayah) + FTS for
        #     topical queries. Chunk citations keep their stable chunk_id.
        if self.tafsir_en_store is not None and self._tafsir_en_corpus_present():
            en_tafsir_hits: list[RetrievedPassage] = []
            for (surah, ayah) in sorted(seeded_tafsir):
                for ch in self.tafsir_en_store.get_for_ayah(surah, ayah, limit=4):
                    en_tafsir_hits.append(
                        RetrievedPassage(
                            citation_id=f"tafsir-en:{ch['chunk_id']}", surah=surah, ayah=ayah,
                            arabic="", source_id=ch["source_id"], tier=2,
                            leg="tafsir", score=1.0,
                            translation=ch["text"],
                            scholar=ch["scholar"],
                        )
                    )
            for ch in self.tafsir_en_store.search_fts(query, limit=limit):
                if f"tafsir-en:{ch['chunk_id']}" in {h.citation_id for h in en_tafsir_hits}:
                    continue
                en_tafsir_hits.append(
                    RetrievedPassage(
                        citation_id=f"tafsir-en:{ch['chunk_id']}",
                        surah=ch["surah"], ayah=ch["ayah_start"],
                        arabic="", source_id=ch["source_id"], tier=2,
                        leg="tafsir", score=ch["rank"],
                        translation=ch["text"],
                        scholar=ch["scholar"],
                    )
                )
            tafsir_hits.extend(en_tafsir_hits)
        # Leg 6: vector/semantic (§8 — never vector search ALONE, but vector
        # search bridges vocabulary gaps: 'I am lonely' -> 2:186 'I am near').
        # Hits resolve to full passages via the corpus stores so tier and
        # provenance stay intact; then the same §8 filter applies.
        vector_hits: list[RetrievedPassage] = []
        if self.vector_store is not None and self.vector_store.size:
            seen_cids = {
                p.citation_id
                for leg in (ref_hits, fts_hits, translation_hits, hadith_hits, tafsir_hits)
                for p in leg
            }
            if semantic_only:
                # emotional mode: take the highest-cosine hits across ALL
                # vector queries (cosine is comparable across queries in the
                # same space; positional RRF would favor whichever expansion
                # ran first instead of the genuinely closest passages).
                candidates: dict[str, dict] = {}
                for vq in vector_queries:
                    for hit in self.vector_store.search(vq, top_k=limit):
                        cid = hit["citation_id"]
                        if cid not in candidates or hit["score"] > candidates[cid]["score"]:
                            candidates[cid] = hit
                ranked = sorted(candidates.values(), key=lambda h: -h["score"])
                for hit in ranked:
                    if hit["citation_id"] in seen_cids:
                        continue
                    passage = self._resolve_vector_hit(hit)
                    if passage:
                        vector_hits.append(passage)
                vector_hits = vector_hits[:limit]
            else:
                per_query = max(4, limit // 2)
                for vq in vector_queries:
                    for hit in self.vector_store.search(vq, top_k=per_query):
                        if hit["citation_id"] in seen_cids:
                            continue
                        passage = self._resolve_vector_hit(hit)
                        if passage:
                            vector_hits.append(passage)
                            seen_cids.add(passage.citation_id)
                vector_hits = vector_hits[: limit * 2]

        # Fusion: RRF over all legs; exact-reference hits always survive
        fused = self._rrf(
            [ref_hits, fts_hits, translation_hits, hadith_hits, tafsir_hits, vector_hits], k=60
        )
        # Mandatory source filter (§8) — every passage must pass
        filtered = [p for p in fused if self._filter_passes(p)]
        # Tier balancing is for INFORMATIONAL queries; for emotional/semantic
        # queries rank purity matters more than tier quotas.
        if semantic_only:
            return filtered[:limit]
        return self._tier_balanced(filtered, limit)

    def _hadith_row_to_passage(self, row: dict, leg: str, score: float) -> RetrievedPassage:
        return RetrievedPassage(
            citation_id=row["citation_id"], surah=0, ayah=0,
            arabic=row["arabic"], source_id=row["source_id"], tier=1,
            leg=leg, score=score, translation=row.get("english") or "",
            collection=row["source_id"], hadithnumber=row.get("hadithnumber"),
            grades=row.get("grades") or None,
        )

    def _resolve_vector_hit(self, hit: dict) -> RetrievedPassage | None:
        """Map a vector hit's citation_id back to a full passage with provenance."""
        cid = hit["citation_id"]
        try:
            if cid.startswith("quran:"):
                _, s, a = cid.split(":")
                row = self.store.get_ayah(int(s), int(a), lang="en")
                if row:
                    return self._to_passage(row, "vector", hit["score"])
            elif cid.startswith("hadith:"):
                _, source, num = cid.split(":")
                if self.hadith_store:
                    row = self.hadith_store.get_hadith(source, int(num))
                    if row:
                        return self._hadith_row_to_passage(row, "vector", hit["score"])
            elif cid.startswith("tafsir-en:"):
                chunk_id = cid.split(":", 1)[1]
                if self.tafsir_en_store:
                    row = self.tafsir_en_store.get_chunk(chunk_id)
                    if row:
                        return RetrievedPassage(
                            citation_id=cid, surah=row["surah"], ayah=row["ayah_start"],
                            arabic="", source_id=row["source_id"], tier=2,
                            leg="vector", score=hit["score"],
                            translation=row["text"], scholar=row["scholar"],
                        )
            elif cid.startswith("tafsir:"):
                _, source, s, a = cid.split(":")
                if self.tafsir_store:
                    row = self.tafsir_store.get_tafsir(int(s), int(a), source_id=source)
                    if row:
                        return RetrievedPassage(
                            citation_id=cid, surah=int(s), ayah=int(a),
                            arabic="", source_id=source, tier=2,
                            leg="vector", score=hit["score"], translation=row["tafsir"],
                        )
        except (ValueError, KeyError):
            return None
        return None

    @staticmethod
    def _tier_balanced(passages: list[RetrievedPassage], limit: int) -> list[RetrievedPassage]:
        """RRF ranking with tier + source diversity: at least 1 TIER 0, 1 TIER 1,
        and 2 TIER 2 passages from DIFFERENT tafsir sources when available (§6:
        tiering shapes retrieval policy; a pure-FTS list buries tafsir, and a
        single tafsir work monopolizing tier 2 hides scholarly perspectives)."""
        if len(passages) <= limit:
            return passages
        out: list[RetrievedPassage] = []
        chosen_t2_sources: set[str] = set()
        for tier, min_count in ((0, 1), (1, 1)):
            already = sum(1 for p in out if p.tier == tier)
            for p in passages:
                if len(out) >= limit:
                    break
                if p.tier == tier and already < min_count and p not in out:
                    out.append(p)
                    already += 1
        # tier 2: prefer distinct sources (kemenag, sadi, ibn kathir, qurtubi)
        t2_count = 0
        for p in passages:
            if len(out) >= limit or t2_count >= 2:
                break
            if p.tier == 2 and p not in out:
                if p.source_id not in chosen_t2_sources or t2_count == 0:
                    out.append(p)
                    chosen_t2_sources.add(p.source_id)
                    t2_count += 1
        for p in passages:  # fill remaining slots by RRF rank
            if len(out) >= limit:
                break
            if p not in out:
                out.append(p)
        return sorted(out, key=lambda p: -p.score)

    def _hadith_corpus_present(self) -> bool:
        try:
            return self.hadith_store.hadith_count() > 0
        except Exception:
            return False

    def _tafsir_corpus_present(self) -> bool:
        try:
            return self.tafsir_store.tafsir_count() > 0
        except Exception:
            return False

    def _tafsir_en_corpus_present(self) -> bool:
        try:
            return self.tafsir_en_store.chunk_count() > 0
        except Exception:
            return False

    def _references_in(self, query: str) -> list[dict]:
        refs = []
        try:
            refs.append(normalize_reference(query))
        except ValueError:
            pass
        seen = {(r["surah"], r["ayah"]) for r in refs}
        for r in extract_references(query):
            key = (r["surah"], r["ayah"])
            if key not in seen:
                refs.append(r)
                seen.add(key)
        # "Surah 112" alone -> whole-surah expansion (first ayahs as evidence)
        for m in SURAH_ONLY_RE.finditer(query):
            surah = int(m.group(1) or m.group(2))
            if 1 <= surah <= 114 and not any(k[0] == surah for k in seen):
                for key in self._surah_seed(surah):
                    refs.append({"surah": surah, "ayah": key})
                    seen.add((surah, key))
        return refs[:10]

    def _surah_seed(self, surah: int, count: int = 3) -> list[int]:
        """Seed ayahs for a surah-only query: first `count` ayahs, capped."""
        total = self.store.surah_ayah_count(surah)
        return list(range(1, min(count, total) + 1))

    def _to_passage(self, row: dict, leg: str, score: float) -> RetrievedPassage:
        return RetrievedPassage(
            citation_id=row["citation_id"],
            surah=row["surah"],
            ayah=row["ayah"],
            arabic=row["arabic"],
            translation=row.get("translation", ""),
            source_id=row["source_id"],
            tier=0,  # everything in this corpus is TIER 0 Qur'an for now
            leg=leg,
            score=score,
        )

    def _filter_passes(self, passage: RetrievedPassage) -> bool:
        if self.policy.must_not_retrieve(passage.source_id):
            return False
        try:
            record = self.policy.registry.get(passage.source_id)
        except KeyError:
            return False
        return self.policy.retrieval_filter(record)

    @staticmethod
    def _rrf(legs: list[list[RetrievedPassage]], k: int = 60) -> list[RetrievedPassage]:
        """Reciprocal rank fusion across legs, preserving the best passage."""
        scores: dict[str, tuple[float, RetrievedPassage]] = {}
        for leg in legs:
            for rank, passage in enumerate(leg):
                contribution = 1.0 / (k + rank + 1)
                if passage.citation_id in scores:
                    score, best = scores[passage.citation_id]
                    # prefer the richest passage: reference > translation > fts
                    better_rank = _LEG_PRIORITY[passage.leg] < _LEG_PRIORITY[best.leg]
                    same_rank_richer = (
                        _LEG_PRIORITY[passage.leg] == _LEG_PRIORITY[best.leg]
                        and passage.translation
                        and not best.translation
                    )
                    if better_rank or same_rank_richer:
                        best = passage
                    scores[passage.citation_id] = (score + contribution, best)
                else:
                    scores[passage.citation_id] = (contribution, passage)
        fused = []
        for citation_id, (score, passage) in scores.items():
            fused.append(
                RetrievedPassage(
                    citation_id=citation_id, surah=passage.surah, ayah=passage.ayah,
                    arabic=passage.arabic, translation=passage.translation,
                    source_id=passage.source_id, tier=passage.tier,
                    leg=passage.leg, score=score,
                    collection=passage.collection,
                    hadithnumber=passage.hadithnumber,
                    grades=passage.grades,
                    scholar=passage.scholar,
                )
            )
        return sorted(fused, key=lambda p: -p.score)
