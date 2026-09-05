from __future__ import annotations

import json

import pytest

from agent.companion import ratings as R


@pytest.fixture()
def ratings_file(tmp_path, monkeypatch):
    path = tmp_path / "ratings.jsonl"
    monkeypatch.setattr(R, "RATINGS_PATH", path)
    return path


class TestRatingStore:
    def test_record_and_read(self, ratings_file):
        R.record_rating("s1", 1, "up", answer_excerpt="good",
                        pipeline={"intent": "loneliness"})
        R.record_rating("s1", 2, "down", answer_excerpt="bad",
                        pipeline={"intent": "dua_request"})
        recs = R.read_ratings()
        assert len(recs) == 2
        assert recs[0]["rating"] == "up" and recs[0]["turn"] == 1
        assert recs[1]["rating"] == "down"

    def test_invalid_rating_rejected(self, ratings_file):
        from agent.companion.ratings import RatingError

        with pytest.raises(RatingError):
            R.record_rating("s1", 1, "meh")

    def test_rerate_supersedes(self, ratings_file):
        """Thumb up then down on the same turn: analysis reads the LATEST."""
        R.record_rating("s1", 1, "up")
        R.record_rating("s1", 1, "down")
        latest = R.latest_ratings()
        assert latest[("s1", 1)]["rating"] == "down"
        a = R.ratings_analysis()
        assert a["down"] == 1 and a["up"] == 0

    def test_analysis_buckets(self, ratings_file):
        R.record_rating("s1", 1, "down", pipeline={"intent": "dua_request",
                                                   "mode": "qa",
                                                   "evidence_status": "insufficient_evidence"})
        R.record_rating("s2", 1, "down", pipeline={"intent": "dua_request",
                                                   "mode": "qa",
                                                   "evidence_status": "partially_answerable"})
        R.record_rating("s3", 1, "up", pipeline={"intent": "loneliness",
                                                  "mode": "companion",
                                                  "evidence_status": ""})
        a = R.ratings_analysis()
        assert a["total_rated_turns"] == 3
        assert a["down"] == 2
        assert a["down_by_intent"].get("dua_request") == 2
        assert a["down_by_evidence_status"].get("insufficient_evidence") == 1
        assert len(a["down_turns"]) == 2

    def test_pipeline_snapshot_from_chat_log(self, tmp_path, monkeypatch):
        """enrich_from_chat_log builds the self-contained pipeline record
        from the logged turn."""

        # write a fake chat turn
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        rec = {
            "ts": "2026-01-01T00:00:00+00:00", "session_id": "sx", "turn": 3,
            "user": {"text": "q"},
            "companion": {"text": "a", "mode": "qa", "intent": "dua_request",
                          "emotion": None, "risk": "low", "route": "rag"},
            "citations": ["hadith:x:1"],
            "evidence": {"status": "answerable", "sufficiency": 0.8,
                         "claim_verdicts": [{"verdict": "supports",
                                             "claim_type": "attribution"}]},
            "latency_s": 5.0, "sensitive": False, "schema": 1,
        }
        (log_dir / "sx-20260101.jsonl").write_text(json.dumps(rec) + "\n")
        monkeypatch.setattr("agent.companion.logging.LOG_DIR", log_dir)

        snap = R.enrich_from_chat_log("sx", 3, log_dir=log_dir)
        assert snap["intent"] == "dua_request"
        assert snap["citations"] == ["hadith:x:1"]
        assert snap["evidence_status"] == "answerable"
        assert snap["claim_verdicts"][0]["verdict"] == "supports"

    def test_empty_ratings(self, ratings_file):
        a = R.ratings_analysis()
        assert a["total_rated_turns"] == 0
        assert a["down_rate"] == 0.0
