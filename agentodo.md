# AGENTIC TODO — Local-First Sunni Islamic AI Knowledge & Content Platform

> **Purpose:** Build a local-first, source-grounded Islamic study and content assistant around small open models, with Ling-3.0-tiny as the primary agent/reasoning model and Gemma 4 E4B QAT as the efficiency/mobile-oriented model.
>
> **Hard requirement:** The knowledge corpus, retrieval layer, evaluation set, prompts, and generated religious answers MUST be restricted to the project's approved **Sunni Islamic source registry**. Do not ingest, retrieve, cite, summarize, or mix material from non-approved traditions. Do not silently merge conflicting doctrinal sources.

---
## Projects references
  Use https://github.com/ben-kodbiz/Aya as foundation
## 0. Project Principles

### Non-negotiable

- [ ] Sunni-only knowledge policy.
- [ ] Source-first, not model-memory-first.
- [ ] Every substantive religious claim should have provenance.
- [ ] Never invent Qur'an references, hadith, scholars, book titles, page numbers, grades, or quotations.
- [ ] If a source cannot be verified, say so.
- [ ] Clearly distinguish:
  - Qur'an
  - authentic Sunnah / hadith
  - tafsir
  - fiqh
  - aqidah
  - seerah
  - scholarly interpretation
  - contemporary commentary
- [ ] Do not present an AI answer as a fatwa or as the authority of a scholar.
- [ ] When a matter has legitimate Sunni scholarly disagreement, represent the disagreement rather than fabricating a single universal position.
- [ ] Do not use model pretraining as evidence.
- [ ] Do not expose hidden chain-of-thought. Store only concise provenance, decisions, confidence, and validation results.
- [ ] Local-first and open-source wherever practical.
- [ ] No mandatory cloud API.
- [ ] All model backends must be replaceable behind an OpenAI-compatible interface.

---

# 1. Target Architecture

```text
                         ┌──────────────────────┐
                         │       PWA / Web       │
                         │ Android future client │
                         └───────────┬──────────┘
                                     │
                              OpenAI-compatible API
                                     │
                         ┌───────────▼───────────┐
                         │    Agent Orchestrator  │
                         │ routing / policy / ACL │
                         └───────────┬───────────┘
                                     │
             ┌───────────────────────┼────────────────────────┐
             │                       │                        │
             ▼                       ▼                        ▼
      ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
      │ Intent/Entity│       │ Retrieval    │       │ User Memory  │
      │ Extraction   │       │ Orchestrator │       │ / Study State│
      └──────┬───────┘       └──────┬───────┘       └──────┬───────┘
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    ▼
                         ┌────────────────────┐
                         │ Sunni Knowledge    │
                         │ Graph + FTS + Vector│
                         └─────────┬──────────┘
                                   │
                    ┌──────────────┼───────────────┐
                    ▼              ▼               ▼
                 Qur'an         Hadith          Tafsir
                    │              │               │
                    └──────────────┼───────────────┘
                                   ▼
                         ┌────────────────────┐
                         │ Source Validator   │
                         │ citation checker   │
                         └─────────┬──────────┘
                                   │
                      ┌────────────┼────────────┐
                      ▼                         ▼
               Ling-3.0-tiny              Gemma 4 E4B QAT
               primary agent              fast/mobile path
               + reasoning                + multimodal path
                      │                         │
                      └────────────┬────────────┘
                                   ▼
                         ┌────────────────────┐
                         │ Response Validator │
                         │ safety + provenance│
                         └─────────┬──────────┘
                                   ▼
                              User response
```

---

# 2. Model Strategy

## 2.1 Ling-3.0-tiny

Primary model for:

- agent orchestration
- reasoning
- tool selection
- structured output
- complex RAG synthesis
- content planning
- difficult questions
- entity/relationship reasoning
- source comparison

Current model characteristics to account for:

- 7.9B total parameters
- approximately 1.3B activated parameters/token
- sparse MoE
- 128 routed experts
- 8 routed + 1 shared expert/token
- hybrid KDA/MLA architecture
- supports thinking and non-thinking modes
- INT4 weights available
- MIT licensed

Reference:
https://huggingface.co/inclusionAI/Ling-3.0-tiny

### Tasks

- [ ] Benchmark Ling locally.
- [ ] Prefer INT4 for constrained desktop deployment.
- [ ] Test llama.cpp compatibility.
- [ ] Test a vLLM/SGLang-compatible backend separately if required.
- [ ] Expose an OpenAI-compatible `/v1` API.
- [ ] Implement `thinking=false` for routine requests.
- [ ] Implement `thinking=true` only for complex reasoning.
- [ ] Measure VRAM, RAM, prompt processing, generation speed, and long-context behavior on the target PC.
- [ ] Do not assume active parameter count equals VRAM requirement.

---

## 2.2 Gemma 4 E4B QAT

Use as the second model family and especially as the future edge/mobile candidate.

Recommended initial checkpoint:

`google/gemma-4-E4B-it-qat-q4_0-gguf`

Google provides Gemma 4 E4B QAT variants, including mobile-oriented checkpoints. Official QAT models use 4-bit weights with 16-bit activations; Google documents QAT as a way to reduce memory while preserving quality close to higher precision. Gemma 4 supports multimodal input and long context, and E4B is explicitly positioned for compute/memory efficiency.

Reference:
https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf

Mobile-oriented checkpoint:
`google/gemma-4-E4B-it-qat-mobile-transformers`

References:
https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-transformers
https://ai.google.dev/gemma/docs/core

### Tasks

- [ ] Benchmark Gemma E4B QAT against Ling.
- [ ] Test GGUF with llama.cpp.
- [ ] Test mobile-transformers checkpoint for Android feasibility.
- [ ] Test image/OCR capability where useful.
- [ ] Test Malay.
- [ ] Test Arabic.
- [ ] Test English.
- [ ] Test Qur'an/Arabic text preservation.
- [ ] Test function/tool calling.
- [ ] Test thinking mode.
- [ ] Test memory footprint on target Android devices later.

---

# 3. Model Routing Policy

Do NOT permanently hard-code one model.

Implement:

```yaml
routing:
  simple_chat: gemma_e4b
  classification: gemma_e4b
  entity_extraction: gemma_e4b
  simple_rag: gemma_e4b
  complex_rag: ling_tiny
  source_comparison: ling_tiny
  difficult_reasoning: ling_tiny
  content_generation: ling_tiny
  mobile_default: gemma_e4b
```

Allow runtime configuration.

Example:

```text
simple question
    ↓
Gemma E4B
    ↓
fast answer

complex question
    ↓
Ling
    ↓
retrieve → reason → validate → answer
```

---

# 4. Hardware Profiles

Design for three profiles.

## Profile A — Developer workstation

Target example:

- NVIDIA RTX 3060 12GB
- 64GB system RAM
- modern Ryzen CPU
- Linux
- local SSD/NVMe

Use:

- Ling INT4
- Gemma E4B QAT
- local embeddings
- SQLite
- vector index
- local inference server

### Tasks

- [ ] Establish baseline VRAM/RAM measurements.
- [ ] Keep at least 8–12GB system RAM free for the host.
- [ ] Avoid unnecessarily huge context windows.
- [ ] Start at 4K–8K context.
- [ ] Increase context only after measuring KV-cache cost.
- [ ] Do not run multiple large models simultaneously unless benchmarks justify it.

---

## Profile B — Ordinary PC / laptop

Target:

- 16–32GB RAM
- integrated or modest GPU

Use:

- Gemma E4B QAT
- smaller embedding model
- SQLite/FTS
- optional vector index
- CPU or partial GPU offload

---

## Profile C — Android

Target:

- 8GB RAM minimum for serious experimentation
- 12GB+ preferred
- Snapdragon/MediaTek/Exynos devices vary substantially

Use:

- Gemma E4B QAT mobile checkpoint first
- reduced context
- local SQLite/FTS
- small embedding model or server-assisted retrieval
- optional downloadable Sunni knowledge packs

Do NOT require Ling on Android initially.

---

# 5. Sunni Knowledge Policy

## 5.1 Source Registry

Create:

```text
knowledge/
├── registry/
│   ├── approved_sources.yaml
│   ├── excluded_sources.yaml
│   └── source_policy.md
├── quran/
├── hadith/
├── tafsir/
├── aqidah/
├── fiqh/
├── seerah/
├── scholars/
└── topics/
```

Every source gets metadata:

```yaml
id:
title:
author:
type:
language:
tradition:
school:
publisher:
edition:
license:
url:
verification_status:
allowed: true
```

## 5.2 Approved-source gate

Before ingestion:

```text
source
  ↓
metadata extraction
  ↓
Sunni registry check
  ↓
license check
  ↓
quality check
  ↓
ALLOW / REJECT / MANUAL REVIEW
```

Never ingest first and filter later.

## 5.3 Excluded-source gate

Maintain an explicit exclusion registry for sources that are outside the project's defined Sunni corpus.

The application must not:

- retrieve excluded material
- use excluded material as citation
- merge excluded material into embeddings
- use excluded material for fine-tuning
- use excluded material for evaluation
- silently blend doctrinal positions

Avoid derogatory descriptions of excluded groups; this is a corpus-control mechanism, not a harassment feature.

---

# 6. Source Hierarchy

Implement source classes.

```text
TIER 0 — Qur'an
TIER 1 — verified/authenticated Sunnah material
TIER 2 — approved classical Sunni tafsir
TIER 3 — approved Sunni aqidah / fiqh / seerah works
TIER 4 — approved contemporary Sunni scholarship
TIER 5 — approved educational material
```

Important:

- Tier ordering is for provenance and retrieval policy.
- Do not automatically treat every Tier 1–5 item as equally authoritative.
- Hadith authenticity metadata must be preserved.
- Scholarly disagreement must be represented explicitly.

---

# 7. Knowledge Ingestion Pipeline

```text
PDF / HTML / EPUB / TXT / structured data
                │
                ▼
          Source Registry
                │
                ▼
          License checker
                │
                ▼
        Document extraction
                │
                ▼
       OCR if required
                │
                ▼
        Arabic/text cleanup
                │
                ▼
      structural segmentation
                │
                ▼
       metadata extraction
                │
                ▼
       entity extraction
                │
                ▼
       citation extraction
                │
                ▼
        quality validation
                │
                ▼
       chunk generation
                │
                ▼
      embeddings + FTS index
                │
                ▼
       knowledge graph links
                │
                ▼
       versioned knowledge DB
```

### Tasks

- [ ] Build ingestion CLI.
- [ ] Preserve original source files separately from normalized text.
- [ ] Preserve Arabic exactly where possible.
- [ ] Store page/chapter/section information.
- [ ] Store hadith collection/book/chapter/number metadata where available.
- [ ] Store Qur'an surah/ayah identifiers.
- [ ] Store tafsir author/work/volume/page metadata where available.
- [ ] Generate stable document IDs.
- [ ] Generate stable chunk IDs.
- [ ] Hash source documents.
- [ ] Make ingestion deterministic and repeatable.

---

# 8. Retrieval Architecture

Do NOT use vector search alone.

Use hybrid retrieval:

```text
                    Query
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       SQLite FTS   BM25       Vector
          │           │           │
          └───────────┼───────────┘
                      ▼
                Reciprocal Rank
                  / reranking
                      │
                      ▼
                source filtering
                      │
                      ▼
                provenance pack
```

### Retrieval filters

Mandatory:

```text
allowed == true
tradition == SUNNI
verification_status != rejected
```

Optional filters:

- Qur'an only
- Hadith only
- Tafsir only
- specific scholar
- specific book
- language
- topic
- date/edition

---

# 9. Knowledge Graph

Start simple with SQLite tables.

Entities:

```text
surah
ayah
hadith
collection
scholar
book
chapter
topic
person
place
event
concept
```

Relationships:

```text
ayah -> explained_by -> tafsir
ayah -> related_to -> topic
hadith -> belongs_to -> collection
hadith -> narrated_by -> narrator
book -> authored_by -> scholar
scholar -> wrote -> book
topic -> related_to -> ayah
topic -> related_to -> hadith
```

Do not introduce Neo4j unless SQLite becomes a demonstrated bottleneck.

---

# 10. Entity Extraction

Use a small model or deterministic rules before calling the main model.

Extract:

```text
Quran references
Hadith references
Scholar names
Book names
People
Places
Topics
Arabic terms
Fiqh concepts
Aqidah concepts
```

Candidate:

- GLiNER or another small NER model
- Ling/Gemma fallback
- regex/rule-based Qur'an and hadith reference parsers

### Tasks

- [ ] Benchmark deterministic extraction vs NER.
- [ ] Create Sunni Islamic entity labels.
- [ ] Normalize Arabic and English/Malay aliases.
- [ ] Create canonical entity IDs.

---

# 11. Agent Tool Layer

Expose tools with strict schemas.

Initial tools:

```text
search_quran()
search_hadith()
search_tafsir()
search_scholar()
search_topic()
get_ayah()
get_hadith()
get_source_metadata()
get_related_sources()
verify_hadith_claim()
verify_quran_reference()
retrieve_provenance()
save_study_note()
get_study_history()
```

Later:

```text
search_lecture()
transcribe_audio()
summarize_lecture()
generate_content_draft()
```

Every tool must enforce the Sunni source filter.

---

# 12. Response Pipeline

```text
User question
      │
      ▼
Intent classification
      │
      ▼
Entity extraction
      │
      ▼
Source-policy check
      │
      ▼
Query planning
      │
      ▼
Hybrid retrieval
      │
      ▼
Evidence/provenance pack
      │
      ▼
Ling/Gemma
      │
      ▼
Claim extraction
      │
      ▼
Citation verification
      │
      ▼
Unsupported-claim detector
      │
      ▼
Response composer
      │
      ▼
User
```

If a claim cannot be supported:

```text
DO NOT GUESS.
```

Return:

> "I could not verify this from the approved source corpus."

---

# 13. Hadith Verification

Build a dedicated validator.

Input:

```text
"The Prophet ﷺ said ..."
```

Pipeline:

```text
claim
 ↓
hadith candidate retrieval
 ↓
collection matching
 ↓
wording comparison
 ↓
source metadata
 ↓
grading/authentication metadata
 ↓
validator
 ↓
verified / uncertain / unsupported
```

Never allow the LLM to manufacture hadith grading.

---

# 14. Qur'an Verification

Implement deterministic references.

Accept:

```text
2:255
Al-Baqarah 255
Ayat al-Kursi
```

Normalize to:

```json
{
  "surah": 2,
  "ayah": 255
}
```

Then resolve against the Qur'an dataset.

Never allow the model to invent verse text.

---

# 15. Memory

Separate memory types.

## User profile

```text
language
UI preferences
study preferences
```

## Study memory

```text
topics studied
ayahs studied
hadith studied
notes
questions
bookmarks
```

## Conversation memory

Short-lived conversational context.

## Knowledge memory

Canonical source data.

Never mix user memory with authoritative knowledge.

---

# 16. Fine-Tuning Strategy

## Rule

**RAG is the primary knowledge mechanism. Fine-tuning is the behavior mechanism.**

Do not fine-tune the model simply to memorize the Islamic corpus.

### Fine-tuning targets

Good:

- source-aware answering
- citation formatting
- structured JSON
- tool calling
- Malay response style
- Arabic preservation
- clarification behavior
- uncertainty handling
- refusal to fabricate
- routing
- content-generation style

Bad:

- stuffing entire books into SFT
- teaching unsupported religious claims
- using unverified social-media posts as ground truth
- mixing non-approved sources into training data

---

# 17. Fine-Tuning Pipeline

```text
Approved Sunni sources
        │
        ▼
   source validation
        │
        ▼
      extraction
        │
        ▼
    human curation
        │
        ▼
 instruction dataset
        │
        ├── QA
        ├── citation
        ├── classification
        ├── tool calling
        ├── uncertainty
        └── content generation
        │
        ▼
       train/val/test
        │
        ▼
       QLoRA/SFT
        │
        ▼
      evaluation
        │
        ▼
   regression tests
        │
        ▼
      release LoRA
```

### Tools

Prefer:

- Transformers
- PEFT
- TRL
- Unsloth where compatible
- llama.cpp for deployment where supported

Do not assume every QAT/mobile checkpoint is a good fine-tuning base.

For QAT models, keep a separate full/base model training track and deployment quantization track unless compatibility is demonstrated.

---

# 18. Dataset Format

Use JSONL.

Example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Answer only from approved Sunni sources. Cite evidence. Never invent religious references."
    },
    {
      "role": "user",
      "content": "Explain the meaning of this ayah."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "sources": [
    {
      "source_id": "quran:2:255",
      "type": "quran"
    },
    {
      "source_id": "tafsir:approved-work:...",
      "type": "tafsir"
    }
  ]
}
```

---

# 19. Evaluation Suite

Build a dedicated benchmark before fine-tuning.

Minimum initial set:

```text
100 Qur'an questions
100 Hadith questions
100 Tafsir questions
50 Aqidah questions
50 Fiqh questions
50 Seerah questions
50 Malay questions
50 Arabic-reference questions
50 hallucination traps
50 source/citation tests
```

Test:

```text
Ling-3.0-tiny
Gemma 4 E4B QAT
Qwen 3.5 4B
Qwen 3.5 9B
```

Use identical retrieval and evidence packs.

Metrics:

```text
source accuracy
citation accuracy
hallucination rate
unsupported claim rate
Qur'an reference accuracy
hadith attribution accuracy
Sunni corpus compliance
Malay quality
Arabic preservation
tool-call accuracy
latency
VRAM
RAM
tokens/sec
```

---

# 20. Adversarial Evaluation

Create deliberate traps.

Examples:

```text
fake hadith
wrong ayah number
wrong scholar attribution
fabricated book
mixed-source question
ambiguous narration
conflicting scholarly views
missing source
poor OCR
Arabic spelling variation
Malay translation ambiguity
```

Expected behavior:

```text
UNKNOWN
UNVERIFIED
NEEDS SOURCE
SCHOLARLY DIFFERENCE
```

Never force an answer.

---

# 21. Social Media / Content Pipeline

This should reuse the same verified knowledge system.

```text
Approved source
      │
      ▼
Topic extraction
      │
      ▼
Research / retrieval
      │
      ▼
Evidence pack
      │
      ▼
Ling content planner
      │
      ▼
Draft
      │
      ▼
Citation validator
      │
      ▼
Human review
      │
      ▼
Publish-ready content
```

Outputs:

```text
YouTube Shorts script
TikTok script
Instagram carousel
Facebook post
X/thread-style post
blog article
PWA study card
```

Never automatically publish religious claims without a review gate.

---

# 22. Existing YouTube/Tadabbur Pipeline Integration

Future integration:

```text
YouTube
   │
   ▼
yt-dlp
   │
   ▼
audio/video
   │
   ▼
Whisper
   │
   ▼
transcript
   │
   ▼
speaker/topic segmentation
   │
   ▼
Ling/Gemma tagging
   │
   ▼
Sunni knowledge retrieval
   │
   ▼
citation/claim analysis
   │
   ▼
knowledge archive
```

Store the original lecture separately from canonical religious sources.

A lecture is **secondary material**, not automatically authoritative.

---

# 23. Android Roadmap

## Phase A — Server-backed Android PWA

First release:

```text
Android browser
      │
      ▼
PWA
      │
      ▼
local/web API
      │
      ▼
desktop/server inference
```

Features:

- Qur'an search
- hadith search
- tafsir search
- citations
- bookmarks
- study history
- offline UI shell

This gets a usable Android product quickly.

---

## Phase B — Offline knowledge packs

Package:

```text
SQLite
+
FTS index
+
compressed metadata
```

Downloadable by topic/language.

Example:

```text
Quran Core
Hadith Core
Tafsir Pack
Malay Pack
Arabic Pack
```

No model required for basic search.

This is important because **deterministic offline search is much cheaper than offline LLM inference**.

---

## Phase C — On-device Gemma

Use:

`google/gemma-4-E4B-it-qat-mobile-transformers`

or the appropriate officially supported mobile runtime/checkpoint.

Tasks:

- benchmark memory
- benchmark latency
- benchmark battery impact
- benchmark thermals
- test Android NNAPI/GPU/vendor acceleration where available
- test 4K/8K context
- implement model download separately from APK
- allow users to choose whether to install the model

---

## Phase D — Hybrid mobile agent

```text
                 Android
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Local search          Gemma
          │                   │
          └─────────┬─────────┘
                    ▼
              Local answer
```

For difficult questions:

```text
Android
   │
   ▼
optional trusted server
   │
   ▼
Ling
   │
   ▼
verified response
```

The app must clearly indicate when processing leaves the device.

---

# 24. Privacy

Default:

```text
local-first
no telemetry by default
no conversation upload by default
no advertising SDK
no unnecessary analytics
```

If optional remote inference exists:

- [ ] explicit user consent
- [ ] clear privacy notice
- [ ] never silently upload conversation history
- [ ] allow local-only mode
- [ ] make server URL configurable

---

# 25. Suggested Repository Layout

```text
sunni-agent/
├── README.md
├── AGENTS.md
├── AGENTICTODO.md
├── LICENSE
│
├── apps/
│   ├── web/
│   └── android/
│
├── agent/
│   ├── router/
│   ├── planner/
│   ├── tools/
│   ├── memory/
│   ├── validators/
│   └── prompts/
│
├── models/
│   ├── ling/
│   ├── gemma/
│   ├── embeddings/
│   └── configs/
│
├── knowledge/
│   ├── registry/
│   ├── quran/
│   ├── hadith/
│   ├── tafsir/
│   ├── aqidah/
│   ├── fiqh/
│   ├── seerah/
│   └── processed/
│
├── ingestion/
├── retrieval/
├── graph/
├── evaluation/
├── finetune/
├── content/
├── scripts/
├── configs/
└── tests/
```

---

# 26. Development Phases

## Phase 0 — Architecture

- [ ] Create repository.
- [ ] Write source policy.
- [ ] Create approved Sunni source registry.
- [ ] Create excluded-source registry.
- [ ] Create model abstraction.
- [ ] Create configuration system.
- [ ] Create test framework.

**Do not build UI yet.**

---

## Phase 1 — Model Lab

- [ ] Download/test Ling INT4.
- [ ] Download/test Gemma E4B QAT GGUF.
- [ ] Benchmark on RTX 3060 12GB.
- [ ] Benchmark CPU fallback.
- [ ] Compare latency and VRAM.
- [ ] Compare Malay.
- [ ] Compare Arabic.
- [ ] Compare structured JSON.
- [ ] Compare tool calling.
- [ ] Compare reasoning.
- [ ] Record results in `evaluation/model-benchmark.md`.

**Exit condition:** one model selected as default for each task class.

---

## Phase 2 — Sunni Knowledge Core

- [ ] Implement source registry.
- [ ] Implement ingestion.
- [ ] Implement Qur'an dataset.
- [ ] Implement approved hadith dataset.
- [ ] Implement approved tafsir.
- [ ] Implement metadata.
- [ ] Implement SQLite schema.
- [ ] Implement FTS.
- [ ] Implement citation IDs.

**Exit condition:** every retrieved passage has stable provenance.

---

## Phase 3 — RAG

- [ ] Implement embeddings.
- [ ] Implement vector search.
- [ ] Implement hybrid retrieval.
- [ ] Implement reranking.
- [ ] Implement source filtering.
- [ ] Implement evidence packs.
- [ ] Implement citation validator.

**Exit condition:** no answer is generated without a traceable evidence pack for source-dependent questions.

---

## Phase 4 — Agent

- [ ] Intent router.
- [ ] Entity extraction.
- [ ] Tool schemas.
- [ ] Tool execution.
- [ ] Memory.
- [ ] Planner.
- [ ] Validator.
- [ ] Retry logic.
- [ ] Model routing.

**Exit condition:** agent can complete multi-step source-grounded tasks.

---

## Phase 5 — Evaluation

- [ ] Build benchmark.
- [ ] Build hallucination tests.
- [ ] Build citation tests.
- [ ] Build Sunni-source compliance tests.
- [ ] Run all models.
- [ ] Produce regression report.

**Exit condition:** measurable baseline before fine-tuning.

---

## Phase 6 — Fine-Tuning

Only start after Phase 5.

- [ ] Curate approved Sunni instruction dataset.
- [ ] Create train/validation/test splits.
- [ ] QLoRA/SFT experiments.
- [ ] Compare base vs tuned.
- [ ] Check catastrophic forgetting.
- [ ] Check citation behavior.
- [ ] Check hallucination rate.
- [ ] Check multilingual behavior.
- [ ] Release adapter only if it improves objective metrics.

**Exit condition:** fine-tuning demonstrably improves behavior without reducing source accuracy.

---

## Phase 7 — Content Engine

- [ ] YouTube transcript ingestion.
- [ ] Topic extraction.
- [ ] Evidence retrieval.
- [ ] Short-form script generation.
- [ ] Citation checking.
- [ ] Human approval queue.
- [ ] Export to Markdown/JSON/CSV.
- [ ] Integrate with existing Tadabbur workflow.

---

## Phase 8 — PWA

- [ ] Mobile-first UI.
- [ ] Search.
- [ ] Chat.
- [ ] Source cards.
- [ ] Citation display.
- [ ] Bookmarks.
- [ ] Study history.
- [ ] Offline shell.
- [ ] Installable PWA.

---

## Phase 9 — Android

- [ ] Package PWA or build native shell.
- [ ] Offline SQLite.
- [ ] Offline source packs.
- [ ] Gemma mobile benchmark.
- [ ] On-device inference prototype.
- [ ] Model download manager.
- [ ] Privacy/local-only mode.
- [ ] Hybrid remote inference option.
- [ ] Battery/thermal testing.

---

# 27. Definition of Done

The project is NOT considered successful merely because:

```text
"the chatbot talks nicely"
```

Minimum success criteria:

- [ ] Uses only approved Sunni corpus.
- [ ] Rejects unapproved sources at ingestion.
- [ ] Rejects unapproved sources at retrieval.
- [ ] Religious claims have provenance.
- [ ] Qur'an references are deterministic.
- [ ] Hadith claims have source/authentication metadata where available.
- [ ] Model admits uncertainty.
- [ ] No fabricated citations.
- [ ] RAG beats raw model knowledge on the project's benchmark.
- [ ] Fine-tuning is optional, measured, and reversible.
- [ ] Ling and Gemma can be swapped without rewriting the agent.
- [ ] Runs locally on the target workstation.
- [ ] Has a realistic Android path.
- [ ] No mandatory cloud dependency.
- [ ] Human review remains required for publish-ready religious content.

---

# 28. First Agent Task

The coding agent MUST NOT attempt the whole project in one pass.

Start with:

```text
1. Create repository skeleton.
2. Create AGENTICTODO.md.
3. Create source-policy implementation.
4. Create model abstraction.
5. Create Ling/Gemma benchmark harness.
6. Run model benchmarks.
7. Produce benchmark report.
8. STOP and report results.
```

Only proceed to knowledge ingestion after the model benchmark and source-policy tests pass.

---

# 29. Golden Rule

> **The LLM is not the source of Islam.**
>
> The approved Sunni knowledge corpus is the source layer.
>
> The retrieval system finds evidence.
>
> The validator checks evidence.
>
> The model reasons over that evidence and communicates it.
>
> The application preserves provenance.
>
> The user remains able to inspect the source.

That separation is the foundation of the project.
