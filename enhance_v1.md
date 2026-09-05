# Ilman — Enhancement V1

## Modular Architecture, Evidence Lifecycle & Claim-Safety Upgrade

**Target branch:** `feature03`
**Enhancement:** `enhance_v1`
**Status:** Implementation specification
**Priority:** Architecture / Safety / Extensibility
**Principle:** Strengthen the existing Ilman architecture without replacing working components.

---

# 1. Objective

Enhance Ilman from the current validated RAG + companion architecture into a modular, evidence-first Islamic AI system.

The enhancement MUST preserve the existing principles:

1. The LLM is not an Islamic authority.
2. Retrieval does not automatically make a claim valid.
3. Every religious claim must be evaluated against evidence.
4. Unsupported claims must not reach the user.
5. Evidence must have an explicit authority level and role.
6. Companion conversations must not automatically become religious Q&A.
7. Private conversation data must not silently become training data.
8. Models must remain replaceable.
9. Core validation MUST NOT be bypassable by modules/plugins.
10. Prefer local/free/open-source components.

The enhancement introduces:

```text
Evidence Lifecycle
Claim Dependency Graph
Source Authority Matrix
Knowledge Modules
Capability Modules
Memory Provenance
Severity-Weighted Evaluation
Model Roles
Privacy / Retention Controls
```

---

# 2. Target Architecture

The target architecture is:

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │ SAFETY / POLICY │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ INTENT + STATE  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  MODULE ROUTER  │
                  └────────┬────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   KNOWLEDGE MODULES  CAPABILITY MODULES  COMPANION
          │                │                │
          │                │                │
     Quran/Hadith       Dua/Study        Emotion
     Tafsir/Fiqh        Reflection       Memory
     Seerah             Content
          │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                    QUERY PLANNER
                           │
                           ▼
                    HYBRID RETRIEVAL
                           │
                           ▼
                 EVIDENCE QUARANTINE
                           │
                           ▼
                   EVIDENCE PACK
                           │
                           ▼
                    MODEL GENERATION
                           │
                           ▼
                  CLAIM EXTRACTION
                           │
                           ▼
              CLAIM DEPENDENCY GRAPH
                           │
                           ▼
                EVIDENCE VALIDATION
                           │
                           ▼
              SOURCE AUTHORITY CHECK
                           │
                           ▼
             LANGUAGE STRENGTH VALIDATOR
                           │
                           ▼
                       REPAIR
                           │
                           ▼
                    REVALIDATION
                           │
                           ▼
                     FINAL ANSWER
```

No module may bypass:

```text
Safety
Policy
Evidence
Claim Validation
Authority Validation
Language Gate
Revalidation
```

---

# 3. Implementation Rules

## 3.1 Do NOT rewrite the working system

Before implementation:

* inspect the existing architecture;
* identify existing interfaces;
* reuse existing validators;
* reuse existing evidence structures;
* reuse existing evaluation harness;
* avoid unnecessary dependencies;
* avoid changing public APIs unless required;
* maintain backward compatibility where practical.

The agent MUST NOT perform a large rewrite merely to make the architecture "cleaner."

---

# 4. Enhancement P0 — Evidence Lifecycle

## Goal

Turn evidence handling into an explicit state machine.

Current evidence handling should be formalized into:

```text
DISCOVERED
    ↓
RETRIEVED
    ↓
FILTERED
    ↓
QUARANTINED
    ↓
ADMITTED
    ↓
USED
    ↓
VALIDATED
    ↓
FINAL
```

Rejected evidence:

```text
FILTERED
    ↓
REJECTED
```

must be terminal.

---

## 4.1 Evidence states

Implement an enum similar to:

```python
EvidenceState:
    DISCOVERED
    RETRIEVED
    FILTERED
    QUARANTINED
    ADMITTED
    USED
    VALIDATED
    REJECTED
    FINAL
```

Allowed transitions MUST be explicitly defined.

Example:

```text
RETRIEVED → FILTERED
FILTERED → QUARANTINED
QUARANTINED → ADMITTED
QUARANTINED → REJECTED
ADMITTED → USED
USED → VALIDATED
VALIDATED → FINAL
```

Invalid transitions must raise an error.

---

# 5. Immutable Evidence Pack

After quarantine/admission, create an immutable evidence package.

Suggested structure:

```python
EvidencePack(
    pack_id,
    query_id,
    created_at,
    sources,
    passages,
    retrieval_scores,
    authority_metadata,
    quarantine_results,
    checksum,
)
```

Once frozen:

```python
evidence_pack.freeze()
```

the evidence contents cannot be modified.

The LLM MUST NOT directly modify the EvidencePack.

---

## 5.1 Evidence fishing prevention

Do NOT allow:

```text
LLM
 ↓
"this evidence isn't enough"
 ↓
system automatically finds evidence supporting the generated claim
```

Instead:

```text
LLM
 ↓
unsupported claim
 ↓
query planner
 ↓
NEW RETRIEVAL REQUEST
 ↓
NEW EVIDENCE PACK
 ↓
NEW GENERATION
```

Every new retrieval operation receives a new `query_id` / evidence-pack identity.

---

# 6. Enhancement P0 — Claim Dependency Graph

## Goal

Move from independent claim validation to dependency-aware validation.

A response may contain:

```text
Claim A
   ↓
Claim B
   ↓
Claim C
```

If Claim A becomes unsupported, dependent claims must be reconsidered.

---

## 6.1 Claim model

Extend the existing claim representation where appropriate:

```python
Claim(
    id,
    text,
    type,
    confidence,
    evidence_refs,
    dependencies,
    severity,
    status,
)
```

Suggested claim types:

```text
DIRECT_FACT
QURAN_CLAIM
HADITH_CLAIM
ATTRIBUTION
TAFSIR
FIQH_RULING
INFERENCE
CAUSAL
GUARANTEE
DIAGNOSIS
PREDICTION
ADVICE
OPINION
```

---

# 7. Claim Relations

Implement explicit relationships:

```text
SUPPORTS
DEPENDS_ON
INFERRED_FROM
CONTRADICTS
QUALIFIES
REFINES
```

Example:

```text
C1: Hadith establishes X.

C2: X is recommended.

C3: Therefore doing X has benefit.

C4: X will cure a medical condition.
```

Graph:

```text
C1 ──SUPPORTS────► C2
C2 ──SUPPORTS────► C3
C3 ──DEPENDS_ON──► C4
```

If C2 fails:

```text
C2 = UNSUPPORTED

C3 = INVALIDATED
C4 = INVALIDATED
```

The validator must propagate invalidation through dependent claims.

---

# 8. Claim Severity

Add:

```text
LOW
MEDIUM
HIGH
CRITICAL
```

Suggested interpretation:

### LOW

Minor unsupported descriptive statement.

### MEDIUM

Potentially misleading interpretation.

### HIGH

Religious ruling, attribution, causal claim, or strong prescription.

### CRITICAL

Examples include:

* falsely attributing statements to Allah;
* falsely attributing statements to the Prophet ﷺ;
* declaring something definitively halal/haram without sufficient authority;
* medical guarantees presented as religious fact;
* dangerous certainty in high-risk situations.

Severity is used for evaluation and prioritization.

---

# 9. Enhancement P0 — Source Authority Matrix

Create a formal source authority layer.

Suggested structure:

```python
SourceAuthority(
    source_id,
    source_type,
    authority_level,
    permitted_claim_types,
    restrictions,
    provenance,
)
```

---

## 9.1 Example authority model

```text
QURAN
  ↓
Quranic claims

SAHIH HADITH
  ↓
Prophetic attribution

TAFSIR
  ↓
Tafsir interpretation

APPROVED FIQH SOURCE
  ↓
Fiqh/ruling claims

SCHOLAR / SECONDARY SOURCE
  ↓
Secondary explanation

LECTURE / VIDEO
  ↓
Secondary material

LLM
  ↓
NO RELIGIOUS AUTHORITY
```

The LLM can formulate language but cannot promote itself to an authoritative source.

---

# 10. Claim Authority Requirements

Create a policy mapping:

```yaml
QURAN_CLAIM:
  required_sources:
    - quran

PROPHETIC_ATTRIBUTION:
  required_sources:
    - hadith

TAFSIR:
  required_sources:
    - tafsir

FIQH_RULING:
  required_sources:
    - approved_fiqh

MEDICAL_CLAIM:
  required_sources:
    - appropriate_medical_source
```

A source may be semantically relevant but still fail the authority requirement.

Example:

```text
Claim:
"Islam prohibits X."

Source:
random lecture

Result:
AUTHORITY_FAIL
```

This is different from:

```text
ENTAILMENT_FAIL
```

Both validations must exist.

---

# 11. Two-Dimensional Evidence Validation

Every claim should be evaluated on at least two axes:

```text
                 Evidence
                    │
             ┌──────┴──────┐
             │             │
        Entailment      Authority
             │             │
       Does source     Is source
       support claim?  qualified?
```

Possible results:

```text
SUPPORTED
PARTIAL_SUPPORT
ENTAILMENT_FAIL
AUTHORITY_FAIL
CONTRADICTED
INSUFFICIENT_EVIDENCE
```

Do not collapse these into one boolean.

---

# 12. Enhancement P1 — Knowledge Modules

Introduce a formal interface:

```python
class KnowledgeProvider:
    name
    version

    def search(query, filters):
        ...

    def get_document(document_id):
        ...

    def get_metadata(document_id):
        ...

    def validate_source(document_id):
        ...
```

Initial modules:

```text
quran
hadith
tafsir
fiqh
seerah
```

Modules should be independently enableable.

---

# 13. Knowledge Module Rules

Knowledge modules:

* provide evidence;
* provide metadata;
* provide provenance;
* do not generate final answers;
* do not bypass claim validation;
* do not bypass source authority policy.

Example:

```text
Hadith Module
     ↓
Hadith evidence
     ↓
Core Evidence Layer
     ↓
Claim Validator
```

Not:

```text
Hadith Module
     ↓
LLM
     ↓
User
```

---

# 14. Enhancement P1 — Capability Modules

Capability modules perform tasks.

Examples:

```text
dua
study
reflection
tadabbur
content
bookmark
search
```

Interface:

```python
class Capability:
    name
    version

    def can_handle(context):
        ...

    def plan(context):
        ...

    def execute(plan):
        ...
```

Capabilities MUST return structured results to the core.

They must not directly bypass validation.

---

# 15. Knowledge vs Capability

Maintain strict separation:

```text
KNOWLEDGE
=========
Quran
Hadith
Tafsir
Fiqh
Seerah

CAPABILITY
==========
Dua Finder
Study Assistant
Reflection
Tadabbur
Content Generator
```

Knowledge answers:

> "What evidence do we have?"

Capability answers:

> "What operation should we perform?"

This separation is mandatory.

---

# 16. Enhancement P1 — Module Router

Create a lightweight module router.

Example:

```text
USER REQUEST
     ↓
INTENT
     ↓
MODULE ROUTER
     │
     ├── Quran
     ├── Hadith
     ├── Tafsir
     ├── Fiqh
     ├── Dua
     ├── Study
     └── Companion
```

The router should select the minimum required modules.

Do not invoke every module for every query.

---

# 17. Companion Mode

Preserve the existing distinction between conversational/emotional requests and explicit religious questions.

Example:

```text
"I feel lonely."
```

should not automatically become:

```text
Quran retrieval
Hadith retrieval
Fiqh retrieval
```

Instead:

```text
Emotion detection
      ↓
Companion
      ↓
empathetic response
      ↓
optional gentle question
```

If the user explicitly asks:

```text
"What does Islam say about loneliness?"
```

then:

```text
Companion
      ↓
Islamic knowledge modules
      ↓
Evidence pipeline
```

---

# 18. Enhancement P1 — Memory Provenance

Every persistent memory item must contain provenance.

Suggested model:

```python
Memory(
    id,
    content,
    category,
    source,
    confidence,
    created_at,
    expires_at,
)
```

Source classes:

```text
EXPLICIT_USER
SYSTEM
IMPORTED
INFERRED
TEMPORARY
```

---

# 19. Memory Rules

### EXPLICIT_USER

May be persisted if allowed by application policy.

### SYSTEM

Application configuration.

### IMPORTED

Imported from a trusted user-provided source.

### INFERRED

Must NOT automatically become permanent personal memory.

### TEMPORARY

Session-only context.

Critical rule:

```text
INFERRED ≠ FACT
```

The system must never silently convert model inference into persistent user identity/profile information.

---

# 20. Memory Retention

Add:

```text
created_at
updated_at
expires_at
source
```

Allow:

```text
delete memory
delete session
delete all memories
```

The system should support future retention policies without requiring a database redesign.

---

# 21. Enhancement P1 — Severity-Weighted Evaluation

Existing evaluation metrics should remain.

Add:

```text
SEVERITY_WEIGHTED_UNSUPPORTED_ESCAPE
```

Concept:

```text
weighted_escape =
    escaped unsupported severity
    /
    generated unsupported severity
```

This prevents a minor unsupported statement from being treated as equivalent to a serious religious attribution.

---

# 22. Evaluation Categories

Expand the test suite gradually.

Target:

```text
Normal QA
Hadith
Quran
Tafsir
Fiqh
Dua
Seerah
Companion
Inference
Attribution
Adversarial
Multilingual
```

Initial target:

```text
500+ tests
```

Long-term:

```text
1000+
```

Do NOT block current development waiting for 1000 tests.

Build the framework first.

---

# 23. Required Safety Metrics

Track at minimum:

```text
RELIGIOUS_FALSE_SUPPORT_RATE
UNSUPPORTED_CLAIM_ESCAPE_RATE
AUTHORITY_FAILURE_RATE
CLAIM_INVALIDATION_RATE
REPAIR_SUCCESS_RATE
REVALIDATION_FAILURE_RATE
SEVERITY_WEIGHTED_ESCAPE
```

The system should report these independently.

---

# 24. Adversarial Test Cases

Add tests specifically designed to fool the system.

Examples:

```text
False attribution
Source mismatch
Citation laundering
Partial citation
Inference inflation
Causal inflation
Guarantee inflation
Authority confusion
Hadith/source mixing
Tafsir treated as Quran
Lecture treated as revelation
Model opinion treated as ruling
```

Example:

```text
Evidence:
"X is mentioned in a lecture."

Generated:
"The Prophet ﷺ commanded X."

Expected:
FAIL
```

---

# 25. Enhancement P1 — Model Roles

Separate models by role rather than treating all models identically.

Example:

```yaml
models:

  router:
    model: ling-tiny

  generation:
    model: qwen

  validation:
    model: gemma

  embedding:
    model: embedding-model
```

The exact models remain configurable.

No architecture code should hard-code Qwen, Gemma, Ling, or another provider.

---

# 26. Model Role Rules

Each model should have a declared role:

```text
ROUTER
GENERATOR
VALIDATOR
EMBEDDER
CLASSIFIER
```

A generator should not automatically be trusted as a validator merely because it is capable of reasoning.

Model replacement must require configuration changes rather than architectural rewrites.

---

# 27. Model Escalation

Future-compatible architecture:

```text
Easy request
    ↓
small model

Moderate request
    ↓
medium model

Complex request
    ↓
larger model
```

Do not implement complicated routing until evaluation data proves it is useful.

First establish:

```text
correctness
latency
VRAM
CPU
failure rate
```

---

# 28. Fine-Tuning Gate

DO NOT begin fine-tuning merely because a larger model or LoRA is available.

Required order:

```text
Architecture
     ↓
RAG
     ↓
Validation
     ↓
Evaluation
     ↓
Failure analysis
     ↓
Prompt / architecture improvements
     ↓
Baseline benchmark
     ↓
QLoRA experiment
```

Fine-tuning must be measurable against the baseline.

---

# 29. Training Data Isolation

Conversation logs must NOT automatically become training data.

Required pipeline:

```text
Conversation
     ↓
Candidate dataset
     ↓
Sanitization
     ↓
Human review
     ↓
Quality filtering
     ↓
Approval
     ↓
Training dataset
```

No automatic:

```text
chat logs → training
```

---

# 30. Enhancement P2 — Privacy / Retention

Structured logging should remain useful for debugging while minimizing stored sensitive information.

Add configurable:

```text
LOG_RETENTION_DAYS
SESSION_RETENTION_DAYS
MEMORY_RETENTION_DAYS
ENABLE_CONTENT_LOGGING
```

Future support:

```text
session deletion
memory deletion
full data purge
```

Do not log chain-of-thought.

Store:

```text
claims
evidence IDs
validation results
scores
pipeline states
errors
latency
```

rather than private internal reasoning traces.

---

# 31. Sensitive Conversation Handling

Sensitive conversations should have a distinct handling path.

At minimum:

```text
normal
sensitive
restricted
```

The system should minimize persistent logging for restricted conversations.

Do not introduce additional personal-data collection merely for analytics.

---

# 32. External Web Sources

Do NOT make unrestricted internet search part of the core religious reasoning pipeline.

Preferred:

```text
Approved Corpus
      ↓
Retrieval
      ↓
Validation
      ↓
Answer
```

If external sources are eventually supported:

```text
EXTERNAL_UNTRUSTED
```

must be an explicit source category.

External information must not automatically become authoritative Islamic evidence.

---

# 33. Source Promotion

Never allow:

```text
web page
 ↓
LLM
 ↓
"authoritative Islamic source"
```

Instead:

```text
external source
      ↓
UNTRUSTED
      ↓
classification
      ↓
policy review
      ↓
approved source registry
      ↓
authoritative source
```

Promotion must be explicit.

---

# 34. Plugin Security Boundary

Every module must operate behind a capability boundary.

Modules may:

```text
request retrieval
request model generation
return structured results
request validation
```

Modules may NOT:

```text
disable validation
change authority policy
mark unsupported claims as supported
inject arbitrary evidence
modify frozen EvidencePack
bypass safety policy
```

---

# 35. Suggested Directory Structure

Adapt this to the existing repository rather than blindly replacing it:

```text
ilman/
├── agent/
│   ├── router/
│   ├── planner/
│   ├── companion/
│   ├── memory/
│   └── runtime/
│
├── core/
│   ├── policy/
│   ├── evidence/
│   │   ├── lifecycle.py
│   │   ├── pack.py
│   │   └── authority.py
│   │
│   ├── claims/
│   │   ├── model.py
│   │   ├── graph.py
│   │   └── propagation.py
│   │
│   └── validation/
│
├── modules/
│   ├── knowledge/
│   │   ├── quran/
│   │   ├── hadith/
│   │   ├── tafsir/
│   │   ├── fiqh/
│   │   └── seerah/
│   │
│   └── capabilities/
│       ├── dua/
│       ├── study/
│       ├── reflection/
│       ├── tadabbur/
│       └── content/
│
├── models/
│   ├── router.py
│   ├── generator.py
│   ├── validator.py
│   └── embeddings.py
│
├── evaluation/
│   ├── datasets/
│   ├── metrics/
│   ├── adversarial/
│   └── reports/
│
└── tests/
    ├── claims/
    ├── evidence/
    ├── authority/
    ├── modules/
    ├── companion/
    └── adversarial/
```

Do not create duplicate functionality if equivalent components already exist.

---

# 36. Backward Compatibility

Existing functionality must continue working.

Before modifications:

```text
run existing test suite
```

Record baseline.

After each P0/P1 feature:

```text
run unit tests
run integration tests
run evaluation harness
```

No enhancement is considered complete if it silently breaks existing behavior.

---

# 37. Implementation Phases

## Phase 1 — Evidence Lifecycle

Implement:

* EvidenceState
* transition validation
* immutable EvidencePack
* rejection terminal state
* evidence provenance

Acceptance:

```text
invalid state transitions fail
frozen evidence cannot be modified
rejected evidence cannot return to pipeline
```

---

## Phase 2 — Claim Graph

Implement:

* Claim dependencies
* Claim relations
* dependency propagation
* invalidation
* severity

Acceptance:

```text
unsupported parent claim
        ↓
dependent claims invalidated
```

---

## Phase 3 — Authority Layer

Implement:

* SourceAuthority
* source registry
* claim/source compatibility
* authority failure result

Acceptance:

```text
relevant but unauthorized source
        ↓
AUTHORITY_FAIL
```

---

## Phase 4 — Modular Architecture

Implement:

* KnowledgeProvider
* Capability
* ModuleRouter

Start with adapters around existing functionality.

Do not immediately migrate everything.

---

## Phase 5 — Memory Provenance

Implement:

* source
* confidence
* expiry
* memory classes
* deletion

Ensure inferred information does not automatically become persistent memory.

---

## Phase 6 — Evaluation Upgrade

Implement:

* severity-weighted metrics
* authority metrics
* adversarial tests
* claim dependency tests

Maintain the existing benchmark as a regression suite.

---

## Phase 7 — Model Roles

Implement configurable:

```text
router
generator
validator
embedding
```

Do not optimize model routing prematurely.

---

## Phase 8 — Privacy / Logging

Implement:

* retention
* deletion
* restricted logging
* training-data isolation

---

# 38. Testing Requirements

Every new subsystem requires unit tests.

Minimum:

```text
Evidence lifecycle tests
Evidence immutability tests
Claim graph tests
Dependency propagation tests
Authority tests
Module isolation tests
Memory provenance tests
Model-role tests
Privacy tests
```

---

# 39. Critical Integration Tests

Test the complete pipeline:

```text
USER
 ↓
INTENT
 ↓
MODULE ROUTER
 ↓
RETRIEVAL
 ↓
QUARANTINE
 ↓
EVIDENCE PACK
 ↓
GENERATION
 ↓
CLAIM EXTRACTION
 ↓
CLAIM GRAPH
 ↓
ENTAILMENT
 ↓
AUTHORITY
 ↓
LANGUAGE GATE
 ↓
REPAIR
 ↓
REVALIDATION
 ↓
ANSWER
```

At least one test must verify that every stage is actually executed.

---

# 40. Failure Injection

Add tests where individual stages fail.

Examples:

```text
retrieval returns irrelevant evidence
retrieval returns contradictory evidence
authority registry rejects source
claim has no evidence
claim depends on failed claim
validator fails
repair introduces new unsupported claim
model produces excessive certainty
```

Expected behavior must be deterministic.

---

# 41. Repair Safety

Repair is not automatically trusted.

After repair:

```text
NEW RESPONSE
     ↓
CLAIM EXTRACTION
     ↓
FULL VALIDATION
```

Do NOT only validate the modified sentence.

A repair may introduce new claims elsewhere.

---

# 42. Language Strength Gate

Continue enforcing the distinction between:

```text
evidence says
evidence suggests
scholars differ
may indicate
possibly
```

and stronger statements:

```text
Islam commands
Allah guarantees
The Prophet ﷺ said
This will cure
This is definitely haram
```

The stronger the language, the stronger the evidence requirement.

---

# 43. No Authority Leakage

The following must never happen:

```text
LLM confidence
      ↓
religious certainty
```

Confidence from the model is not evidence.

Likewise:

```text
retrieval score
      ↓
truth score
```

is invalid.

Retrieval relevance and truth/authority are separate concepts.

---

# 44. Performance Requirements

The enhancement should remain usable on local hardware.

Do not introduce a large number of additional model calls without measurement.

Track:

```text
latency
VRAM
RAM
CPU
tokens
model calls
retrieval calls
validation calls
```

Prefer deterministic code for:

```text
state transitions
authority policy
severity
dependency propagation
```

Use LLMs where semantic judgment is actually required.

---

# 45. Deterministic vs LLM Responsibilities

Prefer deterministic code for:

```text
state machine
authority rules
claim graph traversal
severity mapping
retention
permissions
module boundaries
configuration
```

Use LLMs for:

```text
intent classification
claim extraction
semantic entailment
language analysis
repair suggestions
companion language generation
```

This minimizes unnecessary model dependence.

---

# 46. Observability

Every pipeline execution should have a trace ID.

Example:

```text
trace_id
query_id
evidence_pack_id
claim_graph_id
model_request_id
validation_id
```

This allows:

```text
user query
   ↓
evidence
   ↓
claim
   ↓
validation
```

to be reconstructed during debugging.

---

# 47. Evaluation Report

Generate machine-readable evaluation output.

Example:

```json
{
  "run_id": "...",
  "total_cases": 500,
  "passed": 492,
  "failed": 8,
  "religious_false_support_rate": 0.0,
  "unsupported_claim_escape_rate": 0.0,
  "authority_failure_rate": 0.012,
  "repair_success_rate": 0.94,
  "severity_weighted_escape": 0.001
}
```

Exact schema may follow the existing evaluation system.

---

# 48. Definition of Done

Enhancement V1 is complete when:

### Evidence

* [ ] Evidence lifecycle exists.
* [ ] Invalid transitions are rejected.
* [ ] EvidencePack can be frozen.
* [ ] Rejected evidence is terminal.
* [ ] Evidence provenance is retained.

### Claims

* [ ] Claims have stable IDs.
* [ ] Claims support dependencies.
* [ ] Claim graph exists.
* [ ] Dependency invalidation works.
* [ ] Claim severity exists.

### Authority

* [ ] Source authority registry exists.
* [ ] Claim/source compatibility is checked.
* [ ] Entailment and authority are separate.
* [ ] Unauthorized sources cannot establish protected claim types.

### Modules

* [ ] KnowledgeProvider interface exists.
* [ ] Capability interface exists.
* [ ] Module router exists.
* [ ] Modules cannot bypass core validation.

### Companion

* [ ] Companion mode remains distinct from religious QA.
* [ ] Explicit Islamic questions invoke knowledge modules.
* [ ] Existing companion behavior remains functional.

### Memory

* [ ] Memory provenance exists.
* [ ] Inferred memory is distinguishable from explicit memory.
* [ ] Expiry/deletion mechanisms exist.
* [ ] Inferred information does not silently become permanent memory.

### Evaluation

* [ ] Severity-weighted metric exists.
* [ ] Authority failures are measured.
* [ ] Claim dependency tests exist.
* [ ] Adversarial tests exist.
* [ ] Existing regression suite passes.

### Models

* [ ] Model roles are configurable.
* [ ] Models remain swappable.
* [ ] No model is treated as an authority.
* [ ] Fine-tuning remains gated behind evaluation.

### Privacy

* [ ] Retention configuration exists.
* [ ] Sensitive logging is minimized.
* [ ] Conversation logs are not automatically training data.
* [ ] Deletion path exists.

---

# 49. Explicit Non-Goals

Do NOT implement these as part of Enhancement V1:

```text
❌ Autonomous unrestricted web research
❌ Autonomous religious decision making
❌ Automatic source promotion
❌ Automatic training from user conversations
❌ Large-scale model fine-tuning
❌ Complex multi-agent swarm
❌ Cloud dependency
❌ Plugin marketplace
❌ Full Android application
❌ Massive database migration
❌ Rewriting working architecture
```

These may be considered later.

---

# 50. Priority Order

The implementation priority is:

```text
P0
├── Evidence Lifecycle
├── Immutable EvidencePack
├── Claim Dependency Graph
└── Source Authority Matrix

P1
├── Knowledge Modules
├── Capability Modules
├── Module Router
├── Memory Provenance
├── Severity Metrics
└── Adversarial Evaluation

P2
├── Model Roles
├── Privacy / Retention
└── Advanced Observability

P3
└── Fine-Tuning / QLoRA
```

---

# 51. Final Architectural Principle

Ilman should evolve around this invariant:

```text
             MODEL
               │
               ▼
             CLAIM
               │
        ┌──────┴──────┐
        ▼             ▼
     EVIDENCE      AUTHORITY
        │             │
        └──────┬──────┘
               ▼
            POLICY
               │
               ▼
          LANGUAGE GATE
               │
               ▼
          REVALIDATION
               │
               ▼
             USER
```

The model generates language.

The evidence establishes support.

The authority layer determines whether the evidence is qualified for the claim.

The policy layer determines what may be said.

The validator determines whether the final answer remains grounded.

**The model must never become the authority.**

---

# 52. Agent Instructions

When implementing this specification:

1. Inspect the current repository first.
2. Identify existing equivalent functionality.
3. Reuse existing code wherever possible.
4. Do not duplicate validators.
5. Do not rewrite stable components without evidence.
6. Implement one phase at a time.
7. Run tests after every phase.
8. Preserve the existing evaluation harness.
9. Record architectural changes.
10. Do not introduce unnecessary dependencies.
11. Prefer deterministic implementations for policy/state logic.
12. Keep model providers configurable.
13. Never bypass the core validation pipeline.
14. Never silently weaken an existing safety rule.
15. Do not begin QLoRA/fine-tuning until the baseline evaluation is stable.
16. If an implementation conflicts with an existing safety invariant, stop and document the conflict rather than silently changing the invariant.

---

# 53. Expected Result

After Enhancement V1, Ilman should be able to support:

```text
                 ILMAN CORE
                     │
       ┌─────────────┼─────────────┐
       │             │             │
    QURAN         HADITH        TAFSIR
       │             │             │
       ├─────────────┼─────────────┤
       │             │             │
     FIQH         SEERAH          DUA
       │             │             │
       └─────────────┼─────────────┘
                     │
              CORE VALIDATION
                     │
          ┌──────────┼──────────┐
          │          │          │
       COMPANION   STUDY     TADBABBUR
          │          │          │
          └──────────┼──────────┘
                     │
                  USER
```

while maintaining:

```text
Evidence-first
Source-aware
Claim-aware
Authority-aware
Module-safe
Model-agnostic
Privacy-aware
Evaluation-driven
Local-first
```

This becomes the foundation for future Ilman features without turning the core agent into an unmaintainable collection of special cases.
