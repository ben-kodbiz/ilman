# ILMAN — FIX_ME_V2

## Companion Harness: State → Policy → Context → Model → Validation

> **Purpose:** Evolve Ilman from a strong source-grounded Islamic QA agent into a context-aware Islamic companion while preserving all existing religious-source, provenance, safety, and local-first invariants.

> **Important:** This is an incremental enhancement. **Do not rewrite the existing retrieval/knowledge architecture.** Build the companion harness around it.

---

# 0. Current State

The project already contains:

* Intent routing
* Entity handling
* SQLite memory
* Tool-calling agent loop
* Islamic source policy
* Qur'an retrieval
* Hadith retrieval
* Tafsir retrieval
* Hybrid FTS + vector retrieval
* Evidence packs
* Citation validation
* Response validation
* Grounded evaluation
* Model abstraction
* API + PWA
* 172+ tests

The existing knowledge architecture is considered the stable foundation.

## Main remaining weakness

The current system is stronger at:

> "Answer my Islamic question."

than:

> "Understand what kind of interaction I need right now."

The next version must therefore add a **Companion Harness** around the existing agent.

---

# 1. Non-Negotiable Architecture

Do NOT turn the LLM into the source of truth.

Do NOT solve the companion problem simply by increasing model size.

Do NOT replace the current RAG architecture.

Do NOT introduce Neo4j, a distributed event bus, multi-agent swarm, or another large infrastructure dependency.

The target architecture is:

```text
                         USER
                           │
                           ▼
                    ┌─────────────┐
                    │ UNDERSTAND  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           Intent        Emotion      Entities
              │            │            │
              └────────────┼────────────┘
                           ▼
                    ┌─────────────┐
                    │ STATE ENGINE│
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ POLICY      │
                    │ ENGINE      │
                    └──────┬──────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Memory          RAG        Follow-up
          Router         Router        Policy
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    ┌─────────────┐
                    │   CONTEXT   │
                    │   BUILDER   │
                    └──────┬──────┘
                           ▼
                    Ling / Gemma
                           │
                           ▼
                    ┌─────────────┐
                    │ VALIDATION  │
                    └──────┬──────┘
                           ▼
                         USER
```

The LLM generates language.

The harness controls:

* state
* memory
* intent
* emotion
* policy
* retrieval
* context
* safety
* validation
* tool access

---

# 2. Build the Companion State Engine

Create:

```text
agent/state/
    __init__.py
    models.py
    state_machine.py
    manager.py
```

## State model

Implement a structured state object similar to:

```python
ConversationState(
    mode="companion",
    intent="emotional_support",
    emotion="loneliness",
    risk="low",
    user_goal="be_heard",
    requires_rag=False,
    requires_followup=True,
    turn_count=1,
)
```

Do not store free-form state when a structured enum/value is sufficient.

---

# 3. Conversation Modes

Implement explicit modes:

```text
qa
study
companion
reflection
dua
crisis
```

The mode is NOT necessarily permanent.

The system can transition between modes.

Example:

```text
"I feel lonely"
        ↓
companion

"Is there anything in Islam about loneliness?"
        ↓
qa / companion

"Explain the hadith you mentioned."
        ↓
study / qa

"Can you help me reflect on this?"
        ↓
reflection
```

---

# 4. State Machine

Implement:

```text
IDLE
  ↓
UNDERSTAND
  ↓
ROUTE
  ↓
RESPOND
  ↓
FOLLOW_UP
  ↓
CONTINUE
```

Special routes:

```text
ROUTE → SAFETY
ROUTE → RAG
ROUTE → MEMORY
ROUTE → COMPANION
```

Example:

```text
User
 ↓
UNDERSTAND
 ↓
intent = emotional_support
emotion = loneliness
risk = low
 ↓
ROUTE
 ↓
COMPANION
 ↓
warm response
 ↓
one follow-up
 ↓
CONTINUE
```

---

# 5. Companion Policy Engine

Create:

```text
agent/policy/
    companion_policy.py
    routing_policy.py
    response_policy.py
```

The policy engine is the most important addition in this version.

The policy engine decides:

```text
Should we retrieve?
Should we use memory?
Should we ask a question?
How many questions?
Should we provide Islamic guidance?
How much evidence?
What tone?
How long should the response be?
Should we enter safety mode?
```

The LLM should NOT make all of these decisions itself.

---

# 6. Policy Output

Create a structured policy object:

```python
ResponsePolicy(
    mode="companion",
    tone="warm",
    verbosity="short",
    requires_rag=False,
    requires_memory=True,
    requires_followup=True,
    max_followups=1,
    allow_islamic_reflection=True,
    preach=False,
)
```

The exact implementation may differ, but the decision must be machine-readable.

---

# 7. Critical Example — "I Feel Lonely"

This becomes a permanent golden regression test.

Input:

```text
I feel lonely.
```

Expected interpretation:

```json
{
  "intent": "emotional_support",
  "emotion": "loneliness",
  "risk": "low",
  "mode": "companion",
  "requires_rag": false,
  "requires_followup": true
}
```

Expected policy:

```json
{
  "tone": "warm",
  "verbosity": "short",
  "preach": false,
  "max_followups": 1
}
```

Desired behavior:

```text
Acknowledge first.

Do not immediately dump Quran verses.

Do not produce a lecture.

Do not pretend to be a human friend.

Do not claim personal feelings.

Ask one gentle follow-up.

Offer Islamic reflection only when appropriate.
```

Example behavioral target:

```text
"I'm sorry you're feeling that way. Feeling alone can be really heavy.

If you want, you can tell me a little about what's making you feel lonely today."
```

The exact wording must remain model-generated.

The test should evaluate **behavior**, not exact text matching.

---

# 8. Emotional Routing

Expand the current intent/emotion system into structured categories.

Minimum:

```text
neutral
loneliness
sadness
grief
fear
anxiety
anger
guilt
confusion
discouragement
hopelessness
gratitude
motivation
spiritual_low
relationship_problem
life_problem
```

Do not assume every emotional category requires Islamic retrieval.

For example:

```text
"I feel lonely."
→ companion response

"What does Islam say about loneliness?"
→ RAG

"Give me a dua for loneliness."
→ RAG + dua

"Which ayah talks about Allah being near?"
→ Qur'an retrieval
```

---

# 9. Separate Emotion From Intent

Do not use:

```text
emotion == intent
```

Instead:

```text
emotion = loneliness
intent = emotional_support
```

Another example:

```text
emotion = anger
intent = islamic_question
```

Another:

```text
emotion = sadness
intent = quran_question
```

This distinction is essential.

---

# 10. Memory Router

Upgrade memory into explicit categories.

Use SQLite as the primary structured store.

Minimum categories:

```text
PROFILE
PREFERENCE
FACT
STUDY
CONVERSATION
OPEN_THREAD
SUMMARY
```

Example:

```text
PREFERENCE
"I prefer Malay explanations."

STUDY
"Currently studying Surah Al-Kahf."

OPEN_THREAD
"User wants to revisit a discussion about patience."

SUMMARY
"Previous conversation discussed maintaining consistency in worship."
```

Do not put every conversation turn into permanent memory.

---

# 11. Memory Lifecycle

Every candidate memory must pass:

```text
candidate
   ↓
importance
   ↓
stability
   ↓
privacy
   ↓
deduplication
   ↓
save / reject
```

The system should NOT remember:

* every emotion
* every casual sentence
* temporary statements
* unnecessary personal details

unless explicitly useful or explicitly requested.

---

# 12. Memory Retrieval

Implement:

```text
agent/memory/
    router.py
    extractor.py
    ranking.py
    lifecycle.py
```

Memory retrieval should return only relevant memories.

Do NOT inject the entire user profile into every prompt.

Target:

```text
recent conversation
+
relevant memories
+
current state
+
relevant evidence
```

not:

```text
entire database
```

---

# 13. Recent Context vs Long-Term Memory

Maintain three separate layers:

```text
RECENT CONTEXT
last N turns

SESSION STATE
current conversation state

LONG-TERM MEMORY
stable/relevant information
```

Do not mix these.

---

# 14. Context Builder

Create:

```text
agent/context/
    models.py
    builder.py
    compression.py
```

Build a controlled `ContextPack`.

Example:

```json
{
  "mode": "companion",
  "intent": "emotional_support",
  "emotion": "loneliness",
  "risk": "low",

  "recent_context": [],

  "relevant_memory": [],

  "evidence": [],

  "policy": {
    "tone": "warm",
    "verbosity": "short",
    "followup": true,
    "preach": false
  }
}
```

Only the required context should reach the model.

---

# 15. Context Budget

Implement explicit limits.

For example:

```text
recent conversation: limited
memory: top relevant items only
evidence: only retrieved evidence
instructions: compact
```

Never assume that a large context window means the model should receive everything.

Small models benefit from carefully curated context.

---

# 16. RAG Router

Existing RAG remains authoritative.

Add a routing decision before retrieval:

```text
requires_rag = true/false
```

Examples:

```text
"I feel lonely."
→ false

"What does Islam say about loneliness?"
→ true

"Tell me about Surah 94."
→ true

"I'm having a bad day."
→ false

"Give me a Quran verse for patience."
→ true
```

---

# 17. Emotional Evidence

Do not automatically retrieve religious evidence for every emotion.

Instead:

```text
emotion
   ↓
policy
   ↓
does user want Islamic guidance?
   ↓
yes → RAG
no  → companion response
```

If appropriate, Islamic guidance can be offered after the emotional acknowledgment.

---

# 18. Follow-Up Intelligence

The companion should generally ask **one useful question**, not several.

Bad:

```text
Why are you lonely?
What happened?
How long has this been happening?
Do you have friends?
Where is your family?
```

Better:

```text
"Do you want to tell me what's been making you feel alone?"
```

Implement:

```text
max_followups = 1
```

per response by default.

Allow policy to override this when necessary.

---

# 19. Safety Router

Create an explicit safety route:

```text
agent/safety/
    classifier.py
    policy.py
    router.py
```

The safety system must be independent from the normal emotional-support path.

At minimum distinguish:

```text
low
elevated
high
```

Do not allow the normal companion policy to override safety policy.

For high-risk situations:

```text
SAFETY POLICY
    ↓
supportive response
    ↓
encourage contacting a trusted person
    ↓
encourage appropriate local emergency/crisis support
    ↓
no harmful instructions
```

Never attempt to hide or minimize safety signals.

---

# 20. Companion Personality

Define personality as behavior rather than a giant prompt.

Target characteristics:

```text
warm
calm
respectful
patient
humble
non-judgmental
Islamically grounded
concise
curious
```

Avoid:

```text
preachy
condescending
overconfident
dependency-forming
possessive
romantic
"only I understand you"
"I'll always be here instead of people"
```

The companion should support human relationships rather than replace them.

---

# 21. Islamic Grounding Rules

Preserve existing invariants.

The LLM is NOT the source of Islam.

For religious claims:

```text
retrieve
  ↓
evidence
  ↓
generate
  ↓
validate
  ↓
answer
```

Never allow:

```text
LLM memory
→ Islamic fact
```

without verification.

Never fabricate:

* ayahs
* hadith
* hadith grades
* scholars
* books
* quotations
* page numbers
* chains
* fatwas

---

# 22. Evidence Policy

Separate:

```text
EMPATHY
```

from:

```text
RELIGIOUS CLAIM
```

Empathy does not require a citation.

Religious factual claims do.

Example:

```text
"I can understand why that feels difficult."
```

No citation required.

But:

```text
"The Qur'an says..."
```

requires verified Qur'an evidence.

---

# 23. Model Adapter

Keep the current model abstraction.

Do not couple the companion engine to Ling or Gemma.

Target:

```text
ModelAdapter
    ├── Ling
    ├── Gemma
    ├── Qwen
    └── future models
```

The harness should work with all of them.

---

# 24. Model Routing

Do not optimize routing prematurely.

Start with configuration:

```yaml
models:
  companion: gemma_e4b
  classification: gemma_e4b
  simple_rag: gemma_e4b
  complex_rag: ling_tiny
  difficult_reasoning: ling_tiny
```

Allow runtime override.

The evaluation suite determines whether the routing policy is actually beneficial.

---

# 25. Response Validation

Existing religious validation must remain.

Add companion validation:

```text
ResponseValidator
    ├── source validation
    ├── citation validation
    ├── unsupported claim detection
    ├── safety validation
    ├── companion tone validation
    ├── policy compliance
    └── follow-up validation
```

A response can be factually correct but still fail companion policy.

Example:

```text
User:
"I feel lonely."

Response:
"According to Surah X, Allah..."

```

Potentially factually correct.

But:

```text
COMPANION POLICY = FAIL
```

because the response skipped emotional acknowledgment.

---

# 26. Companion Evaluation Dataset

Create:

```text
evaluation/
    companion/
        cases.jsonl
        scenarios.jsonl
        rubric.yaml
```

Include categories:

```text
loneliness
sadness
grief
anxiety
anger
guilt
discouragement
spiritual_low
normal_chat
Islamic_question
Quran_question
Hadith_question
follow_up
memory
topic_switch
safety
```

---

# 27. Multi-Turn Evaluation

Do NOT evaluate the companion only with isolated prompts.

Example:

```text
Turn 1:
"I feel lonely."

Turn 2:
"Yeah. I don't really have anyone to talk to."

Turn 3:
"I've felt this way for a while."

Turn 4:
"I don't know if Allah even hears me."
```

Evaluate:

```text
state continuity
emotion continuity
memory relevance
mode transitions
tone
follow-up quality
Islamic grounding
safety routing
```

---

# 28. Companion Score

Implement a weighted score:

```text
Context retention          20%
Emotional appropriateness  20%
Islamic grounding         15%
Hallucination              15%
Follow-up quality          10%
Safety                     10%
Conciseness                 5%
Policy compliance           5%
```

The exact weights can later change.

Track scores per model.

Example:

```text
                 Gemma    Ling
--------------------------------
Context           88       91
Emotion           91       87
Grounding         86       94
Hallucination      6%       5%
Follow-up         90       84
Safety            98       98
Latency           95       72
```

Do not select the model purely on benchmark accuracy.

---

# 29. Regression Gates

Every change must run:

```text
pytest
```

plus:

```text
grounded regression
companion regression
memory regression
policy regression
```

A model change must not silently degrade:

```text
source accuracy
citation validity
safety
companion behavior
```

---

# 30. Golden Companion Cases

Create permanent tests for:

```text
"I feel lonely."
"I'm having a bad day."
"I feel distant from Allah."
"Can you listen?"
"I don't know what to do."
"What does Islam say about loneliness?"
"Give me a dua for sadness."
"Explain this ayah."
"Who narrated this hadith?"
```

The tests should verify routing and policy rather than requiring exact generated wording.

---

# 31. Observability

Log structured metadata, NOT private conversation unnecessarily.

Useful internal telemetry:

```text
intent
emotion
risk
mode
policy decision
RAG used
memory used
model selected
validation result
latency
token counts
```

Do NOT log hidden chain-of-thought.

Do NOT expose internal reasoning.

Keep privacy-first defaults.

---

# 32. Debug Trace

Add a developer-only trace such as:

```json
{
  "intent": "emotional_support",
  "emotion": "loneliness",
  "risk": "low",
  "mode": "companion",
  "memory_hits": 1,
  "rag_used": false,
  "policy": {
    "followup": true,
    "preach": false
  },
  "model": "gemma_e4b",
  "validation": "pass"
}
```

This makes agent failures diagnosable without exposing hidden reasoning.

---

# 33. API

Extend the existing API response internally so the backend can expose structured metadata where appropriate.

Possible internal structure:

```python
AgentResult(
    response=...,
    mode=...,
    intent=...,
    citations=...,
    state=...,
)
```

Do not expose sensitive internal state by default to the public client.

---

# 34. UI

The PWA should eventually reflect conversation mode subtly.

Do NOT create a complicated UI.

Possible states:

```text
Companion
Study
Q&A
Reflection
Dua
```

The user should not need to manually select these in normal use.

Automatic routing is preferred.

Manual mode selection can be added later.

---

# 35. Do Not Overbuild

Explicitly DO NOT add during this phase:

```text
Neo4j
multi-agent swarm
distributed message bus
Kubernetes deployment
large vector database
20+ autonomous tools
autonomous background agents
giant prompts
cloud-only APIs
mandatory paid services
```

SQLite + existing retrieval + local models are sufficient.

---

# 36. Implementation Order

Implement in this exact order.

## Phase 1 — State

```text
[ ] ConversationState
[ ] mode
[ ] intent
[ ] emotion
[ ] risk
[ ] user_goal
[ ] requires_rag
[ ] requires_followup
[ ] state transitions
```

## Phase 2 — Policy

```text
[ ] CompanionPolicy
[ ] ResponsePolicy
[ ] routing rules
[ ] RAG decision
[ ] follow-up decision
[ ] preach/reflect decision
[ ] safety override
```

## Phase 3 — Context

```text
[ ] ContextPack
[ ] context builder
[ ] recent context
[ ] relevant memory
[ ] evidence
[ ] policy
[ ] context limits
```

## Phase 4 — Memory

```text
[ ] memory categories
[ ] memory router
[ ] memory extraction
[ ] importance scoring
[ ] deduplication
[ ] relevant-memory retrieval
[ ] forget/delete support
```

## Phase 5 — Companion Engine

```text
[ ] emotional routing
[ ] companion mode
[ ] follow-up logic
[ ] personality behavior
[ ] topic continuation
[ ] topic switching
```

## Phase 6 — Safety

```text
[ ] risk classifier
[ ] safety policy
[ ] safety override
[ ] regression tests
```

## Phase 7 — Validation

```text
[ ] companion validator
[ ] policy validator
[ ] unsupported companion behavior detection
[ ] safety validation
```

## Phase 8 — Evaluation

```text
[ ] companion dataset
[ ] multi-turn scenarios
[ ] scoring
[ ] model comparison
[ ] regression gates
```

## Phase 9 — UI

```text
[ ] mode indicators if useful
[ ] conversation continuity
[ ] memory controls
[ ] simple feedback mechanism
```

---

# 37. Required Tests

Minimum new test areas:

```text
tests/companion/
tests/state/
tests/policy/
tests/context/
tests/memory/
tests/safety/
```

Test:

```text
state transitions
intent/emotion separation
RAG routing
memory routing
follow-up limits
policy overrides
safety overrides
context limits
multi-turn continuity
```

---

# 38. Acceptance Test — "I Feel Lonely"

The implementation is NOT complete until:

```text
INPUT
"I feel lonely."
```

produces a route equivalent to:

```text
intent = emotional_support
emotion = loneliness
risk = low
mode = companion
rag = false
followup = true
```

and the response:

```text
acknowledges the feeling
+
is warm
+
is concise
+
does not immediately preach
+
does not fabricate religious claims
+
does not pretend to be human
+
does not encourage dependency
+
asks at most one useful question
```

---

# 39. Acceptance Test — Explicit Islamic Question

Input:

```text
"What does Islam say about loneliness?"
```

Expected:

```text
intent = islamic_question
mode = qa/companion
rag = true
evidence = required
citations = required
```

The system may begin empathetically but must transition into source-grounded Islamic guidance.

---

# 40. Acceptance Test — Simple Chat

Input:

```text
"Good morning."
```

Expected:

```text
intent = normal_chat
emotion = neutral
rag = false
```

Do not invoke the Islamic retrieval pipeline unnecessarily.

---

# 41. Acceptance Test — Memory

Conversation:

```text
User:
"I'm studying Surah Al-Kahf."

Later:

User:
"Let's continue our study."
```

Expected:

```text
memory retrieval
→ relevant study memory
→ study context
```

Do not retrieve unrelated memories.

---

# 42. Acceptance Test — Topic Switch

Conversation:

```text
User:
"Explain this hadith."

Later:

User:
"Actually, I'm feeling lonely today."
```

Expected:

```text
previous mode = study
new intent = emotional_support
new mode = companion
```

The system must not continue discussing the hadith simply because that was the previous topic.

---

# 43. Acceptance Test — Return to Study

Conversation:

```text
User:
"I'm feeling lonely."

Assistant:
[companion response]

User:
"Thanks. Now can we continue the hadith?"
```

Expected:

```text
mode = study
intent = hadith_question/study
RAG = true
```

The system must transition naturally.

---

# 44. Model Benchmark

Benchmark at minimum:

```text
Ling-3.0-tiny
Gemma 4 E4B QAT
Qwen 3.5 4B
Qwen 3.5 9B
```

Measure:

```text
companion score
grounded accuracy
hallucination
citation accuracy
follow-up quality
context retention
Malay
English
Arabic
latency
VRAM
RAM
tokens/sec
```

Do not assume the largest model wins.

---

# 45. Fine-Tuning Strategy

Do NOT fine-tune first.

First establish:

```text
Harness
 ↓
Evaluation
 ↓
Baseline
 ↓
Failure analysis
 ↓
Dataset
 ↓
QLoRA/SFT
 ↓
Benchmark again
```

Fine-tuning should target behavioral deficiencies discovered through evaluation.

Potential future fine-tuning targets:

```text
intent classification
emotion classification
structured routing
companion response style
follow-up behavior
Islamic source-aware response formatting
```

Do not fine-tune Qur'an/Hadith factual knowledge into the model as a replacement for RAG.

---

# 46. Definition of Done

Version 2 is complete when:

```text
[ ] State engine works
[ ] Companion policy works
[ ] Context builder works
[ ] Memory router works
[ ] RAG router works
[ ] Safety router works
[ ] Companion validator works
[ ] Multi-turn evaluation exists
[ ] "I feel lonely" regression passes
[ ] Explicit Islamic questions still pass grounded regression
[ ] Existing tests remain green
[ ] No source-policy invariants are weakened
[ ] Ling/Gemma can use the same harness
[ ] No mandatory cloud service exists
```

---

# 47. Final Architecture

The final system should conceptually become:

```text
                         ILMAN
                           │
                    ┌──────▼──────┐
                    │ UNDERSTAND  │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
        Intent           Emotion          Entity
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    ┌─────────────┐
                    │ STATE ENGINE│
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ POLICY      │
                    │ ENGINE      │
                    └──────┬──────┘
                           │
             ┌─────────────┼──────────────┐
             │             │              │
             ▼             ▼              ▼
          MEMORY          RAG          SAFETY
          ROUTER         ROUTER         ROUTER
             │             │              │
             └─────────────┼──────────────┘
                           ▼
                    ┌─────────────┐
                    │ CONTEXT     │
                    │ BUILDER     │
                    └──────┬──────┘
                           ▼
                 ┌──────────────────┐
                 │ LOCAL MODEL      │
                 │ Ling / Gemma /   │
                 │ Qwen / future    │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ VALIDATION       │
                 │                  │
                 │ Source           │
                 │ Citation         │
                 │ Safety           │
                 │ Policy           │
                 │ Companion        │
                 └────────┬─────────┘
                          ▼
                         USER
```

---

# 48. Core Principle

The objective is NOT to build:

> "A chatbot that knows Islam."

The objective is:

> **A local-first Islamic companion harness that knows when to listen, when to answer, when to retrieve, when to remember, when to ask, when to verify, and when not to say too much.**

The model supplies language and reasoning.

The harness supplies:

```text
identity
state
memory
policy
context
safety
tools
Islamic grounding
verification
evaluation
```

This is the foundation that should allow relatively small local models to behave substantially better than their raw benchmark capability suggests.

**Do not optimize model size until this harness has been measured.**
