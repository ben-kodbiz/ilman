"""Model abstraction (agentodo.md §3, §28 step 4).

Every backend MUST be replaceable behind an OpenAI-compatible `/v1` interface.
Model routing is runtime config, never hard-coded. The agent never imports a
specific model SDK.

`chat_template_kwargs` passes vendor-specific toggles (e.g. Ling
`enable_thinking`) through to the backend without this code knowing about them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from agent.core.config import AppConfig, BackendConfig


@dataclass
class ChatMessage:
    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""  # set on role="tool" result messages


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""  # backend tool_call id; tool results must reference it

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> ToolCall:
        fn = raw.get("function", {})
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        return cls(name=fn.get("name", ""), arguments=args, id=raw.get("id") or "")


@dataclass
class ModelResponse:
    content: str
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    backend: str = ""
    model_id: str = ""


@dataclass
class ModelBackend:
    """One OpenAI-compatible endpoint. No vendor SDKs allowed."""

    config: BackendConfig
    timeout_s: float = 300.0

    def chat(
        self,
        model_id: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    **({"tool_calls": m.tool_calls} if m.tool_calls else {}),
                    # tool results must reference the assistant's tool_call id
                    **({"tool_call_id": m.tool_call_id} if m.role == "tool" and m.tool_call_id else {}),
                }
                for m in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if chat_template_kwargs:
            # Pass through vendor toggles (e.g. {"enable_thinking": false}).
            payload["chat_template_kwargs"] = chat_template_kwargs
        start = time.perf_counter()
        resp = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key()}"},
            json=payload,
            timeout=self.timeout_s,
        )
        latency = time.perf_counter() - start
        if resp.status_code != 200:
            raise RuntimeError(f"backend {self.config.name} HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message", {})
        usage = data.get("usage", {})
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        return ModelResponse(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or "",
            tool_calls=[ToolCall.from_openai(tc) for tc in msg.get("tool_calls") or []],
            finish_reason=choice.get("finish_reason", ""),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": reasoning_tokens,
            },
            latency_s=latency,
            backend=self.config.name,
            model_id=model_id,
        )


class ModelRouter:
    """Routes a task class to (backend, model_id) purely from runtime config."""

    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self._backends: dict[str, ModelBackend] = {}

    def backend(self, name: str) -> ModelBackend:
        if name not in self._backends:
            if name not in self.app_config.backends:
                raise ValueError(f"backend '{name}' not defined in config")
            self._backends[name] = ModelBackend(self.app_config.backends[name])
        return self._backends[name]

    def resolve(self, task_class: str) -> tuple[ModelBackend, str]:
        backend_cfg, model_id = self.app_config.backend_for(task_class)
        return self.backend(backend_cfg.name), model_id

    def chat(
        self,
        task_class: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> ModelResponse:
        backend, model_id = self.resolve(task_class)
        return backend.chat(
            model_id,
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            chat_template_kwargs=chat_template_kwargs,
        )
