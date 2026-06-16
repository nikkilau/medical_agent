"""Wrappers for the downloaded MedAgents and MDAgents source trees.

These classes intentionally call the source projects' workflow functions and
only adapt their LLM calls to this repo's provider interface.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from micro_mdt.models import LLMMessage, PatientCase
from micro_mdt.providers import LLMProvider

from .workflow import Workflow, WorkflowResult


REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_source_import_stubs() -> None:
    if "termcolor" not in sys.modules:
        mod = types.ModuleType("termcolor")
        mod.cprint = lambda *args, **kwargs: print(*args[:1])
        sys.modules["termcolor"] = mod

    if "prettytable" not in sys.modules:
        mod = types.ModuleType("prettytable")

        class PrettyTable:
            def __init__(self, *args, **kwargs) -> None:
                self.rows: list[Any] = []

            def add_row(self, row) -> None:
                self.rows.append(row)

            def __str__(self) -> str:
                return "\n".join(str(row) for row in self.rows)

        mod.PrettyTable = PrettyTable
        sys.modules["prettytable"] = mod

    if "pptree" not in sys.modules:
        mod = types.ModuleType("pptree")

        class Node:
            def __init__(self, name, parent=None) -> None:
                self.name = name
                self.parent = parent
                self.children: list[Any] = []
                if parent is not None and hasattr(parent, "children"):
                    parent.children.append(self)

        mod.Node = Node
        mod.print_tree = lambda *args, **kwargs: None
        sys.modules["pptree"] = mod

    if "google" not in sys.modules:
        sys.modules["google"] = types.ModuleType("google")
    if "google.generativeai" not in sys.modules:
        genai = types.ModuleType("google.generativeai")
        genai.configure = lambda *args, **kwargs: None
        genai.GenerativeModel = lambda *args, **kwargs: None
        sys.modules["google.generativeai"] = genai

    if "jsonlines" not in sys.modules:
        mod = types.ModuleType("jsonlines")

        class Reader:
            def __init__(self, file_obj) -> None:
                self.file_obj = file_obj

            def __iter__(self):
                import json

                for line in self.file_obj:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

        mod.Reader = Reader
        sys.modules["jsonlines"] = mod

    if "nltk" not in sys.modules:
        nltk = types.ModuleType("nltk")
        tokenize = types.ModuleType("nltk.tokenize")
        tokenize.sent_tokenize = lambda text: re.split(r"(?<=[.!?])\s+", text.strip()) if text else []
        sys.modules["nltk"] = nltk
        sys.modules["nltk.tokenize"] = tokenize

    if "rouge_score" not in sys.modules:
        rouge_score = types.ModuleType("rouge_score")
        rouge_scorer = types.ModuleType("rouge_score.rouge_scorer")

        class RougeScorer:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def score(self, prediction, target):
                value = 1.0 if prediction == target else 0.0
                metric = types.SimpleNamespace(fmeasure=value)
                return {"rouge1": metric, "rouge2": metric, "rougeL": metric}

        rouge_scorer.RougeScorer = RougeScorer
        sys.modules["rouge_score"] = rouge_score
        sys.modules["rouge_score.rouge_scorer"] = rouge_scorer


def _load_source_module(alias: str, path: Path, extra_path: Path):
    _install_source_import_stubs()
    inserted = False
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))
        inserted = True
    try:
        module = types.ModuleType(alias)
        module.__file__ = str(path)
        module.__package__ = ""
        sys.modules[alias] = module
        source = path.read_text(encoding="utf-8")
        source = _patch_source_compatibility(source)
        exec(compile(source, str(path), "exec"), module.__dict__)
        return module
    finally:
        if inserted:
            try:
                sys.path.remove(str(extra_path))
            except ValueError:
                pass


def _patch_source_compatibility(source: str) -> str:
    source = re.sub(
        r'^\s*print\(f"\{.*?moderator\'s final decision \(by majority vote\):", _decision\)\s*$',
        '    print("moderator final decision (by majority vote):", _decision)',
        source,
        flags=re.MULTILINE,
    )
    return source


def _parse_medqa_question_and_options(text: str) -> tuple[str, dict[str, str]]:
    lines = text.splitlines()
    q_lines: list[str] = []
    options: dict[str, str] = {}
    option_re = re.compile(r"^\s*([A-E])[\).]\s*(.+?)\s*$")
    seen_option = False
    for line in lines:
        match = option_re.match(line)
        if match:
            seen_option = True
            options[match.group(1).upper()] = match.group(2).strip()
        elif not seen_option and line.strip():
            q_lines.append(line.strip())
    question = " ".join(q_lines).strip() or text
    return question, options


def _format_mdagents_question(case: PatientCase) -> str:
    question, options = _parse_medqa_question_and_options(case.text)
    if not options:
        return case.text
    opt_text = " ".join(f"({key}) {value}" for key, value in options.items())
    return f"{question} Options: {opt_text}"


def _flatten_source_response(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            parts.append(f"{key}: {_flatten_source_response(item)}")
        return "\n".join(parts)
    if isinstance(value, (list, tuple)):
        return "\n".join(_flatten_source_response(item) for item in value)
    return str(value)


class _MedAgentsProviderHandler:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def get_output_multiagent(
        self,
        system_role: str,
        user_input: str,
        max_tokens: int,
        temperature: float = 0,
        **_: Any,
    ) -> str:
        messages = []
        if system_role:
            messages.append(LLMMessage(role="system", content=system_role))
        messages.append(LLMMessage(role="user", content=user_input))
        return self.provider.complete(messages, temperature=temperature)


@dataclass
class _MedAgentsArgs:
    method: str = "syn_verif"
    max_attempt_vote: int = 3


class MedAgentsSourceWorkflow(Workflow):
    """B3 baseline using the downloaded `MedAgents/` source workflow."""

    def __init__(self, provider: LLMProvider, *, method: str = "syn_verif", max_attempt_vote: int = 3) -> None:
        super().__init__("B3_MedAgents_Source")
        self.provider = provider
        self.method = method
        self.max_attempt_vote = max_attempt_vote
        self.module = _load_source_module(
            "medagents_source_utils",
            REPO_ROOT / "MedAgents" / "utils.py",
            REPO_ROOT / "MedAgents",
        )

    def __call__(self, case: PatientCase) -> WorkflowResult:
        question, options = _parse_medqa_question_and_options(case.text)
        args = _MedAgentsArgs(method=self.method, max_attempt_vote=self.max_attempt_vote)
        data_info = self.module.fully_decode(
            case.case_id,
            case.case_id,
            question,
            options,
            "",
            _MedAgentsProviderHandler(self.provider),
            args,
            dataobj=None,
        )
        pred = data_info.get("pred_answer") or ""
        raw = data_info.get("raw_output") or ""
        final = f"Option: {pred}\n{raw}" if pred else raw
        return WorkflowResult(
            case=case,
            final_answer=final,
            rounds=self._estimate_rounds(data_info),
            difficulty="N/A",
            trace=[{"step": "MedAgents.fully_decode", "method": self.method}],
            metadata={"source": "MedAgents", "method": self.method, "raw": data_info},
        )

    def _estimate_rounds(self, data_info: dict) -> int:
        if self.method.startswith("base"):
            return 1
        rounds = 2
        rounds += len(data_info.get("question_domains") or [])
        rounds += len(data_info.get("option_domains") or [])
        rounds += len(data_info.get("vote_history") or [])
        return max(rounds, 1)


class _MDAgentSourceAgent:
    def __init__(self, instruction, role, examplers=None, model_info="gpt-4o-mini", img_path=None):
        self.instruction = instruction
        self.role = role
        self.messages = [LLMMessage(role="system", content=instruction)] if instruction else []

    def chat(self, message, img_path=None, chat_mode=True):
        provider = _MDAgentSourceAgent.provider
        messages = list(self.messages)
        messages.append(LLMMessage(role="user", content=message))
        out = provider.complete(messages, temperature=0.0)
        self.messages.append(LLMMessage(role="user", content=message))
        self.messages.append(LLMMessage(role="assistant", content=out))
        return out

    def temp_responses(self, message, img_path=None):
        return {0.0: self.chat(message, img_path=img_path)}


_MDAgentSourceAgent.provider = None  # type: ignore[attr-defined]


@dataclass
class _MDAgentsArgs:
    dataset: str = "medqa"
    model: str = "gpt-4o-mini"
    difficulty: str = "adaptive"


class MDAgentsSourceWorkflow(Workflow):
    """B4 baseline using the downloaded `MDAgents/` source workflow."""

    def __init__(self, provider: LLMProvider, *, difficulty: str = "adaptive") -> None:
        super().__init__("B4_MDAgents_Source")
        self.provider = provider
        self.difficulty = difficulty
        self.module = _load_source_module(
            "mdagents_source_utils",
            REPO_ROOT / "MDAgents" / "utils.py",
            REPO_ROOT / "MDAgents",
        )
        self.module.Agent = _MDAgentSourceAgent

    def __call__(self, case: PatientCase) -> WorkflowResult:
        _MDAgentSourceAgent.provider = self.provider
        question = _format_mdagents_question(case)
        args = _MDAgentsArgs(difficulty=self.difficulty)
        difficulty = self.module.determine_difficulty(question, self.difficulty)
        if difficulty not in {"basic", "intermediate", "advanced"}:
            difficulty = "basic"
        if difficulty == "basic":
            final_decision = self.module.process_basic_query(question, [], args.model, args)
        elif difficulty == "intermediate":
            final_decision = self.module.process_intermediate_query(question, [], args.model, args)
        else:
            final_decision = self.module.process_advanced_query(question, args.model, args)
        final_text = _flatten_source_response(final_decision)
        return WorkflowResult(
            case=case,
            final_answer=final_text,
            rounds={"basic": 2, "intermediate": 8, "advanced": 12}.get(difficulty, 2),
            difficulty=difficulty,
            trace=[{"step": "MDAgents.source", "difficulty": difficulty}],
            metadata={"source": "MDAgents", "difficulty": difficulty, "raw": final_decision},
        )
