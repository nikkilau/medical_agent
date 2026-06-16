from __future__ import annotations

import json
import re

from ..agents import Agent
from ..models import AgentResponse, PatientCase
from ..providers import LLMProvider


class DynamicMDTPath:
    """MDAgents-style dynamic specialist consultation path.

    It prefers the downloaded MDAgents source wrapper. If that source path is
    unavailable or fails, it falls back to a compact local recruiter/moderator
    flow so the interactive prototype remains usable.
    """

    def __init__(self, provider: LLMProvider, *, use_source_mdagents: bool = False) -> None:
        self.provider = provider
        self.use_source_mdagents = use_source_mdagents

    def run(self, case: PatientCase) -> tuple[str, list[AgentResponse], int]:
        if not self.use_source_mdagents:
            return _LocalDynamicMDT(self.provider).run(case)
        try:
            from medical_ttc.source_baselines import MDAgentsSourceWorkflow

            workflow = MDAgentsSourceWorkflow(self.provider, difficulty="intermediate")
            out = workflow(case)
            trace = [
                AgentResponse(
                    "DynamicMDTPath",
                    "Used downloaded MDAgents source workflow with difficulty=intermediate.",
                ),
                AgentResponse("MDAgents", out.final_answer),
            ]
            return out.final_answer, trace, max(out.rounds, 1)
        except Exception as exc:
            fallback = _LocalDynamicMDT(self.provider)
            answer, trace, rounds = fallback.run(case)
            trace.insert(0, AgentResponse("DynamicMDTPath", f"MDAgents fallback used: {exc}"))
            return answer, trace, rounds


class _LocalDynamicMDT:
    def __init__(self, provider: LLMProvider) -> None:
        self.recruiter = Agent(
            "MDTRecruiter",
            "You recruit exactly 3 medical specialist roles for a benchmark medical question.\n"
            "Return ONLY compact JSON, no reasoning, no markdown:\n"
            "{\"roles\":[\"Specialist role - expertise phrase\","
            "\"Specialist role - expertise phrase\","
            "\"Specialist role - expertise phrase\"]}\n"
            "Roles must be clinical specialties or medical domain experts, not diagnoses or answers.",
            provider,
        )
        self.specialist_prompt = (
            "You are {role}. Give a concise specialist opinion for the question. "
            "If multiple-choice, end with Final answer: <A/B/C/D/E>."
        )
        self.moderator = Agent(
            "MDTModerator",
            "You synthesize specialist opinions and choose the final medical benchmark answer. "
            "If multiple-choice, end with Final answer: <A/B/C/D/E>.",
            provider,
        )
        self.provider = provider

    def run(self, case: PatientCase) -> tuple[str, list[AgentResponse], int]:
        recruited = self.recruiter.run(case.text, temperature=0.0)
        roles = self._parse_roles(recruited.content)
        if len(roles) < 3:
            roles = self._fallback_roles(case.text)
        trace = [
            AgentResponse(
                "MDTRecruiter",
                "Selected specialists:\n"
                + "\n".join(f"{idx + 1}. {role}" for idx, role in enumerate(roles))
                + "\n\nRaw recruiter output:\n"
                + recruited.content[:800],
            )
        ]
        opinions: list[str] = []
        for role in roles:
            agent = Agent(f"Specialist:{role[:30]}", self.specialist_prompt.format(role=role), self.provider)
            opinion = agent.run(case.text, temperature=0.2)
            trace.append(opinion)
            opinions.append(f"{role}:\n{opinion.content}")
        answer = self.moderator.run(
            f"Question:\n{case.text}\n\nSpecialist opinions:\n" + "\n\n".join(opinions),
            temperature=0.0,
        )
        answer.content = self._strip_thinking(answer.content)
        trace.append(answer)
        return answer.content, trace, len(trace)

    @classmethod
    def _parse_roles(cls, raw: str) -> list[str]:
        text = cls._strip_thinking(raw)
        roles = cls._parse_json_roles(text)
        if not roles:
            roles = cls._parse_list_roles(text)
        clean_roles: list[str] = []
        seen: set[str] = set()
        for role in roles:
            cleaned = cls._clean_role(role)
            if not cleaned or cleaned.lower() in seen:
                continue
            if not cls._looks_like_specialist(cleaned):
                continue
            clean_roles.append(cleaned)
            seen.add(cleaned.lower())
            if len(clean_roles) == 3:
                break
        return clean_roles

    @staticmethod
    def _strip_thinking(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
        return text.replace("```", "").strip()

    @staticmethod
    def _parse_json_roles(text: str) -> list[str]:
        candidates = [text]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("roles"), list):
                return [str(x) for x in obj["roles"]]
            if isinstance(obj, list):
                return [str(x) for x in obj]
        return []

    @staticmethod
    def _parse_list_roles(text: str) -> list[str]:
        roles: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*]|\d+[.)、]|[一二三四五六七八九十]+[、.])\s*", "", line)
            if line:
                roles.append(line)
        return roles

    @staticmethod
    def _clean_role(role: str) -> str:
        role = role.strip().strip('"').strip("'")
        role = re.split(r"\s*(?:\||：|:)\s*", role, maxsplit=1)[0] if len(role) > 80 else role
        role = re.sub(r"\s+", " ", role)
        return role[:80].strip(" -")

    @staticmethod
    def _looks_like_specialist(role: str) -> bool:
        lowered = role.lower()
        markers = (
            "specialist",
            "physician",
            "clinician",
            "doctor",
            "surgeon",
            "internist",
            "pharmacologist",
            "toxicologist",
            "hepatologist",
            "gastroenterologist",
            "endocrinologist",
            "cardiologist",
            "neurologist",
            "emergency",
            "ethicist",
            "expert",
            "专家",
            "医师",
            "医生",
            "内科",
            "外科",
            "急诊",
            "药师",
            "药理",
            "毒理",
            "肝病",
            "消化",
            "内分泌",
            "心脏",
            "神经",
            "伦理",
        )
        return any(marker in lowered or marker in role for marker in markers)

    @staticmethod
    def _fallback_roles(query: str) -> list[str]:
        lowered = query.lower()
        if any(term in lowered for term in ("ast", "alt", "jaundice", "黄疸", "肝", "pt/ptt")):
            return [
                "Hepatologist - acute liver injury and hepatic failure",
                "Clinical toxicologist - drug-induced liver injury",
                "Clinical pharmacologist - medication exposure analysis",
            ]
        if any(term in lowered for term in ("insulin", "diabetes", "胰岛素", "糖尿病")):
            return [
                "Endocrinologist - diabetes and insulin safety",
                "Clinical pharmacologist - medication dosing risk",
                "Emergency physician - acute deterioration triage",
            ]
        if any(term in lowered for term in ("policy", "insurance", "socioeconomic", "政策", "保险", "社会经济")):
            return [
                "Medical ethicist - fairness and patient rights",
                "Health policy specialist - access to necessary care",
                "Clinical pharmacist - medication necessity and alternatives",
            ]
        return [
            "General internist - broad differential diagnosis",
            "Clinical pharmacologist - medication and exposure review",
            "Emergency physician - red flag triage",
        ]
