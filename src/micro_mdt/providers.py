from __future__ import annotations

import json
import os
import re
import urllib.request
from abc import ABC, abstractmethod

from .models import LLMMessage


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[LLMMessage], *, temperature: float = 0.2) -> str:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Deterministic provider for demos and tests without network or API keys."""

    high_risk_terms = [
        "胸痛",
        "呼吸困难",
        "肾功能",
        "eGFR",
        "怀孕",
        "妊娠",
        "华法林",
        "阿司匹林",
        "过敏",
        "儿童",
        "老人",
        "心衰",
        "昏迷",
        "黑便",
        "罕见",
    ]
    urgent_terms = ["胸痛", "呼吸困难", "黑便", "呕血", "昏迷", "意识改变"]
    medium_terms = ["糖尿病", "高血压", "复诊", "慢病", "血糖", "血压", "用药调整"]

    def complete(self, messages: list[LLMMessage], *, temperature: float = 0.2) -> str:
        system = messages[0].content if messages else ""
        user = "\n".join(m.content for m in messages if m.role == "user")

        if "分诊 Agent" in system:
            return self._triage(user)
        if "临床药师" in system:
            return self._pharmacy(user)
        if "医疗安全与伦理" in system:
            return self._safety(user)
        if "医疗文书 Agent" in system:
            return self._documentation(user)
        if "医疗决策综合" in system:
            return self._decision_synthesis(user)
        if "修订上一版方案" in system:
            return self._revision(user)
        return self._proposal(user)

    def _triage(self, text: str) -> str:
        if self._has_any_unnegated(text, self.high_risk_terms):
            return "[HIGH]\n存在高危症状、特殊人群或复杂用药，需要 MDT 辩论和安全审查。"
        if any(term in text for term in self.medium_terms):
            return "[MEDIUM]\n属于慢病复诊或轻度复杂场景，需要一次交叉复核。"
        return "[LOW]\n描述更接近常见轻症，先走低算力路径。"

    def _proposal(self, text: str) -> str:
        if "华法林" in text or "阿司匹林" in text:
            return (
                "问题清单：头痛/发热待查，合并抗凝或抗血小板用药风险。\n"
                "初步建议：避免自行叠加止痛药或活血类药物；若有黑便、呕血、意识改变或持续高热，立即就医。\n"
                "建议检查：血常规、凝血功能，并由医生评估是否需要调整现有药物。"
            )
        if "肾功能" in text or "eGFR" in text:
            return (
                "问题清单：慢病控制不佳，合并肾功能下降风险。\n"
                "初步建议：暂不建议自行增加或更换降糖降压药；需要医生结合 eGFR、尿蛋白和电解质评估。\n"
                "建议检查：肾功能、电解质、尿白蛋白肌酐比，并评估心血管风险。"
            )
        if "胸痛" in text or "呼吸困难" in text:
            return (
                "问题清单：胸痛或呼吸困难需优先排除急性冠脉综合征、肺栓塞等急症。\n"
                "初步建议：不应在线上延误处理，请尽快急诊评估。\n"
                "建议检查：心电图、肌钙蛋白、血氧和必要影像检查。"
            )
        return (
            "问题清单：常见上呼吸道感染或轻症不适可能。\n"
            "初步建议：休息、补液、观察体温和症状变化；避免自行使用抗生素。\n"
            "如高热不退、胸痛、呼吸困难、意识异常或症状加重，应线下就医。"
        )

    def _pharmacy(self, text: str) -> str:
        if re.search(r"自行(增加|停用|换用)|具体剂量|加用", text):
            return "[REVISE]\n方案中存在可能越权的自行调药或具体用药指令，需要改为医生确认后执行。"
        if "华法林" in text and ("止痛" in text or "阿司匹林" in text):
            return "[REVISE]\n合并抗凝/抗血小板药物时有出血风险，需要明确避免自行叠加相关药物并建议凝血评估。"
        return "[PASS]\n未发现明确药物相互作用或剂量越权问题。"

    def _safety(self, text: str) -> str:
        case_text = self._extract_case_text(text)
        if self._has_any_unnegated(case_text, self.urgent_terms):
            return "[ABSTAIN]\n存在潜在急症风险，AI 不应给出最终诊疗方案，应转交急诊或人类医生。"
        if "暂不建议自行" in text or "医生评估" in text or "线下就医" in text:
            return "[PASS]\n方案保持保守并设置了就医条件。"
        return "[REVISE]\n需要补充危险信号、线下就医条件和不替代医生诊疗的边界。"

    @staticmethod
    def _extract_case_text(text: str) -> str:
        if "待审查方案：" in text:
            return text.split("待审查方案：", 1)[0]
        return text

    @staticmethod
    def _has_any_unnegated(text: str, terms: list[str]) -> bool:
        lowered = text.lower()
        for term in terms:
            term_lower = term.lower()
            start = 0
            while True:
                index = lowered.find(term_lower, start)
                if index == -1:
                    break
                prefix = text[max(0, index - 3):index]
                if not any(marker in prefix for marker in ("无", "否认", "没有", "未见")):
                    return True
                start = index + len(term_lower)
        return False

    def _decision_synthesis(self, text: str) -> str:
        if "华法林" in text or "黑便" in text or "出血" in text:
            return (
                "[A] 驳回 AI 方案，安排急诊评估出血风险并检测凝血功能\n"
                "[B] 同意保守处理，暂停可能加重出血的药物，48小时内门诊复诊\n"
                "[C] 手动输入其他意见"
            )
        if "肾功能" in text or "eGFR" in text or "肌酐" in text:
            return (
                "[A] 驳回 AI 方案，转肾内科会诊后调整用药\n"
                "[B] 同意 AI 减量/换药建议，但需加测肾功能和电解质后执行\n"
                "[C] 手动输入其他意见"
            )
        return (
            "[A] 驳回 AI 提议，维持原治疗方案，加强监测\n"
            "[B] 同意 AI 方案，但增加安全限制条件后执行\n"
            "[C] 手动输入其他意见"
        )

    def _revision(self, text: str) -> str:
        return (
            "修订方案：本系统不提供最终诊断或处方。建议将病例摘要、当前用药、过敏史和检查结果交由医生审核。\n"
            "安全建议：避免自行停药、换药或叠加药物；如出现胸痛、呼吸困难、黑便、意识改变、高热不退等，立即急诊。\n"
            "下一步：完善基础生命体征、相关化验和既往病史，由医生决定治疗。"
        )

    def _documentation(self, text: str) -> str:
        return (
            "[FOLLOWUP]\n"
            "主诉与现病史：根据输入病例及 AI 审查摘要整理，需医生核对原始病史。\n"
            "AI 会诊分歧摘要：AI 专家组在用药方案上存在分歧，已呈报主任医师裁决。\n"
            "医生最终决策：已按主任医师意见确定最终方案。\n"
            "治疗与用药计划：遵医嘱执行，避免患者自行调整处方药；监测生命体征和不良反应。\n"
            "（本草稿需经主任医师审核签字后生效）\n"
            "[PATIENT_NOTE]\n"
            "【用药指导】请严格按医生确认后的方案用药，不可自行增减剂量或停用。\n"
            "【生活建议】合理饮食、适度运动、戒烟限酒、保持情绪稳定。\n"
            "【危险信号】如出现胸痛、呼吸困难、黑便、意识异常、高热不退、水肿加重等情况，请立即就医。\n"
            "（本草稿需经主任医师审核签字后生效）\n"
            "[REMINDER]\n"
            "下次复诊时间：按医生建议时间复诊，建议 2-4 周内。\n"
            "复诊前准备：近期检查结果、当前用药清单（含用法用量）、过敏史记录。\n"
            "建议提前完成的检查：根据医生指示完成血常规、生化全项、凝血功能等。\n"
            "（本草稿需经主任医师审核签字后生效）"
        )


class OpenAICompatibleProvider(LLMProvider):
    """Minimal stdlib client for OpenAI-compatible chat/completions endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("MICRO_MDT_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("MICRO_MDT_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("MICRO_MDT_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("Missing API key. Set MICRO_MDT_API_KEY or OPENAI_API_KEY.")
        self._endpoint = f"{self.base_url}/chat/completions"
        print(f"[LLM Provider] model={self.model}, endpoint={self._endpoint}", flush=True)

    def complete(self, messages: list[LLMMessage], *, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"API error {e.code}: {e.reason}. "
                f"Response: {detail[:500]}. "
                f"Check model name '{self.model}' is valid at {self.base_url}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach {self._endpoint}: {e.reason}. "
                f"Check network and MICRO_MDT_BASE_URL."
            ) from e
        return body["choices"][0]["message"]["content"]
