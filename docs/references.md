# References for Medical TTC

> 本文档收录本项目方法设计参考的全部论文。arXiv ID 大部分由 Explore agent 检索得到，2024 Q4 + 2025 的部分论文正式引用前建议手工 verify 一次。

## A. 核心理论基础（TTC 理论与基础推理工具）

- **Snell et al. 2024** — Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters | arXiv:2408.03314
  - 本方案的 north star：compute-optimal allocation + sequential revisions + search/verifier
- **Wei et al. 2022** — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | arXiv:2201.11903
- **Wang et al. 2022** — Self-Consistency Improves Chain of Thought Reasoning | arXiv:2203.11171
- **Madaan et al. 2023** — Self-Refine: Iterative Refinement with Self-Feedback | arXiv:2303.17651
- **Nori et al. 2023** — MedPrompt: Can Generalist Foundation Models Outcompete Special-Purpose Tuning? | arXiv:2311.16452

## B. 自动化工作流生成 / 搜索（AFLOW 同族）

- **AFLOW** (ICLR 2025) — Automating Agentic Workflow Generation | arXiv:2410.10762
  - MCTS over code-represented workflows；本方案 Phase 3 主算法
- **ADAS** (ICLR 2025) — Automated Design of Agentic Systems | arXiv:2408.08435
  - Meta-agent code search，AFLOW 论文直接对比的 baseline
- **AgentSquare** (ICLR 2025) — Automatic LLM Agent Search in Modular Design Space
  - 4 维模块组合（Planning/Reasoning/Tool/Memory）
- **MaAS** (ICML 2025 Oral) — Multi-agent Architecture Search via Agentic Supernet | arXiv:2502.04180
  - Query-adaptive sampling；本方案 Router 的灵感来源
- **ScoreFlow** (2025) — Score-DPO over code workflows | arXiv:2502.04306
  - 用连续 Score-DPO 替代 MCTS 的二元反馈；本方案 Phase 4 disagreement score 灵感来源
- **Maestro** (2025) — Joint Graph & Config Optimization | arXiv:2509.04642
- **RobustFlow** (2025) — Paraphrase-robust workflow optimization | arXiv:2509.21834
  - 本方案 Phase 4 鲁棒性测试灵感来源
- **EvoAgent** (NAACL 2025) — Evolutionary multi-agent | arXiv:2406.14228
- **MetaAgent** (ICML 2025) — FSM-based multi-agent | arXiv:2507.22606
- **EvoAgentX** (2025) — Integrated framework | arXiv:2507.03616
- **FLORA** (2025) — GNN-accelerated workflow evaluation | arXiv:2503.11301
- **ARTEMIS** (2025) — No-code evolutionary optimization | arXiv:2512.09108
- **AutoFlow** (Li et al. 2024) — Automated Workflow Generation for LLM Agents | arXiv:2407.12821

## C. 图结构 / 拓扑搜索

- **GPTSwarm** (ICML 2024) — Language Agents as Optimizable Graphs | arXiv:2402.16823
- **G-Designer** (ICLR 2025) — VAE-based Topology Design | arXiv:2410.11782
- **FLOW** (ICLR 2025) — Modularized Agentic Workflow Automation | arXiv:2501.07834
- **DyLAN** (COLM 2024) — Dynamic LLM-Agent Network | arXiv:2310.02170

## D. Prompt / Text-gradient 优化

- **DSPy/MIPRO** (2024) — Optimizing Instructions and Demonstrations | arXiv:2406.11695
  - 本方案 W₇ DSPy 对照基线使用
- **TextGrad** (2024) — Automatic "Differentiation" via Text | arXiv:2406.07496
- **Trace** (2024) — Generative Optimization over Execution Traces | arXiv:2406.16218
- **MAPRO** (2025) — Multi-agent Prompt Optimization via MAP Inference | arXiv:2510.07475
- **Promptbreeder** (ICLR 2024) — Self-Referential Self-Improvement | arXiv:2309.16797

## E. 医疗多 agent

- **MedAgents** (ACL 2024 Findings) — Zero-shot Role Debate | arXiv:2311.10537
- **MDAgents** (NeurIPS 2024 Oral) — Adaptive Collaboration for Medical Decisions | arXiv:2404.15155
  - 本方案 W₃ 启发式基线复现对象
- **AI Hospital** (2024) — Multi-agent Medical Simulator | arXiv:2402.09742

## F. 安全 / 拒答 / 评测

- **MedSafetyBench** — GitHub `AI4LIFE-GROUP/med-safety-bench`
  - 本方案安全侧评测数据集
- **MedQA (USMLE)** — Jin et al. 2020 | arXiv:2009.13081
  - 本方案主指标数据集，HuggingFace `bigbio/med_qa`
- **Agent Workflow Memory** (ICML 2025) | arXiv:2409.07429

## G. Mixture-of-Agents / Inference-time scaling

- **Symbolic Learning** (Zhou et al. 2024) — Self-Evolving Agents | arXiv:2406.18532
- **AlphaEvolve** (DeepMind 2025) — Code Search via Gemini

---

## 引用建议

正式写论文/报告时，建议优先引用：
- **核心方法**：Snell 2024 (TTC) + AFLOW (workflow search) + MaAS (supernet routing)
- **基线对比**：MDAgents (medical heuristic) + DSPy (prompt-only)
- **评测**：MedQA + MedSafetyBench
- **创新点定位**：与 AFLOW、MaAS、ScoreFlow 的关键区别
