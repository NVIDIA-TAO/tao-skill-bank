## Description: <br>
Run the full, self-contained DEFT improvement loop for NVIDIA TAO CLIP / SigLIP2 Image Attribute Augmentation (IAA) models: zero-shot evaluation, per-attribute gap analysis, text-embedding k-NN mining, history-aware selection, continual-dataset retraining, and re-evaluation against an approved retrieval metric. Customers provide the IAA data export, not an implementation source checkout. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>

## Use Case: <br>
<<<<<<< HEAD:skills/applications/tao-run-deft-iaa/skill-card.md
Developers and engineers use this skill to execute or resume the IAA DEFT workflow on local NVIDIA Docker. The workflow establishes a zero-shot baseline, analyzes weak attributes, mines a caption pool, selects samples without reusing evaluation data, retrains, and repeats until the approved metric target passes or the finite iteration budget is exhausted. A target-free run executes its approved budget and reports the best result. <br>
=======
Developers and engineers use this skill to execute or resume the PAS DEFT workflow on Docker, SLURM, Kubernetes, Brev, or a TAO-capable virtualenv. The workflow establishes a zero-shot baseline, analyzes weak attributes, mines a caption pool, selects samples without reusing evaluation data, retrains, and repeats until the approved metric target passes or the finite iteration budget is exhausted. A target-free run executes its approved budget and reports the best result. <br>
>>>>>>> b99df7e ([TAO-0][Bugfix] Run PAS DEFT actions on every TAO platform (#144)):skills/applications/tao-run-deft-pas/skill-card.md

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: GPU/container cost, interrupted multi-stage runs, stale artifacts, data leakage, or incorrect metric evidence can make an iterative result unreliable. <br>
Mitigation: The skill uses one explicit approval boundary before side effects, a bounded loop, deterministic config and platform-neutral action bundles, native job-record plus exact argv/digest and fresh-output evidence, current-attempt checkpoint provenance, evaluation-split leakage checks, immutable metric contracts, journaled canonical state commits, fail-closed audits, and bounded recovery. Credentials are inherited only from the launching process environment; credential files are never opened or sourced, and values are never printed, inspected, or persisted. <br>

## Reference(s): <br>
- [Pipeline and State](references/pipeline-and-state.md) <br>
- [Pre-Flight Checks](references/preflight.md) <br>
- [CLIP Train / Evaluate](references/clip-train-eval.md) <br>
- [IAA Metric Contract](references/metric-contract.md) <br>
- [Data Layout](references/data-layout.md) <br>
- [Gap Analysis](references/gap-analysis.md) <br>
- [Mining and Selection](references/mining.md) <br>
- [Visualization](references/visualization.md) <br>
- [Scripts and Stage Adapters](references/scripts-and-agents.md) <br>
- [Platform Execution](references/platform-execution.md) <br>

## Skill Output: <br>
**Output Type(s):** [Shell commands, HTML report, JSON state, JSONL event log, metrics and model artifacts] <br>
**Output Format:** [Markdown guidance and deterministic on-disk artifacts] <br>
**Output Parameters:** [Approved IAA DEFT run configuration and iteration budget] <br>
**Other Properties Related to Output:** [`deft_state.json` and `loop_log.jsonl` are canonical; `DEFT_Loop_Report.html` is rendered deterministically after a successful completion audit] <br>

## Evaluation Agents Used: <br>
- Codex (implementation review and local validation) <br>

## Evaluation Tasks: <br>
- Static validation: compile all bundled Python scripts, syntax-check the shell wrapper and metadata, and run the repository skill validator. <br>
- Synthetic state-machine smoke: render and initialize immutable inputs, complete a two-iteration max-budget path, require the completion audit, and render the report twice deterministically. <br>
- Branch/recovery checks: exercise disabled and enabled visualization, optional checksum evidence, nonterminal failure to terminal hard-stop, journaled state/log recovery, orphan-container overlap rejection, and persisted two-attempt limits. <br>
- Negative-path checks: reject cross-iteration artifacts, zero-row mining, false KPI stops, duplicate metric rows, unsupported KPI query types, stale/unbound outputs, command tampering, stale or cross-iteration checkpoint targets, symlink chains/escapes, and malformed state/log labels. <br>

## Evaluation Metrics Used: <br>
- Python and shell syntax success. <br>
- Deterministic transition, artifact-binding, metric-binding, and completion-audit behavior on a synthetic fixture. <br>
- Expected rejection of tested invalid inputs and transitions. <br>

## Evaluation Results: <br>
Repository/static checks and the synthetic two-iteration, visualization, checksum, hard-stop, authenticated-forwarding, recovery, retry-bound, checkpoint-contract, deterministic-report, and clean bundled-runtime import tests passed. The listed invalid evidence, transition, provenance, and path cases were rejected as expected. The checkpoint contract was also checked against a real IAA reference-run relative-symlink artifact shape. A live end-to-end run through the revised wrapper was not performed as part of this review. <br>

## Skill Version(s): <br>
0.3.3 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
