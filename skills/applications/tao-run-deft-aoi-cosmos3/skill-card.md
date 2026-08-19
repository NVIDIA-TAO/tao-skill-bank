## Description: <br>
Run the disk-backed DEFT AOI improvement loop for Cosmos Reason 3 / Cosmos3 with Nano as the default and explicit Edge or Super selection. Use cosmos-rl SFT over single-image ShareGPT records whose assistant label is exactly OK or NG. Proxy KPI errors drive RCCA and real-image mining; a frozen Benchmark KPI metric alone stops the loop. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>

## Use Case: <br>
Developers use this skill to improve Cosmos3 PCB/AOI inspection from Proxy, Benchmark, and Mining inputs. The loop starts with a zero-shot frozen Benchmark gate and runs Proxy only when that gate is unmet, then creates `train_iter_<N>.json` from Mining samples selected after Proxy RCA; later sets grow monotonically while remaining isolated from Proxy and Benchmark. Execution stays platform-neutral, job-recorded, and state-backed. This migration intentionally supports only bare OK/NG labels. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Data leakage, changed Benchmark data, ambiguous mined-image alignment, or non-exact labels can invalidate reported quality. <br>
Mitigation: The bundled validators, frozen Benchmark hash, and atomic stage commits hard-stop these conditions. <br>

## Reference(s): <br>
- [Pre-Flight](references/preflight.md) <br>
- [Pipeline and State](references/pipeline-and-state.md) <br>
- [Cosmos3 Train/Evaluate](references/cosmos-reason.md) <br>
- [Bare AOI Annotation](references/aoi-annotation.md) <br>
- [Metric Contract](references/metric-contract.md) <br>
- [Scripts and Agents](references/scripts-and-agents.md) <br>

## Skill Output: <br>
**Output Type(s):** [TOML specs, JSON state, model checkpoints, HTML report] <br>
**Output Format:** [Disk-backed workflow artifacts] <br>
**Other Properties Related to Output:** [Every GPU stage is tracked by a platform job-record and every DEFT commit is recorded in state] <br>

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility. Validate this workflow, data rights, defect taxonomy, acceptance criteria, and operational risks with the deploying organization. <br>
