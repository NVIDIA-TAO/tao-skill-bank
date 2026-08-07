/**
 * Guard extension (port of the DEFT kit's PreToolUse guard.sh).
 *
 * Blocks known-bad invocations BEFORE the agent burns tokens on them.
 * Claude Code contract was: exit 2 blocks the call, stderr shown to the agent.
 * Pi contract: return { block: true, reason } from a tool_call handler —
 * the reason is returned to the model as the tool error.
 *
 * Guards are the ONLY permission layer under Pi (no permission prompts exist),
 * so this file also carries a small deny-list of generally destructive
 * commands that Claude Code's permission system used to catch.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { execSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";

export default function (pi: ExtensionAPI) {
	const recent: string[] = [];
	let calls = 0;
	let budgetWarned = false;
	const budget = parseInt(process.env.PI_KIT_TURN_BUDGET ?? "0", 10) || 0;

	pi.on("tool_call", async (event) => {
		if (event.toolName !== "bash") return undefined;
		const cmd = String(event.input.command ?? "");
		if (!cmd) return undefined;

		// Turn budget (generic, env-gated) — measured: models that cannot end
		// their turn burn 1000+ messages per session. Warn near the budget,
		// hard-block past it.
		calls++;
		if (budget > 0 && calls > budget) {
			return { block: true, reason: `GUARD(budget): tool-call budget (${budget}) exhausted for this session. STOP now: print the card's STAGE_DONE token as your final message and end your turn. The driver will re-enter with a fresh session.` };
		}
		if (budget > 0 && !budgetWarned && calls > budget - 10) {
			budgetWarned = true;
			return { block: true, reason: `GUARD(budget): only ${budget - calls + 1} tool calls left in this session. Finish the CURRENT card step, commit what is committable, print the STAGE_DONE token, and end your turn. (This call was blocked only to deliver the warning — you may re-issue it.)` };
		}

		// Loop breaker — at temperature 0 the nano model wedges itself repeating
		// NEAR-identical failing commands (measured: a 475-message session cycling
		// `find ... *.pth` variants differing only in whitespace/pipes). Match on
		// a normalized prefix over a sliding window, not exact equality.
		const sig = cmd.replace(/\s+/g, " ").trim().slice(0, 80);
		recent.push(sig);
		if (recent.length > 8) recent.shift();
		const same = recent.filter((s) => s === sig).length;
		if (same >= 4) {
			return {
				block: true,
				reason: `GUARD(loop): you have issued ${same} nearly-identical commands — this is not making progress, and whatever you keep searching for does not exist. STOP repeating. Re-read the card's failure branch and EXECUTE ITS FIX now (edit the spec file / relaunch from commands.log), or re-append the previous stage's ok line and end your turn.`,
			};
		}

		// Guard 0 — NO FABRICATED RESULTS. The nano model was measured writing
		// 'train ok' log entries (with invented metrics) while train.log ended
		// 'Execution status: FAIL' and no container had run. Verify reality
		// before allowing a train-success log entry.
		if (/(log_stage|commit_stage)/.test(cmd) && /--stage\s+train\b/.test(cmd) && !/--status\s+error\b/.test(cmd)) {
			const rd = process.env.PI_KIT_RD ?? process.env.RD ?? "";
			const iterMatch = cmd.match(/--iter-label\s+(\S+)/);
			// The command text is RAW (pre-shell-expansion): cards pass the literal
			// `$ITER`, so resolve env references and quotes before building the path,
			// falling back to the driver-exported ITER.
			let iter = (iterMatch ? iterMatch[1] : "").replace(/^["']|["']$/g, "");
			iter = iter.replace(/\$\{?([A-Za-z_]\w*)\}?/g, (_, v) => process.env[v] ?? "");
			if (!iter) iter = process.env.ITER ?? "";
			const trainLog = rd && iter ? path.join(rd, iter, "train", "train.log") : "";
			let passed = false;
			try {
				if (trainLog) passed = fs.readFileSync(trainLog, "utf8").includes("Execution status: PASS");
			} catch {
				/* missing log = definitely not passed */
			}
			if (!passed) {
				return {
					block: true,
					reason: `GUARD(verify-train): you are logging 'train ok' but ${trainLog || "$RD/<iter>/train/train.log"} does not contain 'Execution status: PASS'. Training has NOT succeeded — never fabricate results or metrics. Follow the card's failure branch instead: fix the spec issue named in the log, relaunch the recorded train command detached, and END YOUR TURN without logging.`,
				};
			}
		}

		// Baseline safety net (Pi has no permission system underneath us).
		if (/\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\b\s+(\/|\$HOME|~)(\s|$)/.test(cmd) || /\bsudo\b/.test(cmd)) {
			return { block: true, reason: "GUARD(safety): destructive/root command blocked by kit policy. Work inside $WS and $RD only; nothing in this workflow needs sudo." };
		}

		// Secrets: the session inherits the provider API key in its environment,
		// and everything the model prints is persisted to the session JSONL and
		// re-sent off-host. No card ever needs a credential value — block env
		// dumps and any command naming a known credential var. (Anchored so
		// routine `set -e` inside compound commands does not false-positive.)
		if (
			/^\s*(env|printenv|export\s+-p|declare\s+-p|set)\s*(\||>|$)/.test(cmd) ||
			/(NVIDIA_INFERENCE_API_KEY|ANTHROPIC_API_KEY|NGC_(API_)?KEY|HF_TOKEN|WANDB_API_KEY)/.test(cmd)
		) {
			return { block: true, reason: "GUARD(secrets): this workflow never needs credential values — do not read or print environment secrets; their values must never enter the session log. Use only the card's named variables ($WS, $RD, $ITER, ...)." };
		}

		// Guard 1 — SigLIP embedding has no cuDNN conv engine on this GPU (sm_75).
		// Scope the check to each `docker run` segment: a compound command may
		// legitimately pair a CPU embedding run with a --gpus knn run.
		const embeddingWithGpus = cmd
			.split(/docker run/)
			.some((seg) => seg.includes("image_embeddings") && seg.includes("--gpus"));
		if (embeddingWithGpus) {
			return {
				block: true,
				reason:
					"GUARD(sm_75-embedding): 'embedding image_embeddings' crashes on GPU on this host (cuDNN: 'GET was unable to find an engine'). Re-run WITHOUT --gpus and with -e CUDA_VISIBLE_DEVICES= (CPU is fast enough: ~100 imgs/min). 'tmm nearest_neighbors' may keep --gpus.",
			};
		}

		// Guard 2 — training writes ~1.9 GB/epoch of checkpoints; require headroom.
		// A truly full disk (0G) must block; only a failed df lets the call through.
		if (cmd.includes("visual_changenet train")) {
			let free: number | undefined;
			try {
				const raw = execSync("df --output=avail -BG / | tail -1", { encoding: "utf8" }).replace(/[^0-9]/g, "");
				if (raw.length > 0) free = parseInt(raw, 10);
			} catch {
				/* if df itself fails, let the call through rather than dead-lock the loop */
			}
			if (free !== undefined && free < 30) {
				return {
					block: true,
					reason: `GUARD(disk): only ${free}G free on /. A 10-epoch ChangeNet training writes ~19G of checkpoints and the run needs headroom (30G minimum). Free space (prune non-selected checkpoints from earlier iterations of THIS run) before training.`,
				};
			}
		}

		// Guard 3 — the ds:aoi container rejects '-e <spec>' for gap_analysis (pure Hydra CLI).
		if (cmd.includes("gap_analysis vcn_aoi") && /\s-e\s+\S+\.ya?ml/.test(cmd)) {
			return {
				block: true,
				reason:
					"GUARD(ds-aoi-cli): the nvcr.io/nvidian/iva/tao-toolkit-ds:aoi image rejects '-e <spec>' for gap_analysis (exits on argparse). Pass pure Hydra overrides (key=value) instead — see your stage card's exact invocation.",
			};
		}

		return undefined;
	});
}
