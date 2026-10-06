// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * OpenAI-compatible "nim" provider for Pi.
 *
 * Defaults to the NVIDIA-internal Inference API so the kit can run on
 * Nemotron/Qwen instead of Anthropic:
 *
 *   pi --model "nim/nvidia/nvidia/Nemotron-3-Nano-30B-A3B:off" ...
 *
 * Everything is configurable from the environment (read once at extension
 * load), so a self-hosted NIM container, vLLM/SGLang server, or any other
 * OpenAI-compatible gateway works without editing this file:
 *
 *   PI_KIT_NIM_BASE_URL      endpoint incl. /v1, not /chat/completions
 *                            (alias: NVIDIA_INFERENCE_BASE_URL)
 *   PI_KIT_NIM_API_KEY_VAR   NAME of the env var holding the key
 *                            (default NVIDIA_INFERENCE_API_KEY)
 *   PI_KIT_NIM_MODELS        comma-separated extra model ids; pack drivers
 *                            append the id from MODEL=nim/<id> automatically
 *   PI_KIT_NIM_CONTEXT_WINDOW / PI_KIT_NIM_MAX_TOKENS
 *                            shared limits (default 131072 / 16384)
 *
 * The key value is NEVER stored in this repo: Pi resolves it from the named
 * variable at request time.
 *
 * Default-endpoint facts (verified 2026-07-22/23 via curl, all default ids):
 *  - /v1/chat/completions, Bearer auth, model field is the full id string
 *  - tool calling works with thinking off; max_tokens 16384 accepted
 *  - thinking is ON server-side by default and is disabled through
 *    chat_template_kwargs.enable_thinking (the `:off` model-ref suffix)
 *  - usage reports prompt/completion tokens only (no cache accounting)
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PROVIDER = "nim";
const DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1";
const DEFAULT_API_KEY_VAR = "NVIDIA_INFERENCE_API_KEY";
const DEFAULT_MODELS = [
	"nvidia/nvidia/Nemotron-3-Nano-30B-A3B",
	"nvidia/qwen/qwen3.6-35b-a3b",
	"nvidia/qwen/qwen3-5-397b-a17b",
	"nvidia/nvidia/nemotron-3-super-v3",
];

const env = (name: string) => process.env[name]?.trim() || "";
const positiveInt = (name: string, fallback: number) => {
	const value = Number.parseInt(env(name), 10);
	return Number.isFinite(value) && value > 0 ? value : fallback;
};

const BASE_URL = (env("PI_KIT_NIM_BASE_URL") || env("NVIDIA_INFERENCE_BASE_URL") || DEFAULT_BASE_URL).replace(/\/+$/, "");
const API_KEY_VAR = /^[A-Za-z_][A-Za-z0-9_]*$/.test(env("PI_KIT_NIM_API_KEY_VAR"))
	? env("PI_KIT_NIM_API_KEY_VAR")
	: DEFAULT_API_KEY_VAR;
const MODEL_IDS = [...new Set([...DEFAULT_MODELS, ...env("PI_KIT_NIM_MODELS").split(",").map((id) => id.trim())])]
	.filter(Boolean);
const CONTEXT_WINDOW = positiveInt("PI_KIT_NIM_CONTEXT_WINDOW", 131072);
const MAX_TOKENS = positiveInt("PI_KIT_NIM_MAX_TOKENS", 16384);

export default function (pi: ExtensionAPI) {
	// Greedy decoding for executor work: at default sampling the nano model
	// sometimes SIMULATES tool output in prose instead of calling the tool
	// (observed ~50% on trivial prompts). temperature 0 pins it to the
	// tool-calling path. Applied to every model on this provider only.
	pi.on("before_provider_request", (event, ctx) => {
		const p = event.payload as Record<string, unknown> | null;
		if (ctx.model?.provider === PROVIDER && p && typeof p === "object") {
			return { ...p, temperature: 0 };
		}
		return undefined;
	});

	pi.registerProvider(PROVIDER, {
		name: "NVIDIA Inference API (OpenAI-compatible)",
		baseUrl: BASE_URL,
		apiKey: `$${API_KEY_VAR}`,
		api: "openai-completions",
		models: MODEL_IDS.map((id) => ({
			id,
			name: id,
			// reasoning MUST be true for pi to emit chat_template_kwargs at all
			// (pi-ai openai-completions.js:518). Thinking is then controlled by
			// the model-ref suffix: `nim/...:off` -> enable_thinking:false.
			// With thinking on, the default endpoint burns the whole completion
			// budget on reasoning (measured 29.9k chars, no tool call) — always
			// pass :off for card execution.
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: CONTEXT_WINDOW,
			maxTokens: MAX_TOKENS,
			compat: {
				supportsDeveloperRole: false,
				supportsReasoningEffort: false,
				maxTokensField: "max_tokens",
				supportsUsageInStreaming: true,
				thinkingFormat: "chat-template",
				chatTemplateKwargs: { enable_thinking: { $var: "thinking.enabled" } },
			},
		})),
	});
}
