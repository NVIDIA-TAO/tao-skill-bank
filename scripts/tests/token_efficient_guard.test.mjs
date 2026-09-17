// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run with Node >=22.19 and Pi 0.85.1+ installed locally, or set PI_TEST_RUNTIME
// to an isolated npm prefix. No model calls, GPU jobs, or real keys are used.
import assert from "node:assert/strict";
import { test, after } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { stripTypeScriptTypes } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const entry = process.env.PI_TEST_RUNTIME
	? pathToFileURL(path.join(process.env.PI_TEST_RUNTIME,
		"node_modules/@earendil-works/pi-coding-agent/dist/index.js")).href
	: import.meta.resolve("@earendil-works/pi-coding-agent");
const source = fs.readFileSync(path.join(repo,
	"skills/core/tao-token-efficient-execution/adapters/pi/guard.ts"), "utf8");
const compiled = stripTypeScriptTypes(source).replaceAll(
	'"@earendil-works/pi-coding-agent"', JSON.stringify(entry));
const { default: register, toolEnvironment, supportsSafeBash } = await import(
	"data:text/javascript;base64," + Buffer.from(compiled).toString("base64"));
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "tao-guard-test-"));
after(() => fs.rmSync(scratch, { recursive: true, force: true }));

function harness(budget = 0) {
	process.env.PI_KIT_TURN_BUDGET = String(budget);
	let handler, bash, aborted = false;
	register({
		on: (name, fn) => { if (name === "tool_call") handler = fn; },
		registerTool: (tool) => { bash = tool; },
	});
	return {
		call: (toolName, input) => handler({ toolName, input },
			{ cwd: scratch, abort: () => { aborted = true; } }),
		bash,
		aborted: () => aborted,
	};
}

for (const command of [
	"env", "true; env", "true && printenv", "bash -c 'env'", "/usr/bin/env",
	"true\nexport -p", "true; declare -p", "true; set",
	"cat /example/.env", "cat .env", "cat .aws/credentials", "cat /example/.aws/credentials",
	"cat /proc/self/environ", 'echo "$NVIDIA_INFERENCE_API_KEY"',
]) {
	test("blocks credential request: " + command, async () => {
		assert.equal((await harness().call("bash", { command }))?.block, true);
	});
}

for (const command of [
	"bash -c 'set -e; echo ready'", "set -eu; echo ready",
	"echo $WS $RD $ITER", "ls /tmp/workspace/results",
]) {
	test("allows normal card command: " + command, async () => {
		assert.equal(await harness().call("bash", { command }), undefined);
	});
}

for (const toolName of ["read", "edit", "write"]) {
	test(toolName + " protects credential files and symlinks", async () => {
		const secret = path.join(scratch, ".env");
		fs.writeFileSync(secret, "DUMMY=fixture");
		const alias = path.join(scratch, "alias-" + toolName);
		fs.symlinkSync(secret, alias);
		const h = harness();
		for (const target of [secret, alias, "/example/.ssh/id_ed25519", "/example/.pi/agent/auth.json"]) {
			assert.equal((await h.call(toolName, { path: target }))?.block, true);
		}
		assert.equal(await h.call(toolName, { path: path.join(scratch, "stage.md") }), undefined);
	});
}

test("budget counts file tools and aborts rather than looping on blocked calls", async () => {
	const h = harness(12);
	for (let i = 0; i < 12; i++) await h.call("read", { path: path.join(scratch, "stage.md") });
	assert.equal((await h.call("read", { path: path.join(scratch, "stage.md") }))?.block, true);
	assert.equal(h.aborted(), true);
});

test("tool environment preserves card constants, excludes unknown variables and startup hooks", () => {
	assert.deepEqual(toolEnvironment({
		WS: "/work", RD: "/work/run", PATH: "/usr/bin:/bin",
		NVIDIA_INFERENCE_API_KEY: "dummy", UNKNOWN_CREDENTIAL: "dummy",
		BASH_ENV: "/example/startup.sh", NODE_OPTIONS: "--inspect",
	}), { WS: "/work", RD: "/work/run", PATH: "/usr/bin:/bin" });
});

test("safe bash requires a supported runtime version", () => {
	for (const version of ["0.84.0", "0.85.0", "0.85.1-beta", "unknown"]) assert.equal(supportsSafeBash(version), false);
	for (const version of ["0.85.1", "0.86.0", "1.0.0"]) assert.equal(supportsSafeBash(version), true);
});

test("real bash subprocess cannot inherit dummy provider credentials", async () => {
	// Run this test process with a clean environment; these are only canaries.
	process.env.NVIDIA_INFERENCE_API_KEY = "test-only-provider-canary";
	process.env.UNKNOWN_CREDENTIAL = "test-only-other-canary";
	process.env.WS = "/test/workspace";
	const startup = path.join(scratch, "startup.sh");
	fs.writeFileSync(startup, "echo startup-hook-ran\n");
	process.env.BASH_ENV = startup;
	try {
		const h = harness();
		const result = await h.bash.execute("test", {
			command: "node -e 'if(process.env.NVIDIA_INFERENCE_API_KEY || process.env.UNKNOWN_CREDENTIAL) process.exit(91); process.stdout.write(process.env.WS)'",
		});
		assert.equal(result.content.map(c => c.text ?? "").join(""), "/test/workspace");
		assert.equal(process.env.NVIDIA_INFERENCE_API_KEY, "test-only-provider-canary");
	} finally {
		delete process.env.NVIDIA_INFERENCE_API_KEY;
		delete process.env.UNKNOWN_CREDENTIAL;
		delete process.env.BASH_ENV;
		delete process.env.WS;
	}
});
