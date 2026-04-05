"""
inference.py -- Baseline agent for IncidentCommanderEnv.

Uses OpenAI-compatible client per hackathon requirements.
Reads API credentials from environment variables (or .env file).
Runs all 3 tasks and produces reproducible baseline scores.

STDOUT FORMAT (mandatory for evaluation):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Runtime constraint: < 20 min on 2 vCPU / 8 GB RAM.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


# -- Load .env file (no extra dependency needed) ---------------------------
def _load_dotenv(path: str = ".env") -> None:
    """Load key=value pairs from a .env file into os.environ."""
    env_path = Path(path)
    if not env_path.exists():
        env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not os.getenv(key):  # don't override existing env vars
            os.environ[key] = value


_load_dotenv()

# -- Configuration ---------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "https://hype4raj-incident-commander-env.hf.space")

BENCHMARK = "incident_commander_env"
TASKS = ["oom_crash", "db_pool_exhaustion", "bad_deployment_cascade"]

MAX_STEPS_PER_TASK = 35
TEMPERATURE = 0.2
MAX_TOKENS = 1024
TASK_TIMEOUT_SECONDS = 360  # 6 min per task


# -- Structured Logging (hackathon evaluation format) ----------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# -- System Prompt ---------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) responding to a production incident in a microservices cluster.

You must diagnose the issue and resolve it by taking actions. Always respond with a single JSON object (no markdown, no explanation outside the JSON):

{
  "thinking": "Brief reasoning about what to do next",
  "action_type": "one of: list_services, describe_service, read_logs, check_metrics, restart_service, scale_service, rollback_deployment, run_diagnostic, update_config, resolve_incident",
  "target_service": "service-name or null",
  "parameters": {}
}

Available actions:
- list_services: Get overview of all services (no target needed)
- describe_service: Get detailed info for a service (config, deployment history, dependencies)
- read_logs: Read service logs. params: {lines: 50, severity: "ERROR"}
- check_metrics: Check CPU, memory, latency, error rate. params: {metric: "all"}
- restart_service: Restart a service. params: {memory_limit: "512Mi"} (optional)
- scale_service: Change replicas. params: {replicas: 3}
- rollback_deployment: Rollback to version. params: {to_version: "v2.3.1"}
- run_diagnostic: Run diagnostic. params: {command: "check_connectivity|check_health|check_resources"}
- update_config: Update config. params: {key: "db.pool.max_size", value: 50}
- resolve_incident: Declare resolved. params: {root_cause: "...", resolution: "..."}

Strategy:
1. Start with list_services to see the cluster state
2. Investigate unhealthy services: read_logs, check_metrics, describe_service
3. Trace the dependency chain to find the root cause
4. Apply the correct fix (restart, rollback, config change)
5. Call resolve_incident when done

IMPORTANT: Don't restart services unless you've diagnosed the issue. Rollback is preferred over restart when a bad deployment is the cause."""

FALLBACK_ACTION = {
    "action_type": "list_services",
    "target_service": None,
    "parameters": {},
}


# -- Action Parsing --------------------------------------------------------

def parse_action(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response into an action dict."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        return {
            "action_type": data.get("action_type", "list_services"),
            "target_service": data.get("target_service"),
            "parameters": data.get("parameters", {}),
        }
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
                return {
                    "action_type": data.get("action_type", "list_services"),
                    "target_service": data.get("target_service"),
                    "parameters": data.get("parameters", {}),
                }
            except json.JSONDecodeError:
                pass

    print(f"[DEBUG] Failed to parse action, using fallback. Response: {text[:200]}", flush=True)
    return FALLBACK_ACTION.copy()


def format_action_string(action: Dict[str, Any]) -> str:
    """Format action as a human-readable string for [STEP] logs."""
    action_type = action.get("action_type", "list_services")
    target = action.get("target_service") or ""
    return f"{action_type}({target})"


# -- Task Runner -----------------------------------------------------------

def run_task(client: OpenAI, task_id: str, env_url: str) -> float:
    """Run a single task episode with structured logging. Returns grader score."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Reset environment
        reset_resp = requests.post(
            f"{env_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        reset_data = reset_resp.json()
        observation = reset_data["observation"]
        max_steps = reset_data.get("info", {}).get("max_steps", MAX_STEPS_PER_TASK)

        # Build conversation history
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"INCIDENT ALERT:\n{observation['message']}\n\n"
                    f"Service Dependency Graph:\n"
                    f"{json.dumps(observation.get('dependency_graph', {}), indent=2)}\n\n"
                    f"Begin investigation. What is your first action?"
                ),
            },
        ]

        start_time = time.time()

        for step in range(1, max_steps + 1):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > TASK_TIMEOUT_SECONDS:
                print(f"[DEBUG] Task {task_id} exceeded {TASK_TIMEOUT_SECONDS}s timeout", flush=True)
                break

            # Call LLM
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"[DEBUG] LLM call failed: {exc}", flush=True)
                response_text = json.dumps(FALLBACK_ACTION)

            # Parse action
            action = parse_action(response_text)
            action_string = format_action_string(action)

            # Add assistant response to history
            messages.append({"role": "assistant", "content": response_text})

            # Execute action
            step_resp = requests.post(
                f"{env_url}/step",
                json=action,
                timeout=30,
            )
            step_data = step_resp.json()

            obs = step_data["observation"]
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            error = obs.get("error")

            rewards.append(reward)
            steps_taken = step

            # Emit structured log immediately after env.step()
            log_step(step=step, action=action_string, reward=reward, done=done, error=error)

            # Add observation to conversation
            obs_text = obs.get("message", "")
            if obs.get("error"):
                obs_text += f"\nERROR: {obs['error']}"

            messages.append({
                "role": "user",
                "content": f"Action result:\n{obs_text}\n\nWhat is your next action?",
            })

            # Keep conversation manageable (last 20 messages + system prompt)
            if len(messages) > 22:
                messages = messages[:1] + messages[-20:]

            if done:
                break

        # Get final state with grader score
        state_resp = requests.get(f"{env_url}/state", timeout=10)
        final_state = state_resp.json()
        score = final_state.get("current_score", 0.0)
        success = final_state.get("incident_resolved", False)
        steps_taken = final_state.get("step_count", steps_taken)

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# -- Main ------------------------------------------------------------------

def main() -> None:
    """Run all tasks and report scores via structured logs."""
    print(f"[DEBUG] IncidentCommanderEnv Baseline Inference", flush=True)
    print(f"[DEBUG] Model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] API: {API_BASE_URL}", flush=True)
    print(f"[DEBUG] Env: {ENV_URL}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    scores: Dict[str, float] = {}

    for task_id in TASKS:
        scores[task_id] = run_task(client, task_id, ENV_URL)

    # Summary (debug output only -- does not interfere with structured logs)
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"[DEBUG] === RESULTS SUMMARY ===", flush=True)
    for task_id, task_score in scores.items():
        print(f"[DEBUG]   {task_id}: {task_score:.3f}", flush=True)
    print(f"[DEBUG]   Average: {avg_score:.3f}", flush=True)


if __name__ == "__main__":
    main()
