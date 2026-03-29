"""
inference.py — Baseline agent loop for IncidentCommanderEnv.

Uses OpenAI-compatible client per hackathon requirements.
Reads API credentials from environment variables.
Runs all 3 tasks and produces reproducible baseline scores.

Runtime constraint: < 20 min on 2 vCPU / 8 GB RAM.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


# ── Configuration ──────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

MAX_STEPS_PER_TASK = 35
TEMPERATURE = 0.2
MAX_TOKENS = 1024
TASK_TIMEOUT_SECONDS = 360  # 6 min per task

TASKS = ["oom_crash", "db_pool_exhaustion", "bad_deployment_cascade"]

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

    print(f"  [WARN] Failed to parse action, using fallback. Response: {text[:200]}")
    return FALLBACK_ACTION.copy()


def run_task(
    client: OpenAI,
    task_id: str,
    env_url: str,
) -> Dict[str, Any]:
    """Run a single task episode, return results."""
    print(f"\n{'=' * 60}")
    print(f"TASK: {task_id}")
    print(f"{'=' * 60}")

    # Reset environment
    reset_resp = requests.post(
        f"{env_url}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    reset_data = reset_resp.json()
    observation = reset_data["observation"]

    print(f"Alert: {observation.get('alert', 'N/A')}")

    # Build conversation history
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"INCIDENT ALERT:\n{observation['message']}\n\n"
                f"Service Dependency Graph:\n{json.dumps(observation.get('dependency_graph', {}), indent=2)}\n\n"
                f"Begin investigation. What is your first action?"
            ),
        },
    ]

    total_reward = 0.0
    start_time = time.time()

    for step in range(1, MAX_STEPS_PER_TASK + 1):
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > TASK_TIMEOUT_SECONDS:
            print(f"  [TIMEOUT] Task exceeded {TASK_TIMEOUT_SECONDS}s limit")
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
            print(f"  [ERROR] LLM call failed: {exc}")
            response_text = json.dumps(FALLBACK_ACTION)

        # Parse action
        action = parse_action(response_text)
        action_summary = f"{action['action_type']}({action.get('target_service', '')})"
        print(f"  Step {step}: {action_summary}")

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response_text})

        # Execute action
        try:
            step_resp = requests.post(
                f"{env_url}/step",
                json=action,
                timeout=30,
            )
            step_data = step_resp.json()
        except Exception as exc:
            print(f"  [ERROR] Step failed: {exc}")
            break

        obs = step_data["observation"]
        reward = step_data.get("reward", 0.0)
        done = step_data.get("done", False)
        total_reward += reward

        print(f"    Reward: {reward:+.4f} | Done: {done}")

        # Add observation to conversation
        obs_text = obs.get("message", "")
        if obs.get("error"):
            obs_text += f"\nERROR: {obs['error']}"

        messages.append({
            "role": "user",
            "content": f"Action result:\n{obs_text}\n\nWhat is your next action?",
        })

        # Keep conversation manageable (last 20 messages)
        if len(messages) > 22:
            messages = messages[:1] + messages[-20:]

        if done:
            break

    # Get final state
    try:
        state_resp = requests.get(f"{env_url}/state", timeout=10)
        final_state = state_resp.json()
    except Exception:
        final_state = {}

    final_score = final_state.get("current_score", 0.0)
    elapsed = time.time() - start_time

    print(f"\n  Final Score: {final_score:.4f}")
    print(f"  Steps Used: {final_state.get('step_count', '?')}/{final_state.get('max_steps', '?')}")
    print(f"  Resolved: {final_state.get('incident_resolved', False)}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        "task_id": task_id,
        "score": final_score,
        "steps_used": final_state.get("step_count", 0),
        "resolved": final_state.get("incident_resolved", False),
        "total_reward": round(total_reward, 4),
        "time_seconds": round(elapsed, 1),
    }


def main() -> None:
    """Run all tasks and report scores."""
    print("IncidentCommanderEnv — Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"API: {API_BASE_URL}")
    print(f"Env: {ENV_URL}")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    results = {}
    total_start = time.time()

    for task_id in TASKS:
        result = run_task(client, task_id, ENV_URL)
        results[task_id] = result

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for task_id, result in results.items():
        status = "RESOLVED" if result["resolved"] else "UNRESOLVED"
        print(
            f"  {task_id:30s}  score={result['score']:.4f}  "
            f"steps={result['steps_used']:2d}  {status}  "
            f"time={result['time_seconds']:.0f}s"
        )

    avg_score = sum(r["score"] for r in results.values()) / len(results) if results else 0
    print(f"\n  Average Score: {avg_score:.4f}")
    print(f"  Total Time: {total_time:.0f}s")

    # Write results to file
    output = {
        "model": MODEL_NAME,
        "tasks": results,
        "average_score": round(avg_score, 4),
        "total_time_seconds": round(total_time, 1),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to baseline_results.json")


if __name__ == "__main__":
    main()
