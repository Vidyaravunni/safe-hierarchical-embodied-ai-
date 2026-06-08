"""
LLM Parser Client for Cloud-Hosted NLP Service (v3)

Interfaces with GPU LLM API (GPT-OSS-120B on AMD MI300X) to parse
natural language instructions into structured task_queue action sequences.

API v3 response schema:
  {
    "command_summary": "...",
    "task_queue": [
      {"task_id": 1, "task_name": "...", "actions": [...]},
      ...
    ],
    "actions": [...],   <- flat list of all actions (backward compat)
  }

Endpoints:
  - /parse            -> task_queue + flat actions
  - /cognitive_parse  -> scene-aware intent + candidate plans

API: http://127.0.0.1:9100
"""

import requests
import json
import os
import time
from typing import Any, List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_API_URL = os.getenv("NLP_API_URL", "http://129.212.178.47:9100/parse")
DEFAULT_API_RETRIES = int(os.getenv("NLP_API_RETRIES", "3"))
DEFAULT_API_RETRY_BACKOFF_SEC = float(os.getenv("NLP_API_RETRY_BACKOFF_SEC", "1.0"))


class LLMParser:
    """Client for cloud-hosted LLM instruction parser (v3 task_queue schema)"""

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        timeout: int = 90,
        retries: int = DEFAULT_API_RETRIES,
        retry_backoff_sec: float = DEFAULT_API_RETRY_BACKOFF_SEC,
    ):
        """
        Initialize LLM parser client

        Args:
            api_url: URL of the NLP parser service (wraps the OpenAI-backed local parser API)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        # v3 adds clean + pour
        self.valid_intents = {
            "navigate", "pick", "place", "open", "close",
            "toggle", "slice", "cook", "break", "clean", "pour", "wait",
        }
        self.last_parse_metadata: Dict[str, Any] = {}

    def parse_instruction(self, command: str) -> List[Dict]:
        """
        Parse natural language instruction — returns flat action list.

        Backward-compatible: callers that expect a plain list of actions
        continue to work unchanged.  The API v3 response also includes
        task_queue; use parse_task_queue() to get structured sub-tasks.

        Returns:
            Flat list of all action dicts in execution order.
        """
        self.last_parse_metadata = {}
        response_data = self._post_parse(command)
        self.last_parse_metadata = response_data if isinstance(response_data, dict) else {}

        if isinstance(response_data, dict) and response_data.get("safety_violation"):
            logger.warning(
                "Command blocked by parser safety layer: %s",
                response_data.get("safety_reason", "unsafe command"),
            )
            return []

        # API v3: prefer flat 'actions' field (all tasks flattened)
        if isinstance(response_data, dict):
            if "actions" in response_data:
                actions = response_data["actions"]
            elif "task_queue" in response_data:
                # Flatten task_queue into a single list
                actions = [
                    act
                    for task in response_data["task_queue"]
                    for act in task.get("actions", [])
                ]
            else:
                raise ValueError(f"Unexpected response format: {response_data.keys()}")
        elif isinstance(response_data, list):
            actions = response_data
        else:
            raise ValueError(f"Unexpected response type: {type(response_data)}")

        if not isinstance(actions, list):
            raise ValueError(f"Expected list of actions, got {type(actions)}")

        validated = []
        for i, action in enumerate(actions):
            validated.append(self._validate_action(action, i))

        logger.info(f"Parsed {len(validated)} actions for: '{command}'")
        return validated

    def parse_task_queue(
        self,
        command: str,
        context: str = "",
        scene_objects: Optional[List[str]] = None,
        scene_metadata: str = "",
        parser_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict], str]:
        """
        Parse command and return the structured task_queue.

        Args:
            command: Natural language instruction from the user.
            context: Optional episodic memory context (recent task history)
                     prepended to the command so the LLM can reason about it.

        Returns:
            (command_summary, task_queue, reasoning) where:
              - command_summary: one-sentence description of the goal
              - task_queue: list of {"task_id", "task_name", "actions"}
              - reasoning: LLM chain-of-thought (empty string if not returned)
        """
        # Prepend episodic memory context so the LLM can use it
        full_command = f"{context}\n\nCURRENT TASK: {command}" if context else command

        self.last_parse_metadata = {}
        try:
            response_data = self._post_parse(
                full_command,
                scene_objects=scene_objects,
                scene_metadata=scene_metadata,
                parser_options=parser_options,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            logger.warning("Using local fallback parser: %s: %s", type(exc).__name__, exc)
            # Even with the cloud API down, run the local command-level safety
            # check so unsafe commands are still blocked (not silently dropped).
            from ae_rl.execution.safety import check_command_safety
            is_safe, safety_reason, _safety_meta = check_command_safety(command)
            if not is_safe:
                self.last_parse_metadata = {
                    "safety_violation": True,
                    "safety_reason": safety_reason,
                    "blocked_at": "command_filter",
                }
                logger.warning("Offline safety block: %s", safety_reason)
                return "Unsafe command blocked (offline check)", [], safety_reason
            # Safe command but API is unavailable — re-raise so the caller can
            # surface a clear "LLM unavailable" message rather than a false block.
            raise
        self.last_parse_metadata = response_data if isinstance(response_data, dict) else {}

        summary, task_queue, reasoning = self._extract_plan(response_data, command)

        # If the LLM returned an empty plan due to intermittent refusal
        # (blocked_at == "llm_refusal"), retry once.  Genuine safety blocks
        # (command_filter, open_vocab_guard, plan_validator) are never retried.
        if (
            not task_queue
            and isinstance(response_data, dict)
            and response_data.get("safety_violation")
            and response_data.get("blocked_at") == "llm_refusal"
        ):
            logger.info("LLM refusal detected — retrying parse once for: '%s'", command)
            try:
                response_data = self._post_parse(
                    full_command,
                    scene_objects=scene_objects,
                    scene_metadata=scene_metadata,
                    parser_options=parser_options,
                )
                self.last_parse_metadata = response_data if isinstance(response_data, dict) else {}
                summary, task_queue, reasoning = self._extract_plan(response_data, command)
            except Exception as e:
                logger.warning("Retry parse failed: %s", e)

        return summary, task_queue, reasoning

    def _extract_plan(
        self,
        response_data: Any,
        command: str,
    ) -> Tuple[str, List[Dict], str]:
        """Extract (summary, task_queue, reasoning) from a parse response."""
        summary   = ""
        reasoning = ""
        if isinstance(response_data, dict):
            if response_data.get("safety_violation"):
                summary = response_data.get("command_summary", "Unsafe command blocked")
                reasoning = response_data.get("safety_reason", "")
                logger.warning("Safety violation returned by parser: %s", reasoning)
                return summary, [], reasoning
            summary   = response_data.get("command_summary", "")
            reasoning = response_data.get("reasoning", "")
            if "task_queue" in response_data:
                task_queue = response_data["task_queue"]
                logger.info(
                    f"Parsed task_queue: {len(task_queue)} tasks for: '{command}'"
                )
                return summary, task_queue, reasoning
            # Fallback: wrap flat actions in a single task
            actions = response_data.get("actions", [])
        elif isinstance(response_data, list):
            actions = response_data
        else:
            actions = []

        # Wrap as single task (backward compat / fallback)
        single_task = [{
            "task_id":   1,
            "task_name": "execute",
            "actions":   actions,
        }]
        logger.info(f"Parsed (single-task fallback): {len(actions)} actions")
        return summary, single_task, reasoning

    def _post_parse(
        self,
        command: str,
        scene_objects: Optional[List[str]] = None,
        scene_metadata: str = "",
        parser_options: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Send POST /parse request and return the raw response dict."""
        logger.info(f"Parsing: '{command}'")
        payload = {
            "command": command,
            "scene_objects": scene_objects or [],
            "scene_metadata": scene_metadata or "",
        }
        if parser_options:
            payload.update(parser_options)

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(
                    "LLM API timeout on attempt %d/%d after %ss",
                    attempt, self.retries, self.timeout,
                )
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(
                    "LLM API connection failure on attempt %d/%d: %s",
                    attempt, self.retries, self.api_url,
                )
            except requests.exceptions.HTTPError as e:
                last_error = e
                status_code = e.response.status_code if e.response is not None else "unknown"
                body = e.response.text if e.response is not None else str(e)
                logger.warning(
                    "LLM API HTTP error on attempt %d/%d: %s - %s",
                    attempt, self.retries, status_code, body,
                )
                # Do not retry most client-side errors.
                if e.response is not None and 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    "LLM API returned invalid JSON on attempt %d/%d: %s",
                    attempt, self.retries, e,
                )

            if attempt < self.retries and self.retry_backoff_sec > 0:
                time.sleep(self.retry_backoff_sec * attempt)

        if isinstance(last_error, requests.exceptions.Timeout):
            logger.error(f"Request timeout after {self.retries} attempt(s)")
            raise last_error
        if isinstance(last_error, requests.exceptions.ConnectionError):
            logger.error(f"Cannot connect to LLM API at {self.api_url}")
            raise last_error
        if isinstance(last_error, requests.exceptions.HTTPError):
            raise last_error
        if isinstance(last_error, json.JSONDecodeError):
            raise ValueError(f"Failed to parse JSON response: {last_error}")
        raise RuntimeError("LLM parse request failed for an unknown reason")

    def cognitive_parse(
        self,
        instruction: str,
        scene_objects: Optional[List[str]] = None,
        season: str = "unknown",
        time_of_day: str = "unknown",
        held_object: Optional[str] = None,
        scene_metadata: Optional[str] = None,
    ) -> Dict:
        """
        Parse instruction through cognitive layer for intent + candidate plans.

        Attempts:
          1. POST /cognitive_parse on the NLP API server
          2. Direct call to vLLM with COGNITIVE_SYSTEM_PROMPT
          3. Returns safe default if both fail

        Args:
            instruction: User's natural language instruction
            scene_objects: Objects currently in the AI2-THOR scene
            season: Current season (winter, summer, etc.)
            time_of_day: Time of day (morning, afternoon, evening)
            held_object: Object currently held by agent (or None)

        Returns:
            Dict with intent, confidence, entities, plans, clarification_questions,
            decision_points. Returns safe default on failure.
        """
        default_result = {
            "intent": "unknown",
            "confidence": 0.0,
            "entities": {"requested_object": None, "recipe_name": None, "target_receptacle": None},
            "plans": [],
            "clarification_questions": ["Could you rephrase?"],
            "decision_points": [],
        }

        payload = {
            "instruction": instruction,
            "scene_objects": scene_objects or [],
            "scene_metadata": scene_metadata or "",
            "season": season,
            "time_of_day": time_of_day,
            "held_object": held_object,
        }

        # Call /cognitive_parse endpoint on GPU LLM API server
        cognitive_url = self.api_url.rsplit("/", 1)[0] + "/cognitive_parse"
        try:
            response = requests.post(
                cognitive_url, json=payload, timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict) and "intent" in result:
                logger.info(f"Cognitive parse via API: intent={result['intent']}")
                return result
        except Exception as e:
            logger.error(f"Cognitive parse failed: {e}")

        # LLM unreachable — return unknown so cognitive engine asks for clarification
        logger.warning("Cognitive parse failed, returning unknown")
        return default_result

    def _validate_action(self, action: Dict, index: int) -> Dict:
        """
        Validate individual action schema

        Args:
            action: Action dictionary to validate
            index: Action position in sequence (for error messages)

        Returns:
            Validated action dictionary

        Raises:
            ValueError: On schema violations
        """
        if not isinstance(action, dict):
            raise ValueError(f"Action {index}: Expected dict, got {type(action)}")

        # Validate intent field
        if "intent" not in action:
            raise ValueError(f"Action {index}: Missing required field 'intent'")

        intent = action["intent"]
        if intent not in self.valid_intents:
            raise ValueError(
                f"Action {index}: Invalid intent '{intent}'. "
                f"Valid intents: {self.valid_intents}"
            )

        # Validate intent-specific fields
        if intent == "navigate":
            if "target" not in action:
                raise ValueError(f"Action {index}: Navigate action missing 'target' field")

        elif intent == "pick":
            if "object" not in action:
                raise ValueError(f"Action {index}: Pick action missing 'object' field")

        elif intent == "place":
            if "object" not in action:
                raise ValueError(f"Action {index}: Place action missing 'object' field")
            if "receptacle" not in action:
                raise ValueError(f"Action {index}: Place action missing 'receptacle' field")

        elif intent in ["open", "close", "toggle", "wash"]:
            if "object" not in action:
                raise ValueError(f"Action {index}: {intent} action missing 'object' field")

        return action



# Convenience function for quick usage
def parse_instruction(command: str, api_url: str = DEFAULT_API_URL) -> List[Dict]:
    """
    Quick utility to parse instruction

    Args:
        command: Natural language instruction
        api_url: LLM API endpoint URL

    Returns:
        List of action dictionaries
    """
    parser = LLMParser(api_url)
    return parser.parse_instruction(command)


if __name__ == "__main__":
    # Test the parser
    parser = LLMParser()

    test_commands = [
        "Pick the apple",
        "Search the apple in the kitchen and place it on the countertop",
        "Open the fridge and get the lettuce"
    ]

    print("Testing LLM Parser Client\n" + "="*50)

    for cmd in test_commands:
        print(f"\nCommand: {cmd}")
        try:
            actions = parser.parse_instruction(cmd)
            print(f"Parsed Actions:")
            for i, action in enumerate(actions, 1):
                print(f"  {i}. {action}")
        except Exception as e:
            print(f"Error: {e}")
