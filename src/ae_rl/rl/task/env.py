"""
Task Environment — Gym-like wrapper for task execution RL.

Wraps AI2-THOR to present reset()/step() interface for the Task DDQN.
Uses existing NavigationController and InteractionController as backends.
The RL agent decides WHICH action to take; the controllers handle HOW.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from ae_rl.rl.task.state import TaskStateModel, ACTION_TYPES, ACTION_TO_IDX, GOAL_TO_IDX
from ae_rl.sim.task_reward import TaskStepReward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskEnv:
    """
    Environment wrapper for task execution RL.

    Actions: 10 discrete (pick, place, open, close, toggle, slice, cook, break, navigate, noop)
    State: 88-dim vector from TaskStateModel
    Reward: Per-step from TaskStepReward (milestones + action outcomes)
    """

    ACTIONS = ACTION_TYPES
    ACTION_SPACE = len(ACTIONS)

    # Task definitions
    TASKS = {
        "make_toast": {
            "id": GOAL_TO_IDX["make_toast"],
            "required_objects": ["Bread", "Toaster"],
            "max_steps": 15,
        },
        "make_coffee": {
            "id": GOAL_TO_IDX["make_coffee"],
            "required_objects": ["Mug", "CoffeeMachine"],
            "max_steps": 15,
        },
        "make_omelette": {
            "id": GOAL_TO_IDX["make_omelette"],
            "required_objects": ["Egg", "Pan", "StoveBurner"],
            "max_steps": 25,
        },
        "pick_and_place": {
            "id": GOAL_TO_IDX["pick_and_place"],
            "required_objects": [],
            "max_steps": 10,
        },
        "wash_object": {
            "id": GOAL_TO_IDX["wash_object"],
            "required_objects": ["SinkBasin", "Faucet"],
            "max_steps": 15,
        },
        "slice_object": {
            "id": GOAL_TO_IDX["slice_object"],
            "required_objects": ["Knife"],
            "max_steps": 20,
        },
        "cook_in_pot": {
            "id": GOAL_TO_IDX["general"],
            "required_objects": ["Pot", "StoveBurner"],
            "max_steps": 25,
        },
        "general": {
            "id": GOAL_TO_IDX["general"],
            "required_objects": [],
            "max_steps": 20,
        },
    }

    # Keyword mapping for auto-detecting task_name from instructions/plans
    _TASK_KEYWORDS = {
        "make_toast": ["toast", "bread", "toaster"],
        "make_coffee": ["coffee", "coffeemachine"],
        "make_omelette": ["omelette", "omelet", "fry egg", "egg"],
        "wash_object": ["wash", "clean", "rinse"],
        "slice_object": ["slice", "cut", "chop"],
        "cook_in_pot": ["rasam", "soup", "stew", "curry", "boil", "pot"],
        "pick_and_place": ["pick", "place", "move", "bring", "fetch", "put", "get"],
    }

    def __init__(
        self,
        controller,
        nav_controller,
        interaction_controller,
        task_name: str = "make_toast",
        sequence_length: int = 5,
        plan_completion_threshold: float = 0.75,
        soft_complete_skipped_nav: bool = False,
        max_nav_retries: int = 2,
    ):
        """
        Args:
            controller: AI2-THOR controller
            nav_controller: NavigationController (RL-based)
            interaction_controller: InteractionController
            task_name: Which task to train on
            sequence_length: LSTM window for state sequences
            plan_completion_threshold: Fallback completion ratio threshold
            soft_complete_skipped_nav: Count skipped failed navigate steps as complete
            max_nav_retries: Failed navigate attempts before skipping that target
        """
        self.controller = controller
        self.nav_controller = nav_controller
        self.interaction_controller = interaction_controller
        self.state_model = TaskStateModel()
        self.sequence_length = sequence_length

        # Task config
        self.task_name = task_name
        self.task_config = self.TASKS.get(task_name, self.TASKS["make_toast"])
        self.task_goal_id = self.task_config["id"]
        self.max_steps = self.task_config["max_steps"]

        # Reward model
        self.reward_model = TaskStepReward(task_name)
        self.plan_completion_threshold = float(plan_completion_threshold)
        self.soft_complete_skipped_nav = bool(soft_complete_skipped_nav)

        # Episode state
        self.plan = []
        self.current_step = 0
        self.completed_steps = []
        self.plan_step_idx = 0  # Tracks which LLM plan step we're on
        self.last_action_success = True
        self.last_action_idx = 0
        self.attempt_count = 0
        self.state_history = deque(maxlen=sequence_length)

        # Navigate retry tracking — skip target after 2 failures
        self._nav_fail_counts = {}  # target -> fail count
        self._max_nav_retries = max(1, int(max_nav_retries))

    def configure_task(self, task_name: str = None, instruction: str = None):
        """
        Dynamically reconfigure task type based on instruction or explicit name.

        Args:
            task_name: Explicit task name (e.g. "make_coffee")
            instruction: Natural language instruction to auto-detect task from
        """
        if task_name and task_name in self.TASKS:
            detected = task_name
        elif instruction:
            detected = self._detect_task_from_instruction(instruction)
        else:
            return  # No change

        if detected != self.task_name:
            logger.info(f"Task reconfigured: {self.task_name} -> {detected}")
            self.task_name = detected
            self.task_config = self.TASKS[detected]
            self.task_goal_id = self.task_config["id"]
            self.max_steps = self.task_config["max_steps"]
            self.reward_model = TaskStepReward(detected)

    def _detect_task_from_instruction(self, instruction: str) -> str:
        """Auto-detect task type from instruction keywords."""
        instruction_lower = instruction.lower()
        # Check cooking tasks first (more specific)
        for task_name, keywords in self._TASK_KEYWORDS.items():
            if task_name in ("pick_and_place",):
                continue  # Check generic tasks last
            for kw in keywords:
                if kw in instruction_lower:
                    return task_name
        # Fallback: pick_and_place for fetch/move/bring
        for kw in self._TASK_KEYWORDS["pick_and_place"]:
            if kw in instruction_lower:
                return "pick_and_place"
        return "general"

    def reset(self, plan: List[Dict], scene: str = None) -> np.ndarray:
        """
        Reset environment for new episode.

        Args:
            plan: LLM-suggested action plan
            scene: Optional scene to reset to

        Returns:
            Initial state sequence (1, seq_len, state_dim)
        """
        if scene:
            self.controller.reset(scene=scene)
            self.controller.step(action="Pass")

        self.plan = plan
        self.current_step = 0
        self.plan_step_idx = 0
        self.completed_steps = [False] * len(plan)
        self.last_action_success = True
        self.last_action_idx = 0
        self.attempt_count = 0
        self._nav_fail_counts = {}
        self.reward_model.reset(self.task_name)

        # Fill state history with initial state
        initial_state = self.state_model.get_task_state(
            self.controller, self.task_goal_id, self.plan,
            0, self.completed_steps,
            plan_step_idx=0,
        )
        self.state_history.clear()
        for _ in range(self.sequence_length):
            self.state_history.append(initial_state)

        return self._get_state_sequence()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one task action.

        The action_idx selects ACTION TYPE. The target object comes from
        the LLM plan or task-relevant heuristic.

        Args:
            action_idx: Index into ACTIONS list

        Returns:
            (next_state_sequence, reward, done, info)
        """
        action_type = self.ACTIONS[action_idx]

        # Determine target object
        target = self._resolve_target(action_type)

        # Check if this follows the LLM plan
        follows_plan = self._check_plan_adherence(action_type, target)

        # Execute the action
        success, message = self._execute_action(action_type, target)

        # Update plan tracking
        if follows_plan and success:
            if self.plan_step_idx < len(self.completed_steps):
                self.completed_steps[self.plan_step_idx] = True
            self.plan_step_idx += 1
            self.attempt_count = 0
        elif not success:
            self.attempt_count += 1

        # Check task completion
        task_complete = self._check_task_complete()

        # Compute reward
        reward, reward_breakdown = self.reward_model.compute_step_reward(
            action_type, success, follows_plan,
            self.controller, task_complete,
        )

        # Update episode state
        self.current_step += 1
        self.last_action_success = success
        self.last_action_idx = action_idx

        # Get next state (use plan_step_idx for plan-related encoding)
        next_state = self.state_model.get_task_state(
            self.controller, self.task_goal_id, self.plan,
            self.current_step, self.completed_steps,
            self.last_action_success, self.last_action_idx, self.attempt_count,
            plan_step_idx=self.plan_step_idx,
        )
        self.state_history.append(next_state)

        # Check termination — also stop when all plan steps exhausted
        plan_exhausted = self.plan_step_idx >= len(self.plan) and len(self.plan) > 0
        done = task_complete or self.current_step >= self.max_steps or plan_exhausted

        info = {
            "success": success,
            "message": message,
            "action_type": action_type,
            "target": target,
            "follows_plan": follows_plan,
            "task_complete": task_complete,
            "plan_exhausted": plan_exhausted,
            "plan_completion_ratio": self._plan_completion_ratio(),
            "done_reason": (
                "task_complete" if task_complete else
                "max_steps" if self.current_step >= self.max_steps else
                "plan_exhausted" if plan_exhausted else
                "running"
            ),
            "reward_breakdown": reward_breakdown,
            "step": self.current_step,
            "milestones": list(self.reward_model.achieved_milestones),
        }

        return self._get_state_sequence(), reward, done, info

    def get_llm_suggested_action(self) -> Optional[int]:
        """Get the action index the LLM plan suggests for the current step."""
        if self.plan_step_idx >= len(self.plan):
            return ACTION_TO_IDX.get("noop", 9)  # Plan exhausted → suggest noop
        action = self.plan[self.plan_step_idx]
        intent = action.get("intent", "noop")
        return ACTION_TO_IDX.get(intent, ACTION_TO_IDX["noop"])

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _get_state_sequence(self) -> np.ndarray:
        """Get state history as LSTM input: shape (1, seq_len, state_dim)."""
        seq = np.array(list(self.state_history))
        return seq[np.newaxis, ...]  # Add batch dimension

    def _resolve_target(self, action_type: str) -> str:
        """
        Determine which object to act on.

        Priority:
        1. If action matches LLM plan's current step type, use plan's target
        2. Otherwise, use task-relevant heuristic
        """
        # Check LLM plan
        if self.plan_step_idx < len(self.plan):
            plan_action = self.plan[self.plan_step_idx]
            plan_intent = plan_action.get("intent", "")

            if plan_intent == action_type:
                # Agent follows plan — use plan's target
                if action_type == "place":
                    # For place: target is the RECEPTACLE, not the object
                    target = (plan_action.get("receptacle")
                              or plan_action.get("target", ""))
                elif action_type == "navigate":
                    target = (plan_action.get("target")
                              or plan_action.get("object", ""))
                else:
                    target = (plan_action.get("object")
                              or plan_action.get("target", ""))
                if target:
                    return target

        # Agent deviates — use task-relevant heuristic
        return self._find_relevant_target(action_type)

    def _find_relevant_target(self, action_type: str) -> str:
        """Find the most relevant target for an off-plan action."""
        objects = self.controller.last_event.metadata.get("objects", [])
        required = set(self.task_config.get("required_objects", []))
        inventory = self.controller.last_event.metadata.get("inventoryObjects", [])

        if action_type == "pick":
            # Pick nearest task-required pickupable object
            for obj in objects:
                if (obj["objectType"] in required
                        and obj.get("pickupable", False)
                        and obj.get("visible", False)):
                    return obj["objectType"]
            # Fallback: any visible pickupable
            for obj in objects:
                if obj.get("pickupable", False) and obj.get("visible", False):
                    return obj["objectType"]

        elif action_type == "place":
            # Place on task-relevant receptacle
            if self.task_name == "make_toast":
                return "Toaster"
            elif self.task_name == "make_coffee":
                return "CoffeeMachine"
            elif self.task_name == "make_omelette":
                return "Pan"
            return "CounterTop"

        elif action_type == "toggle":
            if self.task_name == "make_toast":
                return "Toaster"
            elif self.task_name == "make_coffee":
                return "CoffeeMachine"
            elif self.task_name == "make_omelette":
                return "StoveKnob"
            return "Faucet"

        elif action_type == "navigate":
            # Navigate to nearest required object not yet interacted with
            for req in required:
                return req
            return "CounterTop"

        elif action_type == "open":
            for obj in objects:
                if obj.get("openable", False) and not obj.get("isOpen", False):
                    return obj["objectType"]

        elif action_type == "break":
            return "Egg"

        elif action_type == "cook":
            # Cook whatever is on the pan
            for obj in objects:
                if obj.get("cookable", False):
                    return obj["objectType"]
            return "Egg"

        elif action_type == "slice":
            for obj in objects:
                if obj.get("sliceable", False):
                    return obj["objectType"]

        return ""

    def _check_plan_adherence(self, action_type: str, target: str) -> bool:
        """Check if the current action matches the LLM plan suggestion."""
        if self.plan_step_idx >= len(self.plan):
            return False
        plan_action = self.plan[self.plan_step_idx]
        plan_intent = plan_action.get("intent", "")
        return action_type == plan_intent

    def _execute_action(self, action_type: str, target: str) -> Tuple[bool, str]:
        """Execute an action through the existing controllers."""
        if not target and action_type != "noop":
            return False, f"No target for {action_type}"

        if action_type == "navigate":
            # Skip targets that have already failed too many times
            if self._nav_fail_counts.get(target, 0) >= self._max_nav_retries:
                logger.info(f"Skipping navigate to {target} — failed {self._max_nav_retries} times already")
                # Advance plan past this failed navigate step
                if self.plan_step_idx < len(self.plan):
                    plan_intent = self.plan[self.plan_step_idx].get("intent", "")
                    if plan_intent == "navigate":
                        if self.soft_complete_skipped_nav and self.plan_step_idx < len(self.completed_steps):
                            # Navigation can fail due viewpoint quirks while the object
                            # is still reachable by direct interaction; keep curriculum moving.
                            self.completed_steps[self.plan_step_idx] = True
                        self.plan_step_idx += 1
                return False, f"Skipped {target} (unreachable after {self._max_nav_retries} attempts)"

            try:
                success, msg, steps = self.nav_controller.navigate_to_object(
                    target, max_steps=50
                )
                # Ensure msg is a string (nav controller may return numpy array)
                if not isinstance(msg, str):
                    msg = f"Navigated to {target} ({'success' if success else 'failed'})"

                if not success:
                    self._nav_fail_counts[target] = self._nav_fail_counts.get(target, 0) + 1
                    logger.info(f"Navigate to {target} failed ({self._nav_fail_counts[target]}/{self._max_nav_retries})")

                    # Try force pick if close but not visible (object inside container)
                    if self._nav_fail_counts[target] >= self._max_nav_retries:
                        force_picked = self._try_force_pick(target)
                        if force_picked:
                            return True, f"Force-picked {target} from container"

                return success, msg
            except Exception as e:
                self._nav_fail_counts[target] = self._nav_fail_counts.get(target, 0) + 1
                return False, str(e)

        elif action_type == "noop":
            return True, "No operation"

        else:
            # Build action dict for InteractionController
            action_dict = {"intent": action_type}
            if action_type == "place":
                action_dict["object"] = self._get_held_object_type()
                action_dict["receptacle"] = target
            else:
                action_dict["object"] = target

            agent_state = self._get_agent_state()
            try:
                success, msg = self.interaction_controller.execute_interaction(
                    action_dict, self.controller, agent_state
                )
                return success, msg
            except Exception as e:
                return False, str(e)

    def _try_force_pick(self, target: str) -> bool:
        """
        Try to force-pick an object that's close but not visible (e.g. inside Drawer/Cabinet).

        Opens nearby containers first, then attempts forceAction pickup.
        """
        objects = self.controller.last_event.metadata.get("objects", [])
        inventory = self.controller.last_event.metadata.get("inventoryObjects", [])
        if inventory:
            return False  # Already holding something

        # Find the target object
        target_obj = None
        for obj in objects:
            if obj["objectType"] == target:
                target_obj = obj
                break
        if not target_obj:
            return False

        # Check if object is inside a container — open the container first
        parents = target_obj.get("parentReceptacles") or []
        for parent_id in parents:
            for obj in objects:
                if obj["objectId"] == parent_id and obj.get("openable", False) and not obj.get("isOpen", False):
                    logger.info(f"Opening container {obj['objectType']} to access {target}")
                    self.controller.step(action="OpenObject", objectId=parent_id, forceAction=True)

        # Now try force pick
        target_id = target_obj["objectId"]
        if target_obj.get("pickupable", False):
            logger.info(f"Force-picking {target} (objectId={target_id})")
            event = self.controller.step(
                action="PickupObject", objectId=target_id, forceAction=True
            )
            if event.metadata.get("lastActionSuccess", False):
                logger.info(f"Force-picked {target} successfully")
                return True
            else:
                logger.warning(f"Force-pick failed: {event.metadata.get('errorMessage', 'unknown')}")
        return False

    def _get_agent_state(self) -> Dict:
        """Get current agent state for InteractionController."""
        metadata = self.controller.last_event.metadata
        inventory = metadata.get("inventoryObjects", [])
        agent = metadata.get("agent", {})
        return {
            "holding": inventory[0]["objectType"] if inventory else None,
            "position": agent.get("position", {}),
            "visible_objects": [
                obj["objectType"] for obj in metadata.get("objects", [])
                if obj.get("visible", False)
            ],
        }

    def _get_held_object_type(self) -> str:
        """Get the type of object currently held."""
        inventory = self.controller.last_event.metadata.get("inventoryObjects", [])
        if inventory:
            return inventory[0].get("objectType", "")
        return ""

    def _check_task_complete(self) -> bool:
        """Check if the task's success condition is met.

        Two paths to completion:
        1. Physical world state matches goal (gold standard)
        2. >=75% of plan steps followed successfully (learning signal)
        """
        objects = self.controller.last_event.metadata.get("objects", [])
        plan_ratio = self._plan_completion_ratio()

        if self.task_name == "make_toast":
            for obj in objects:
                if obj["objectType"] in ("Bread", "BreadSliced") and obj.get("isCooked", False):
                    return True

        elif self.task_name == "make_coffee":
            for obj in objects:
                if obj["objectType"] == "Mug" and obj.get("isFilledWithLiquid", False):
                    return True
            for obj in objects:
                if obj["objectType"] == "CoffeeMachine" and obj.get("isToggled", False):
                    for mug in objects:
                        if mug["objectType"] in ("Mug", "Cup"):
                            parents = mug.get("parentReceptacles") or []
                            if any("CoffeeMachine" in p for p in parents):
                                return True

        elif self.task_name == "make_omelette":
            for obj in objects:
                if obj["objectType"] in ("Egg", "EggCracked") and obj.get("isCooked", False):
                    return True

        elif self.task_name == "cook_in_pot":
            for obj in objects:
                if obj.get("isCooked", False) and obj["objectType"] not in ("Bread", "BreadSliced"):
                    return True

        elif self.task_name == "wash_object":
            faucet_on = any(o["objectType"] == "Faucet" and o.get("isToggled", False) for o in objects)
            if faucet_on:
                return True

        elif self.task_name == "slice_object":
            for obj in objects:
                if obj.get("isSliced", False):
                    return True

        elif self.task_name in ("pick_and_place", "general"):
            if self._check_physical_placement(objects):
                return True

        # Universal fallback: >=75% plan steps completed = task done
        # This gives the agent a reachable learning signal during training
        if self.completed_steps and plan_ratio >= self.plan_completion_threshold:
            return True

        return False

    def _plan_completion_ratio(self) -> float:
        """Return fraction of plan steps completed (0.0 to 1.0)."""
        if not self.completed_steps:
            return 0.0
        return sum(self.completed_steps) / len(self.completed_steps)

    def _check_physical_placement(self, objects) -> bool:
        """Check if the last plan step's target object is at the right receptacle."""
        for step in reversed(self.plan):
            if step.get("intent") == "place":
                obj_type = step.get("object", "")
                receptacle = step.get("receptacle", "")
                if not obj_type or not receptacle:
                    continue
                for obj in objects:
                    if obj["objectType"] == obj_type:
                        parents = obj.get("parentReceptacles") or []
                        if any(receptacle in p for p in parents):
                            return True
                break
        return False
