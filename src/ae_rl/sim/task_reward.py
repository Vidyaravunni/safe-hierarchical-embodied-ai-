"""
Task Reward Calculator for Interaction Demo

Two reward systems:
1. TaskReward — post-task completion reward (used by interaction_demo.py)
2. TaskStepReward — per-step intermediate rewards (used by RL task execution)
"""

from typing import Dict, List, Set, Tuple


class TaskReward:
    """Calculate rewards for task execution"""

    def __init__(self):
        self.base_success_reward = 10.0
        self.step_penalty = -0.1
        self.action_reward = 1.0

    def compute_task_reward(self, execution_report: dict, parsed_task: dict) -> dict:
        """
        Compute reward for task execution

        Args:
            execution_report: Report from task executor
            parsed_task: Parsed task from NL parser

        Returns:
            Dictionary with reward breakdown
        """
        reward = 0.0
        breakdown = []

        # Check if task succeeded
        success = execution_report.get("success", False)

        if success:
            reward += self.base_success_reward
            breakdown.append(("task_success", self.base_success_reward))

        # Reward for each completed action
        completed = execution_report.get("actions_completed", 0)
        action_reward = completed * self.action_reward
        reward += action_reward
        breakdown.append(("actions_completed", action_reward))

        # Penalty for steps taken
        steps = execution_report.get("total_steps", 0)
        step_cost = steps * self.step_penalty
        reward += step_cost
        breakdown.append(("step_penalty", step_cost))

        # Calculate phases from task plan
        total_phases = len(parsed_task.get("task_plan", []))
        phases_completed = completed  # actions_completed equals phases completed

        return {
            "total_reward": reward,
            "breakdown": breakdown,
            "success": success,
            "phases_completed": phases_completed,
            "total_phases": total_phases
        }


# =============================================================================
# PER-STEP REWARD FOR RL TASK EXECUTION
# =============================================================================

class TaskStepReward:
    """
    Per-step intermediate reward model for task execution RL.

    Reward signals:
    1. Action success/failure
    2. Plan adherence bonus (following LLM suggestion)
    3. Milestone progress (task-specific sub-goals)
    4. Step penalty (efficiency pressure)
    5. Task completion bonus
    """

    # Per-action rewards
    ACTION_REWARDS = {
        "pick_success": 1.5,
        "pick_fail": -0.5,
        "place_success": 1.5,
        "place_fail": -0.5,
        "open_success": 0.5,
        "open_fail": -0.3,
        "close_success": 0.3,
        "close_fail": -0.2,
        "toggle_success": 1.0,
        "toggle_fail": -0.3,
        "slice_success": 1.5,
        "slice_fail": -0.5,
        "cook_success": 2.0,
        "cook_fail": -0.5,
        "break_success": 1.5,
        "break_fail": -0.5,
        "navigate_success": 0.3,
        "navigate_fail": -0.2,
        "noop": -0.5,
    }

    # Meta rewards
    PLAN_ADHERENCE_BONUS = 0.5
    STEP_PENALTY = -0.1
    INVALID_ACTION_PENALTY = -1.0
    TASK_COMPLETE_BONUS = 10.0

    # Task-specific milestones with rewards
    MILESTONES = {
        "make_toast": {
            "bread_picked": 2.0,
            "near_toaster": 1.0,
            "bread_in_toaster": 3.0,
            "toaster_on": 3.0,
            "toast_done": 0.0,  # Covered by TASK_COMPLETE_BONUS
        },
        "make_coffee": {
            "mug_picked": 2.0,
            "near_coffee_machine": 1.0,
            "mug_on_machine": 3.0,
            "machine_on": 3.0,
            "coffee_done": 0.0,
        },
        "make_omelette": {
            "egg_picked": 1.5,
            "egg_on_pan": 2.0,
            "egg_cracked": 2.0,
            "pan_on_stove": 2.0,
            "stove_on": 2.0,
            "egg_cooked": 3.0,
        },
        "cook_in_pot": {
            "potato_picked": 1.5,
            "potato_sliced": 2.0,
            "potato_in_pot": 2.0,
            "pot_on_stove": 2.0,
            "stove_on": 2.0,
        },
        "pick_and_place": {
            "object_picked": 2.0,
            "object_placed": 3.0,
        },
        "wash_object": {
            "object_picked": 2.0,
            "object_in_sink": 2.0,
            "faucet_on": 3.0,
        },
        "slice_object": {
            "knife_picked": 2.0,
            "object_sliced": 3.0,
        },
    }

    def __init__(self, task_name: str = "make_toast"):
        self.task_name = task_name
        self.achieved_milestones: Set[str] = set()

    def reset(self, task_name: str = None):
        """Reset milestone tracking for new episode."""
        if task_name:
            self.task_name = task_name
        self.achieved_milestones.clear()

    def compute_step_reward(
        self,
        action_type: str,
        success: bool,
        follows_plan: bool,
        controller,
        task_complete: bool = False,
    ) -> Tuple[float, Dict]:
        """
        Compute reward for a single task execution step.

        Args:
            action_type: The action taken (pick, place, etc.)
            success: Whether the action succeeded
            follows_plan: Whether this action matches the LLM suggestion
            controller: AI2-THOR controller (for milestone checking)
            task_complete: Whether the full task is now complete

        Returns:
            (total_reward, breakdown_dict)
        """
        reward = 0.0
        breakdown = {}

        # 1. Action success/failure reward
        key = f"{action_type}_{'success' if success else 'fail'}"
        action_reward = self.ACTION_REWARDS.get(key, 0.0)
        reward += action_reward
        breakdown["action"] = action_reward

        # 2. Plan adherence bonus
        if follows_plan and success:
            reward += self.PLAN_ADHERENCE_BONUS
            breakdown["plan_adherence"] = self.PLAN_ADHERENCE_BONUS

        # 3. Milestone rewards
        milestone_reward = self._check_milestones(controller)
        if milestone_reward > 0:
            reward += milestone_reward
            breakdown["milestones"] = milestone_reward

        # 4. Step penalty
        reward += self.STEP_PENALTY
        breakdown["step_penalty"] = self.STEP_PENALTY

        # 5. Task completion bonus
        if task_complete:
            reward += self.TASK_COMPLETE_BONUS
            breakdown["task_complete"] = self.TASK_COMPLETE_BONUS

        return reward, breakdown

    def _check_milestones(self, controller) -> float:
        """
        Check for newly achieved milestones based on environment state.

        Returns total reward from newly achieved milestones.
        """
        milestones = self.MILESTONES.get(self.task_name, {})
        if not milestones:
            return 0.0

        objects = controller.last_event.metadata.get("objects", [])
        inventory = controller.last_event.metadata.get("inventoryObjects", [])
        reward = 0.0

        # Build quick lookup
        obj_by_type = {}
        for obj in objects:
            otype = obj["objectType"]
            if otype not in obj_by_type:
                obj_by_type[otype] = []
            obj_by_type[otype].append(obj)

        held_types = [o["objectType"] for o in inventory]

        # --- Make Toast milestones ---
        if self.task_name == "make_toast":
            if "bread_picked" not in self.achieved_milestones:
                if "Bread" in held_types:
                    self.achieved_milestones.add("bread_picked")
                    reward += milestones["bread_picked"]

            if "bread_in_toaster" not in self.achieved_milestones:
                for bread in obj_by_type.get("Bread", []):
                    parents = bread.get("parentReceptacles") or []
                    if any("Toaster" in p for p in parents):
                        self.achieved_milestones.add("bread_in_toaster")
                        reward += milestones["bread_in_toaster"]
                        break

            if "toaster_on" not in self.achieved_milestones:
                for toaster in obj_by_type.get("Toaster", []):
                    if toaster.get("isToggled", False):
                        self.achieved_milestones.add("toaster_on")
                        reward += milestones["toaster_on"]
                        break

        # --- Make Coffee milestones ---
        elif self.task_name == "make_coffee":
            if "mug_picked" not in self.achieved_milestones:
                if "Mug" in held_types or "Cup" in held_types:
                    self.achieved_milestones.add("mug_picked")
                    reward += milestones["mug_picked"]

            if "mug_on_machine" not in self.achieved_milestones:
                for mug in obj_by_type.get("Mug", []) + obj_by_type.get("Cup", []):
                    parents = mug.get("parentReceptacles") or []
                    if any("CoffeeMachine" in p for p in parents):
                        self.achieved_milestones.add("mug_on_machine")
                        reward += milestones["mug_on_machine"]
                        break

            if "machine_on" not in self.achieved_milestones:
                for cm in obj_by_type.get("CoffeeMachine", []):
                    if cm.get("isToggled", False):
                        self.achieved_milestones.add("machine_on")
                        reward += milestones["machine_on"]
                        break

        # --- Make Omelette milestones ---
        elif self.task_name == "make_omelette":
            if "egg_picked" not in self.achieved_milestones:
                if "Egg" in held_types:
                    self.achieved_milestones.add("egg_picked")
                    reward += milestones["egg_picked"]

            if "egg_on_pan" not in self.achieved_milestones:
                for egg in obj_by_type.get("Egg", []):
                    parents = egg.get("parentReceptacles") or []
                    if any("Pan" in p for p in parents):
                        self.achieved_milestones.add("egg_on_pan")
                        reward += milestones["egg_on_pan"]
                        break

            if "egg_cracked" not in self.achieved_milestones:
                if obj_by_type.get("EggCracked", []):
                    self.achieved_milestones.add("egg_cracked")
                    reward += milestones["egg_cracked"]

            if "stove_on" not in self.achieved_milestones:
                for knob in obj_by_type.get("StoveKnob", []):
                    if knob.get("isToggled", False):
                        self.achieved_milestones.add("stove_on")
                        reward += milestones["stove_on"]
                        break

            if "egg_cooked" not in self.achieved_milestones:
                for egg in obj_by_type.get("EggCracked", []) + obj_by_type.get("Egg", []):
                    if egg.get("isCooked", False):
                        self.achieved_milestones.add("egg_cooked")
                        reward += milestones["egg_cooked"]
                        break

        # --- Cook in Pot milestones ---
        elif self.task_name == "cook_in_pot":
            if "potato_picked" not in self.achieved_milestones:
                if "Potato" in held_types or "PotatoSliced" in held_types:
                    self.achieved_milestones.add("potato_picked")
                    reward += milestones["potato_picked"]

            if "potato_sliced" not in self.achieved_milestones:
                if obj_by_type.get("PotatoSliced", []):
                    self.achieved_milestones.add("potato_sliced")
                    reward += milestones["potato_sliced"]

            if "potato_in_pot" not in self.achieved_milestones:
                for ps in obj_by_type.get("PotatoSliced", []):
                    parents = ps.get("parentReceptacles") or []
                    if any("Pot" in p for p in parents):
                        self.achieved_milestones.add("potato_in_pot")
                        reward += milestones["potato_in_pot"]
                        break

            if "pot_on_stove" not in self.achieved_milestones:
                for pot in obj_by_type.get("Pot", []):
                    parents = pot.get("parentReceptacles") or []
                    if any("StoveBurner" in p for p in parents):
                        self.achieved_milestones.add("pot_on_stove")
                        reward += milestones["pot_on_stove"]
                        break

            if "stove_on" not in self.achieved_milestones:
                for knob in obj_by_type.get("StoveKnob", []):
                    if knob.get("isToggled", False):
                        self.achieved_milestones.add("stove_on")
                        reward += milestones["stove_on"]
                        break

        # --- Pick and Place milestones ---
        elif self.task_name == "pick_and_place":
            if "object_picked" not in self.achieved_milestones:
                if held_types:
                    self.achieved_milestones.add("object_picked")
                    reward += milestones["object_picked"]

            if "object_placed" not in self.achieved_milestones:
                # Check if any object was recently placed (agent no longer holding)
                if not held_types and "object_picked" in self.achieved_milestones:
                    self.achieved_milestones.add("object_placed")
                    reward += milestones["object_placed"]

        # --- Wash Object milestones ---
        elif self.task_name == "wash_object":
            if "object_picked" not in self.achieved_milestones:
                if held_types:
                    self.achieved_milestones.add("object_picked")
                    reward += milestones["object_picked"]

            if "object_in_sink" not in self.achieved_milestones:
                for obj in objects:
                    parents = obj.get("parentReceptacles") or []
                    if any("SinkBasin" in p for p in parents) and obj.get("objectType") not in ("Faucet",):
                        self.achieved_milestones.add("object_in_sink")
                        reward += milestones["object_in_sink"]
                        break

            if "faucet_on" not in self.achieved_milestones:
                for faucet in obj_by_type.get("Faucet", []):
                    if faucet.get("isToggled", False):
                        self.achieved_milestones.add("faucet_on")
                        reward += milestones["faucet_on"]
                        break

        # --- Slice Object milestones ---
        elif self.task_name == "slice_object":
            if "knife_picked" not in self.achieved_milestones:
                if "Knife" in held_types:
                    self.achieved_milestones.add("knife_picked")
                    reward += milestones["knife_picked"]

            if "object_sliced" not in self.achieved_milestones:
                for obj in objects:
                    if obj.get("isSliced", False):
                        self.achieved_milestones.add("object_sliced")
                        reward += milestones["object_sliced"]
                        break

        return reward
