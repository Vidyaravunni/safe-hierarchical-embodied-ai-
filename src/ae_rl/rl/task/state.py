"""
Task State Model — State representation for task execution RL.

Encodes the AI2-THOR environment into a fixed-size vector that the
Task DDQN can consume. Captures: task goal, progress, held object,
nearby objects, affordances, LLM plan hint, and environment flags.
"""

import numpy as np
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Action type indices (must match TaskEnv.ACTIONS)
ACTION_TYPES = ["pick", "place", "open", "close", "toggle",
                "slice", "cook", "break", "navigate", "noop"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
NUM_ACTIONS = len(ACTION_TYPES)

# Task goal indices
TASK_GOALS = ["make_toast", "make_coffee", "make_omelette", "pick_and_place",
              "wash_object", "slice_object", "heat_in_microwave", "general"]
GOAL_TO_IDX = {g: i for i, g in enumerate(TASK_GOALS)}
NUM_GOALS = len(TASK_GOALS)

# Common kitchen objects for one-hot encoding
OBJECT_TYPES = [
    "None", "Apple", "Bread", "BreadSliced", "Egg", "EggCracked",
    "Lettuce", "Tomato", "Potato",
    "Mug", "Cup", "Bowl", "Plate", "Pan", "Knife", "Fork", "Spoon",
    "Pot",
]
OBJ_TO_IDX = {o: i for i, o in enumerate(OBJECT_TYPES)}
NUM_OBJ_SLOTS = len(OBJECT_TYPES)

# Object categories for LLM hint encoding
OBJ_CATEGORIES = ["food", "kitchenware", "appliance", "receptacle", "other"]
CAT_TO_IDX = {c: i for i, c in enumerate(OBJ_CATEGORIES)}
NUM_CATEGORIES = len(OBJ_CATEGORIES)

# Nearby object detection targets
NEARBY_OBJECTS = [
    "Apple", "Bread", "Egg", "Lettuce", "Tomato", "Potato",
    "Mug", "Cup", "Bowl", "Plate", "Pan", "Knife",
    "Toaster", "CoffeeMachine", "Microwave", "Fridge",
    "StoveBurner", "SinkBasin", "CounterTop", "DiningTable",
]
NEARBY_TO_IDX = {o: i for i, o in enumerate(NEARBY_OBJECTS)}
NUM_NEARBY = len(NEARBY_OBJECTS)

# Category mapping for objects
_FOOD_OBJECTS = {"Apple", "Bread", "BreadSliced", "Egg", "EggCracked",
                 "Lettuce", "Tomato", "Potato"}
_KITCHENWARE = {"Mug", "Cup", "Bowl", "Plate", "Pan", "Knife", "Fork",
                "Spoon", "Pot", "Spatula"}
_APPLIANCE = {"Toaster", "CoffeeMachine", "Microwave", "StoveBurner",
              "StoveKnob", "Faucet"}
_RECEPTACLE = {"Fridge", "Cabinet", "Drawer", "SinkBasin", "CounterTop",
               "DiningTable", "Shelf", "GarbageCan"}


def _get_object_category(obj_type: str) -> str:
    if obj_type in _FOOD_OBJECTS:
        return "food"
    elif obj_type in _KITCHENWARE:
        return "kitchenware"
    elif obj_type in _APPLIANCE:
        return "appliance"
    elif obj_type in _RECEPTACLE:
        return "receptacle"
    return "other"


class TaskStateModel:
    """
    State representation for task execution RL.

    State vector layout (~90 dimensions):
      [0:8]   Task goal one-hot
      [8:10]  Progress (current_step/plan_len, current_step/max_steps)
      [10:28] Held object one-hot (18 slots)
      [28:37] Target object affordances (9 flags)
      [37:57] Nearby object flags (20 binary)
      [57:67] Completed steps bitvector (10 slots)
      [67:82] LLM suggested action (10 action type + 5 object category)
      [82:85] Last action result (success, action_id_norm, attempt_count_norm)
      [85:90] Environment flags (is_holding, near_target, target_visible,
              object_cooked, object_sliced)
    """

    TOTAL_STATE_SIZE = 90
    MAX_PLAN_STEPS = 10
    MAX_TASK_STEPS = 30
    INTERACTION_DISTANCE = 1.5  # meters

    def get_task_state(
        self,
        controller,
        task_goal_id: int,
        plan: List[Dict],
        current_step: int,
        completed_steps: List[bool],
        last_action_success: bool = True,
        last_action_idx: int = 0,
        attempt_count: int = 0,
        plan_step_idx: int = None,
    ) -> np.ndarray:
        """
        Build the full task state vector from AI2-THOR metadata.

        Args:
            controller: AI2-THOR controller instance
            task_goal_id: Index into TASK_GOALS
            plan: LLM suggested plan (list of action dicts)
            current_step: Current episode step (for progress normalization)
            completed_steps: Boolean list of completed plan steps
            last_action_success: Whether last action succeeded
            last_action_idx: Index of last action type taken
            attempt_count: Number of attempts on current step
            plan_step_idx: Which plan step we're on (for LLM hint/affordances)

        Returns:
            np.array of shape (TOTAL_STATE_SIZE,)
        """
        # Use plan_step_idx for plan-related encoding, current_step for progress
        if plan_step_idx is None:
            plan_step_idx = current_step
        state = np.zeros(self.TOTAL_STATE_SIZE, dtype=np.float32)
        metadata = controller.last_event.metadata
        objects = metadata.get("objects", [])
        agent = metadata.get("agent", {})
        inventory = metadata.get("inventoryObjects", [])
        agent_pos = agent.get("position", {"x": 0, "y": 0, "z": 0})

        offset = 0

        # --- 1. Task goal one-hot (8 dims) ---
        if 0 <= task_goal_id < NUM_GOALS:
            state[offset + task_goal_id] = 1.0
        offset += NUM_GOALS

        # --- 2. Progress (2 dims) ---
        plan_len = max(len(plan), 1)
        state[offset] = current_step / plan_len
        state[offset + 1] = current_step / self.MAX_TASK_STEPS
        offset += 2

        # --- 3. Held object one-hot (16 dims) ---
        if inventory:
            held_type = inventory[0].get("objectType", "")
            held_idx = OBJ_TO_IDX.get(held_type, 0)
            state[offset + held_idx] = 1.0
        else:
            state[offset] = 1.0  # "None" slot
        offset += NUM_OBJ_SLOTS

        # --- 4. Target object affordances (9 dims) ---
        # Get target from current plan step (use plan_step_idx, not current_step)
        target_obj = None
        if plan_step_idx < len(plan):
            action = plan[plan_step_idx]
            target_name = action.get("object") or action.get("target") or ""
            # Find this object in scene
            for obj in objects:
                if obj["objectType"].lower() == target_name.lower():
                    target_obj = obj
                    break

        if target_obj:
            affordances = [
                target_obj.get("pickupable", False),
                target_obj.get("openable", False),
                target_obj.get("receptacle", False),
                target_obj.get("toggleable", False),
                target_obj.get("cookable", False),
                target_obj.get("sliceable", False),
                target_obj.get("breakable", False),
                target_obj.get("canFillWithLiquid", False),
                target_obj.get("dirtyable", False),
            ]
            for i, aff in enumerate(affordances):
                state[offset + i] = 1.0 if aff else 0.0
        offset += 9

        # --- 5. Nearby object flags (20 dims) ---
        for obj in objects:
            otype = obj["objectType"]
            if otype in NEARBY_TO_IDX:
                # Check if within interaction distance
                obj_pos = obj.get("position", {})
                dx = obj_pos.get("x", 0) - agent_pos.get("x", 0)
                dz = obj_pos.get("z", 0) - agent_pos.get("z", 0)
                dist = (dx**2 + dz**2) ** 0.5
                if dist < self.INTERACTION_DISTANCE:
                    state[offset + NEARBY_TO_IDX[otype]] = 1.0
        offset += NUM_NEARBY

        # --- 6. Completed steps bitvector (10 dims) ---
        for i, done in enumerate(completed_steps[:self.MAX_PLAN_STEPS]):
            if done:
                state[offset + i] = 1.0
        offset += self.MAX_PLAN_STEPS

        # --- 7. LLM suggested action (15 dims = 10 action + 5 category) ---
        if plan_step_idx < len(plan):
            action = plan[plan_step_idx]
            intent = action.get("intent", "noop")
            action_idx = ACTION_TO_IDX.get(intent, ACTION_TO_IDX["noop"])
            state[offset + action_idx] = 1.0

            # Object category
            obj_name = action.get("object") or action.get("target") or ""
            cat = _get_object_category(obj_name)
            cat_idx = CAT_TO_IDX.get(cat, CAT_TO_IDX["other"])
            state[offset + NUM_ACTIONS + cat_idx] = 1.0
        offset += NUM_ACTIONS + NUM_CATEGORIES

        # --- 8. Last action result (3 dims) ---
        state[offset] = 1.0 if last_action_success else 0.0
        state[offset + 1] = last_action_idx / max(NUM_ACTIONS - 1, 1)
        state[offset + 2] = min(attempt_count / 3.0, 1.0)
        offset += 3

        # --- 9. Environment flags (5 dims) ---
        is_holding = 1.0 if inventory else 0.0
        state[offset] = is_holding

        # Near target
        if target_obj:
            t_pos = target_obj.get("position", {})
            dx = t_pos.get("x", 0) - agent_pos.get("x", 0)
            dz = t_pos.get("z", 0) - agent_pos.get("z", 0)
            dist = (dx**2 + dz**2) ** 0.5
            state[offset + 1] = 1.0 if dist < self.INTERACTION_DISTANCE else 0.0
            state[offset + 2] = 1.0 if target_obj.get("visible", False) else 0.0
            state[offset + 3] = 1.0 if target_obj.get("isCooked", False) else 0.0
            state[offset + 4] = 1.0 if target_obj.get("isSliced", False) else 0.0
        offset += 5

        return state
