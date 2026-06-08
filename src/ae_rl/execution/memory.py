"""
Agent Memory System - Two Layer Architecture with FIFO Forgetting

Layer 1: Scene Memory (RESET PER SCENE)
    - Tracks completed actions, visited objects, object states
    - Prevents action repetition within a scene
    - FIFO bounded: oldest actions/visits are forgotten when capacity is reached
    - Resets when scene changes

Layer 2: Global Semantic Memory (PERSISTENT)
    - Innate knowledge: hardcoded object-location heuristics (never forgotten)
    - Learned knowledge: discovered locations from experience (FIFO bounded)
    - Persists across all scenes
"""

import logging
from collections import deque, OrderedDict
from typing import Dict, Set, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneMemory:
    """
    Scene-Local Memory - RESETS PER SCENE

    Tracks actions and object states within current scene only.
    """

    def __init__(self, max_actions: int = 50, max_visited: int = 100):
        self.max_actions = max_actions
        self.max_visited = max_visited
        self.forgotten_count = 0
        self.reset()

    def reset(self) -> None:
        """Reset all scene memory (called on scene change)."""
        # FIFO bounded: oldest entries drop off when capacity reached
        self.visited_object_ids: deque = deque(maxlen=self.max_visited)

        # Completed actions as FIFO deque: oldest forgotten first
        self.completed_actions: deque = deque(maxlen=self.max_actions)

        # Failed actions (also FIFO bounded)
        self.failed_actions: deque = deque(maxlen=self.max_actions)

        # Object states: bounded via OrderedDict (FIFO eviction)
        self.object_states: OrderedDict = OrderedDict()
        self._max_object_states = self.max_visited

        # Objects that have been arranged/placed
        self.arranged_objects: Dict[str, str] = {}

        # Currently held object
        self.held_object: Optional[str] = None

        # Current scene name
        self.scene_name: Optional[str] = None

        # Track how much has been forgotten this session
        self.forgotten_count = 0

        logger.info("SceneMemory: Reset (FIFO bounded: actions=%d, visited=%d)",
                     self.max_actions, self.max_visited)

    def set_scene(self, scene_name: str) -> None:
        """Set current scene and reset memory."""
        if self.scene_name != scene_name:
            logger.info(f"SceneMemory: Scene changed from {self.scene_name} to {scene_name}")
            self.reset()
            self.scene_name = scene_name

    def record_action(self, intent: str, object_id: str, success: bool,
                      receptacle_id: str = "") -> None:
        """Record a completed/failed action. FIFO: oldest forgotten if full."""
        action_key = (intent, object_id, receptacle_id)

        if success:
            # deque with maxlen auto-drops oldest when full
            was_full = len(self.completed_actions) == self.completed_actions.maxlen
            self.completed_actions.append(action_key)
            if was_full:
                self.forgotten_count += 1

            if object_id not in self.visited_object_ids:
                self.visited_object_ids.append(object_id)

            self._update_state(intent, object_id, receptacle_id)
            logger.info(f"SceneMemory: Recorded {intent} on {self._name(object_id)}")
        else:
            self.failed_actions.append(action_key)

    def _update_state(self, intent: str, object_id: str, receptacle_id: str) -> None:
        """Update object state based on action. FIFO evicts oldest if full."""
        if object_id not in self.object_states:
            # Evict oldest entry if at capacity
            if len(self.object_states) >= self._max_object_states:
                self.object_states.popitem(last=False)  # FIFO: remove oldest
                self.forgotten_count += 1
            self.object_states[object_id] = {}
        else:
            # Move to end (most recently used)
            self.object_states.move_to_end(object_id)

        state = self.object_states[object_id]

        if intent == "wash":
            state["washed"] = True
        elif intent == "cook":
            state["cooked"] = True
        elif intent == "slice":
            state["sliced"] = True
        elif intent == "open":
            state["opened"] = True
            state["closed"] = False
        elif intent == "close":
            state["opened"] = False
            state["closed"] = True
        elif intent == "toggle":
            state["toggled"] = not state.get("toggled", False)
        elif intent == "pick":
            self.held_object = object_id
            state["picked"] = True
        elif intent == "place":
            self.held_object = None
            state["picked"] = False
            if receptacle_id:
                self.arranged_objects[object_id] = receptacle_id

    def is_action_redundant(self, intent: str, object_id: str,
                            receptacle_id: str = "") -> Tuple[bool, str]:
        """Check if action was already completed."""
        action_key = (intent, object_id, receptacle_id)

        # pick, place, open, close are repeatable actions - only check current state, not history
        # (you can pick an object again after placing it, open/close objects multiple times, etc.)
        if intent not in ("pick", "place", "open", "close", "toggle"):
            # Already completed this exact action (for non-repeatable actions)
            if action_key in self.completed_actions:
                return True, f"Already done: {intent} {self._name(object_id)}"

        state = self.object_states.get(object_id, {})

        # Check specific states
        if intent == "wash" and state.get("washed"):
            return True, f"{self._name(object_id)} already washed"

        if intent == "cook" and state.get("cooked"):
            return True, f"{self._name(object_id)} already cooked"

        if intent == "slice" and state.get("sliced"):
            return True, f"{self._name(object_id)} already sliced"

        if intent == "open" and state.get("opened"):
            return True, f"{self._name(object_id)} already open"

        if intent == "close" and not state.get("opened", True):
            return True, f"{self._name(object_id)} already closed"

        if intent == "pick" and self.held_object == object_id:
            return True, f"Already holding {self._name(object_id)}"

        if intent == "place" and receptacle_id:
            current_loc = self.arranged_objects.get(object_id)
            if current_loc == receptacle_id and self.held_object != object_id:
                return True, f"{self._name(object_id)} already on {self._name(receptacle_id)}"

        return False, ""

    def was_action_failed(self, intent: str, object_id: str,
                          receptacle_id: str = "") -> bool:
        """Check if this action failed before."""
        return (intent, object_id, receptacle_id) in self.failed_actions

    def _name(self, object_id: str) -> str:
        """Extract object name from ID."""
        return object_id.split("|")[0] if "|" in object_id else object_id

    def get_summary(self) -> str:
        """Get memory summary."""
        lines = [f"=== Scene Memory ({self.scene_name}) ==="]
        lines.append(f"Completed actions: {len(self.completed_actions)}/{self.completed_actions.maxlen}")
        lines.append(f"Visited objects: {len(self.visited_object_ids)}/{self.visited_object_ids.maxlen}")
        lines.append(f"Forgotten entries: {self.forgotten_count}")
        if self.held_object:
            lines.append(f"Holding: {self._name(self.held_object)}")
        return "\n".join(lines)


class GlobalSemanticMemory:
    """
    Global Semantic Memory - PERSISTENT ACROSS SCENES

    Two sub-layers:
    - Innate knowledge: hardcoded object-location heuristics (never forgotten)
    - Learned knowledge: discovered locations from experience (FIFO bounded)
    """

    def __init__(self, max_learned_locations: int = 5, max_task_history: int = 100):
        self.max_learned_locations = max_learned_locations

        # INNATE knowledge - common sense, never forgotten
        self.innate_object_locations: Dict[str, List[str]] = {
            "Apple": ["Fridge", "CounterTop", "DiningTable", "Bowl"],
            "Bread": ["CounterTop", "DiningTable", "Toaster"],
            "Egg": ["Fridge", "CounterTop", "Bowl"],
            "Lettuce": ["Fridge", "CounterTop"],
            "Tomato": ["Fridge", "CounterTop", "DiningTable"],
            "Potato": ["Fridge", "CounterTop", "Pot"],
            "Mug": ["CoffeeMachine", "Cabinet", "CounterTop", "Sink"],
            "Cup": ["Cabinet", "CounterTop", "Sink", "DiningTable"],
            "Bowl": ["Cabinet", "CounterTop", "Sink", "DiningTable"],
            "Plate": ["Cabinet", "CounterTop", "Sink", "DiningTable"],
            "Pot": ["StoveBurner", "Cabinet", "CounterTop"],
            "Pan": ["StoveBurner", "Cabinet", "CounterTop"],
            "Knife": ["Drawer", "CounterTop", "DiningTable"],
            "Fork": ["Drawer", "DiningTable"],
            "Spoon": ["Drawer", "DiningTable"],
            "Spatula": ["Drawer", "CounterTop"],
            "SoapBottle": ["Sink", "CounterTop"],
            "SaltShaker": ["DiningTable", "CounterTop"],
            "PepperShaker": ["DiningTable", "CounterTop"],
        }

        # LEARNED knowledge - discovered from experience, FIFO bounded per object
        self.learned_object_locations: Dict[str, deque] = {}

        # Task history - FIFO bounded log of past tasks
        self.task_history: deque = deque(maxlen=max_task_history)

        # Task heuristics (innate)
        self.task_heuristics: Dict[str, List[str]] = {
            "make_coffee": ["navigate_Mug", "pick_Mug", "navigate_CoffeeMachine", "place_Mug", "toggle_CoffeeMachine"],
            "wash_object": ["navigate_object", "pick_object", "navigate_Sink", "toggle_Faucet", "place_Sink"],
            "cook_object": ["navigate_object", "pick_object", "navigate_StoveBurner", "place_StoveBurner", "toggle_StoveKnob"],
        }

    def get_expected_locations(self, object_type: str) -> List[str]:
        """
        Get expected locations for an object type.
        Learned locations have priority (come first), then innate.
        """
        learned = list(self.learned_object_locations.get(object_type, []))
        innate = self.innate_object_locations.get(object_type, ["CounterTop"])

        # Merge: learned first (higher priority), then innate (no duplicates)
        combined = learned[:]
        for loc in innate:
            if loc not in combined:
                combined.append(loc)
        return combined

    def add_object_location(self, object_type: str, location: str) -> None:
        """Learn a new location for an object type (FIFO bounded)."""
        if object_type not in self.learned_object_locations:
            self.learned_object_locations[object_type] = deque(maxlen=self.max_learned_locations)

        locs = self.learned_object_locations[object_type]
        if location not in locs:
            locs.append(location)
            logger.info(f"GlobalMemory: Learned {object_type} can be at {location}")

    def record_task(self, instruction: str, success: bool, actions_count: int) -> None:
        """Record a completed task in history (FIFO bounded)."""
        self.task_history.append({
            "instruction": instruction,
            "success": success,
            "actions": actions_count,
        })

    def get_task_heuristic(self, task_type: str) -> List[str]:
        """Get expected action sequence for a task type."""
        return self.task_heuristics.get(task_type, [])


class AgentMemory:
    """
    Combined Memory System with both layers.

    Usage:
        memory = AgentMemory()
        memory.set_scene("FloorPlan1")  # Resets scene memory

        # Check before action
        redundant, reason = memory.is_action_redundant("wash", "Apple|123")
        if redundant:
            print(f"Skipping: {reason}")
        else:
            # Execute action...
            memory.record_action("wash", "Apple|123", success=True)
    """

    def __init__(self):
        self.scene = SceneMemory()
        self.global_knowledge = GlobalSemanticMemory()

    def set_scene(self, scene_name: str) -> None:
        """Set scene (resets scene memory)."""
        self.scene.set_scene(scene_name)

    def record_action(self, intent: str, object_id: str, success: bool,
                      receptacle_id: str = "") -> None:
        """Record action to scene memory."""
        self.scene.record_action(intent, object_id, success, receptacle_id)

        # Also update global knowledge
        if success and intent == "place" and receptacle_id:
            obj_type = object_id.split("|")[0] if "|" in object_id else object_id
            rec_type = receptacle_id.split("|")[0] if "|" in receptacle_id else receptacle_id
            self.global_knowledge.add_object_location(obj_type, rec_type)

    def is_action_redundant(self, intent: str, object_id: str,
                            receptacle_id: str = "") -> Tuple[bool, str]:
        """Check if action is redundant (already done in this scene)."""
        return self.scene.is_action_redundant(intent, object_id, receptacle_id)

    def get_expected_locations(self, object_type: str) -> List[str]:
        """Get where to look for an object (from global knowledge)."""
        return self.global_knowledge.get_expected_locations(object_type)

    def reset_scene(self) -> None:
        """Reset scene memory only."""
        self.scene.reset()

    def get_summary(self) -> str:
        """Get full memory summary."""
        return self.scene.get_summary()


# Global instance
_memory: Optional[AgentMemory] = None


def get_memory() -> AgentMemory:
    """Get global memory instance."""
    global _memory
    if _memory is None:
        _memory = AgentMemory()
    return _memory


def reset_memory() -> None:
    """Reset global memory."""
    global _memory
    _memory = AgentMemory()


if __name__ == "__main__":
    print("Testing Two-Layer Memory System")
    print("=" * 50)

    memory = AgentMemory()
    memory.set_scene("FloorPlan1")

    # Test 1: Wash apple
    print("\n1. Washing Apple...")
    memory.record_action("wash", "Apple|+01.00|+00.90|-02.00", True)

    # Test 2: Try wash again
    redundant, reason = memory.is_action_redundant("wash", "Apple|+01.00|+00.90|-02.00")
    print(f"   Wash again? Redundant={redundant}")
    print(f"   Reason: {reason}")

    # Test 3: Change scene - should reset
    print("\n2. Changing to FloorPlan2...")
    memory.set_scene("FloorPlan2")

    # Test 4: Wash apple in new scene - should NOT be redundant
    redundant, reason = memory.is_action_redundant("wash", "Apple|+01.00|+00.90|-02.00")
    print(f"   Wash apple in new scene? Redundant={redundant}")

    # Test 5: Global knowledge
    print("\n3. Global Knowledge:")
    print(f"   Where to find Apple: {memory.get_expected_locations('Apple')}")
    print(f"   Where to find Knife: {memory.get_expected_locations('Knife')}")

    print("\n" + memory.get_summary())
