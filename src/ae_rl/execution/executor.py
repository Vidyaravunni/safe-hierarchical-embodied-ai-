"""
Task Executor v3 — Sequential Task-Queue Executor
===================================================
Executes one logical sub-task at a time from the task_queue returned by
the v3 NLP parser API.

Flow:
  task_queue = [task1, task2, task3, ...]
  for each task:
      execute all actions in task.actions sequentially
      if task succeeds → move to next task
      if task fails    → stop and report (hard failure)

Architecture:
  - Navigation actions  → NavigationController  (RL DDQN)
  - Interaction actions → InteractionRLController (RL DDQN)
  - Safety checks       → auto-navigate before interaction if agent is far
  - Container handling  → auto-open/close Fridge, Cabinet, Drawer

This replaces TaskExecutorWithRL as the top-level executor for v3 plans.
It keeps TaskExecutorWithSafetyChecks behaviour but operates on the
task_queue structure rather than a flat action list.
"""

import logging
import re
import time
from typing import List, Dict, Tuple, Optional

from ae_rl.rl.nav.controller import NavigationController
from ae_rl.rl.interaction.controller import InteractionRLController
from ae_rl.execution.safety import check_action_safety

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Result types ─────────────────────────────────────────────────────────────

class TaskResult:
    """Result of a single sub-task execution."""
    def __init__(self, task_id: int, task_name: str):
        self.task_id          = task_id
        self.task_name        = task_name
        self.success          = False
        self.actions_done     = 0
        self.actions_failed   = 0
        self.object_not_found = 0   # steps failed due to object absent from scene
        self.invalid_actions  = 0   # steps failed due to bad affordance / precondition
        self.elapsed          = 0.0
        self.step_log: List[Dict] = []
        self.error_message    = ""

    def to_dict(self) -> Dict:
        return {
            "task_id":          self.task_id,
            "task_name":        self.task_name,
            "success":          self.success,
            "actions_done":     self.actions_done,
            "actions_failed":   self.actions_failed,
            "object_not_found": self.object_not_found,
            "invalid_actions":  self.invalid_actions,
            "elapsed":          round(self.elapsed, 2),
            "step_log":         self.step_log,
            "error":            self.error_message,
        }


class QueueReport:
    """Report for the entire task_queue execution."""
    def __init__(self, command: str):
        self.command           = command
        self.success           = False
        self.tasks_completed   = 0
        self.tasks_failed      = 0
        self.total_elapsed     = 0.0
        self.task_results: List[TaskResult] = []
        self.error_message     = ""

    def to_dict(self) -> Dict:
        return {
            "command":          self.command,
            "success":          self.success,
            "tasks_completed":  self.tasks_completed,
            "tasks_failed":     self.tasks_failed,
            "total_elapsed":    round(self.total_elapsed, 2),
            "task_results":     [r.to_dict() for r in self.task_results],
            "error":            self.error_message,
        }

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"QueueReport [{status}] '{self.command}'\n"
            f"  Tasks: {self.tasks_completed} done / {self.tasks_failed} failed\n"
            f"  Time : {self.total_elapsed:.1f}s"
        )


# ─── Sequential Task-Queue Executor ──────────────────────────────────────────

class TaskExecutorV3:
    """
    Executes a task_queue produced by the v3 LLM parser.

    One sub-task at a time: the executor waits for each task to fully
    succeed (or fail) before starting the next.

    Navigation  → NavigationController  (RL DDQN)
    Interaction → InteractionRLController (RL DDQN, online learning)
    """

    # Distance threshold (metres) to decide whether a safety auto-nav is needed
    INTERACTION_DISTANCE = 1.5

    def __init__(
        self,
        nav_controller:         NavigationController,
        interaction_controller: InteractionRLController,
        controller,                      # AI2-THOR controller
        max_nav_steps:          int = 100,
        continue_on_task_failure: bool = False,
        enable_queue_repair:    bool = True,
    ):
        """
        Args:
            nav_controller:           RL navigation controller
            interaction_controller:   RL interaction controller
            controller:               AI2-THOR controller instance
            max_nav_steps:            Max steps per navigate action
            continue_on_task_failure: If True, skip failed task and continue
            enable_queue_repair:      If False, execute raw planner output
        """
        self.nav_ctrl    = nav_controller
        self.int_ctrl    = interaction_controller
        self.controller  = controller
        self.max_nav_steps          = max_nav_steps
        self.continue_on_fail       = continue_on_task_failure
        self.enable_queue_repair    = enable_queue_repair
        self.queue_history: List[QueueReport] = []
        self.safety_stats: Dict[str, int] = {}
        self._reset_safety_stats()
        # Tracks auto-repairs inserted by prepare_task_queue() per call
        self.repair_log: List[str] = []

    # ── Main entry point ──────────────────────────────────────────────────────

    def execute_queue(
        self,
        task_queue: List[Dict],
        command:    str = "",
        verbose:    bool = True,
    ) -> QueueReport:
        """
        Execute a full task_queue, one sub-task at a time.

        Args:
            task_queue: List of {"task_id", "task_name", "actions"} dicts
            command:    Original NL command (for reporting)
            verbose:    Print progress to stdout

        Returns:
            QueueReport summarising outcome for all tasks.
        """
        report      = QueueReport(command)
        queue_start = time.time()
        self._reset_safety_stats()
        task_queue = self.prepare_task_queue(task_queue)

        if verbose:
            print(f"\n  ═══ Executing {len(task_queue)} tasks for: \"{command}\" ═══")

        for task in task_queue:
            task_id   = task.get("task_id",   "?")
            task_name = task.get("task_name", "task")
            actions   = task.get("actions",   [])

            if verbose:
                print(f"\n  ── Task {task_id}: {task_name}  ({len(actions)} actions) ──")

            task_result = self._execute_task(task_id, task_name, actions, verbose)
            report.task_results.append(task_result)

            if task_result.success:
                report.tasks_completed += 1
                if verbose:
                    print(f"  ✓ Task {task_id} '{task_name}' completed in {task_result.elapsed:.1f}s")
            else:
                report.tasks_failed += 1
                report.error_message = (
                    f"Task {task_id} '{task_name}' failed: {task_result.error_message}"
                )
                if verbose:
                    print(f"  ✗ Task {task_id} '{task_name}' FAILED: {task_result.error_message}")

                if not self.continue_on_fail:
                    break   # Stop queue on first task failure

        report.total_elapsed = time.time() - queue_start
        report.success = (report.tasks_failed == 0 and
                          report.tasks_completed == len(task_queue))

        if verbose:
            status = "SUCCESS" if report.success else "FAILED"
            print(f"\n  ═══ Queue {status}: "
                  f"{report.tasks_completed}/{len(task_queue)} tasks done "
                  f"in {report.total_elapsed:.1f}s ═══")

        self.queue_history.append(report)
        return report

    # ── Single task execution ─────────────────────────────────────────────────

    def _execute_task(
        self,
        task_id:   int,
        task_name: str,
        actions:   List[Dict],
        verbose:   bool,
    ) -> TaskResult:
        """Execute all actions in a single sub-task sequentially."""
        result     = TaskResult(task_id, task_name)
        task_start = time.time()

        for i, action in enumerate(actions):
            intent = action.get("intent", "")

            if verbose:
                obj = action.get("target", action.get("object", ""))
                rec = f" → {action['receptacle']}" if "receptacle" in action else ""
                extra = ""
                if intent == "toggle" and "knob_index" in action:
                    extra = f" [knob:{action['knob_index']}]"
                elif intent == "cook":
                    extra = f" [{action.get('device','?')} / {action.get('container','?')}]"
                elif intent == "wait":
                    extra = f" [{action.get('seconds',3)}s]"
                print(f"    [{i+1:2d}/{len(actions)}] {intent:<10s} {obj}{rec}{extra}")

            step_start = time.time()
            success, message, nav_steps = self._dispatch(action)
            elapsed = time.time() - step_start

            result.step_log.append({
                "action_idx": i,
                "action":     action,
                "intent":     intent,
                "success":    success,
                "message":    message,
                "elapsed":    round(elapsed, 2),
                "nav_steps":  nav_steps,
            })

            if success:
                result.actions_done += 1
                if verbose:
                    print(f"           → OK  {message[:60]}")
            else:
                result.actions_failed += 1
                result.error_message = f"Step {i+1} ({intent}): {message}"
                # Classify failure for Metrics V2
                m_upper = message.upper()
                if "[OBJECT_NOT_FOUND]" in m_upper or "NOT FOUND IN SCENE" in m_upper or "NOT FOUND" in m_upper:
                    result.object_not_found += 1
                elif "[INVALID_ACTION]" in m_upper or "NOT PICKUPABLE" in m_upper or "NOT OPENABLE" in m_upper or "NOT SLICEABLE" in m_upper or "NOT TOGGLEABLE" in m_upper or "NOT A RECEPTACLE" in m_upper or "ALREADY HOLDING" in m_upper or "NOT HOLDING" in m_upper:
                    result.invalid_actions += 1
                if verbose:
                    print(f"           → FAIL {message[:60]}")
                # Hard stop — rest of task cannot proceed
                break

        result.elapsed = time.time() - task_start
        result.success = (result.actions_failed == 0 and
                          result.actions_done == len(actions))
        return result

    # ── Action dispatcher ─────────────────────────────────────────────────────

    def _dispatch(self, action: Dict) -> Tuple[bool, str, int]:
        """
        Route one action to the correct controller.

        Returns:
            (success, message, nav_steps)
        """
        intent = action.get("intent", "")

        if intent == "navigate":
            return self._do_navigate(action)
        elif intent == "wait":
            seconds = int(action.get("seconds", 3))
            success, msg = self.int_ctrl.wait_action(seconds)
            return success, msg, 0
        elif intent in ("pick","place","open","close","toggle",
                        "slice","cook","break","clean","pour","drop","throw"):
            success, msg = self._do_interact(action)
            return success, msg, 0
        else:
            return False, f"Unknown intent '{intent}'", 0

    # ── Navigation ────────────────────────────────────────────────────────────

    def _do_navigate(self, action: Dict) -> Tuple[bool, str, int]:
        """Navigate to target using RL DDQN."""
        target = action.get("target", "")
        try:
            # Check if target is inside a closed container first
            containment = self.int_ctrl.get_object_containment_info(target, self.controller)
            if containment.get("found") and not containment.get("is_accessible"):
                closed = containment.get("closed_containers", [])
                for container in closed:
                    logger.info(f"[V3] Auto-opening {container} to reach {target}")
                    nav_ok, _, pre_steps = self._do_navigate(
                        {"intent": "navigate", "target": container}
                    )
                    if nav_ok:
                        self.int_ctrl.open_object(container, self.controller)

            success, _, steps = self.nav_ctrl.navigate_to_object(target, self.max_nav_steps)
            if success:
                return True, f"Navigated to {target} ({steps} steps)", steps

            # Soft success: if near enough, still try interaction
            is_near, dist = self._is_near(target)
            if is_near:
                return True, f"Near {target} ({dist:.2f}m, nav failed)", steps

            # Object not found after navigation — try opening storage containers
            # to expose items that may be hidden inside Fridge, Cabinet, or Drawer
            if not containment.get("found"):
                logger.info(f"[V3] {target} not found anywhere — trying container search")
                exposed = self._open_search_containers(target)
                if exposed:
                    success2, _, steps2 = self.nav_ctrl.navigate_to_object(
                        target, self.max_nav_steps
                    )
                    total_steps = steps + steps2
                    if success2:
                        return True, f"Navigated to {target} after container search ({total_steps} steps)", total_steps
                    is_near2, dist2 = self._is_near(target)
                    if is_near2:
                        return True, f"Near {target} after container search ({dist2:.2f}m)", total_steps
                    return False, f"Could not reach {target} even after container search", total_steps

            return False, f"Could not reach {target} in {steps} steps", steps

        except Exception as e:
            logger.error(f"[V3] Navigation exception: {e}")
            return False, f"Navigation error: {e}", 0

    def _open_search_containers(self, seeking: str) -> bool:
        """
        Open Fridge, Cabinets, and Drawers to expose hidden objects.

        Called when `seeking` is not found in the scene at all.
        Returns True if at least one container was opened.
        """
        # Decide which containers to open based on what we're looking for
        seek_lower = seeking.lower()
        food_items = {"apple", "bread", "egg", "lettuce", "tomato", "potato",
                      "mug", "cup", "bowl", "plate", "pot", "pan", "butter", "milk"}
        utensils   = {"knife", "butcherknife", "fork", "spoon", "spatula", "ladle"}

        containers_to_try: List[str] = []
        if any(f in seek_lower for f in food_items):
            containers_to_try = ["Fridge", "Cabinet", "Drawer"]
        elif any(u in seek_lower for u in utensils):
            containers_to_try = ["Drawer", "Cabinet"]
        else:
            containers_to_try = ["Fridge", "Cabinet", "Drawer"]

        opened_any = False
        for container_type in containers_to_try:
            # Collect object IDs up-front; re-fetch live state inside the loop
            # so we skip containers that were already opened in a previous iteration.
            candidate_ids = [
                o["objectId"]
                for o in self.int_ctrl.find_all_objects_by_type(
                    container_type, self.controller, allow_invisible=True
                )
                if o.get("openable", False) and not o.get("isOpen", False)
            ]

            for obj_id in candidate_ids:
                # Re-fetch live metadata to check isOpen (state may have changed)
                live = self.int_ctrl.find_object_by_id(obj_id, self.controller)
                if live is None or live.get("isOpen", False):
                    continue  # already open or gone

                nav_ok, _, _ = self._do_navigate({"intent": "navigate", "target": container_type})
                if not nav_ok:
                    continue

                event = self.controller.step(
                    action="OpenObject",
                    objectId=obj_id,
                    forceAction=True,
                )
                if event.metadata.get("lastActionSuccess", False):
                    logger.info(f"[V3] Opened {container_type} while searching for {seeking}")
                    opened_any = True
                    # Check if the sought object is now findable
                    obj = self.int_ctrl.find_object_by_type(
                        seeking, self.controller, allow_invisible=True
                    )
                    if obj is not None:
                        logger.info(f"[V3] Found {seeking} after opening {container_type}")
                        return True

        return opened_any

    # ── Interaction ───────────────────────────────────────────────────────────

    def _do_interact(self, action: Dict) -> Tuple[bool, str]:
        """Execute interaction via RL interaction controller."""
        intent      = action.get("intent", "")
        target_obj  = action.get("object", action.get("target", ""))

        # ── Pre-pick: place any held object if it is not the pick target ────────
        if intent == "pick" and target_obj:
            held = self.int_ctrl.get_held_object(self.controller)
            if held and held.get("objectType", "").lower() != target_obj.lower():
                held_type = held.get("objectType", "Unknown")
                logger.info(f"[V3] Holding {held_type}, placing on CounterTop before picking {target_obj}")
                nav_ok, nav_msg, _ = self._do_navigate({"intent": "navigate", "target": "CounterTop"})
                if not nav_ok:
                    return False, f"Cannot reach CounterTop to put down {held_type}: {nav_msg}"
                place_ok, place_msg = self.int_ctrl.execute_interaction(
                    {"intent": "place", "object": held_type, "receptacle": "CounterTop"},
                    self.controller,
                    self._get_agent_state(),
                )
                if not place_ok:
                    return False, f"Could not place {held_type} on CounterTop before picking {target_obj}: {place_msg}"

        # ── Pre-slice: ensure agent holds a Knife before slicing ──────────────
        if intent == "slice":
            knife_ok, knife_msg = self._ensure_holding_knife(target_obj)
            if not knife_ok:
                return False, f"Could not get Knife for slice: {knife_msg}"

        # ── Pre-cook: auto-slice sliceable objects that haven't been sliced ───
        if intent == "cook" and target_obj:
            sliced_ok, new_target = self._auto_slice_if_needed(target_obj)
            if sliced_ok and new_target != target_obj:
                logger.info(f"[V3] Pre-cook slice complete: updating object {target_obj} → {new_target}")
                action = dict(action)          # don't mutate caller's dict
                action["object"] = new_target
                target_obj = new_target

        # Safety: ensure agent is near target before non-navigate interaction
        if target_obj and intent in ("pick","open","close","toggle","slice","cook","break","clean"):
            is_near, dist = self._is_near(target_obj)
            if not is_near:
                logger.info(f"[V3 SAFETY] Not near {target_obj} ({dist:.2f}m) — auto-navigating")
                nav_ok, nav_msg, _ = self._do_navigate(
                    {"intent": "navigate", "target": target_obj}
                )
                if not nav_ok:
                    return False, f"Cannot reach {target_obj} for {intent}: {nav_msg}"

        elif intent == "place":
            obj_to_place = action.get("object", "")
            receptacle = action.get("receptacle", "")
            # Auto-pick recovery: if plan omitted a pick step before this place,
            # fetch the object now (e.g. pick Pot before place Pot → StoveBurner)
            if obj_to_place:
                held = self.int_ctrl.get_held_object(self.controller)
                held_type = held.get("objectType", "").lower() if held else ""
                if not held or held_type != obj_to_place.lower():
                    # Not holding the intended object — try to pick it first
                    logger.info(f"[V3] Not holding {obj_to_place} for place — auto-picking")
                    pick_ok, pick_msg = self._do_interact(
                        {"intent": "pick", "object": obj_to_place}
                    )
                    if not pick_ok:
                        return False, f"Could not pick {obj_to_place} before placing: {pick_msg}"
            if receptacle:
                is_near, dist = self._is_near(receptacle)
                if not is_near:
                    logger.info(f"[V3 SAFETY] Not near {receptacle} — auto-navigating for place")
                    self._do_navigate({"intent": "navigate", "target": receptacle})

                # Auto-open: if receptacle is openable and closed, open it before placing
                rec_obj = self.int_ctrl.find_object_by_type(
                    receptacle, self.controller, allow_invisible=True
                )
                if rec_obj and rec_obj.get("openable", False) and not rec_obj.get("isOpen", False):
                    logger.info(f"[V3] Auto-opening {receptacle} before placing {obj_to_place}")
                    self.controller.step(
                        action="OpenObject",
                        objectId=rec_obj["objectId"],
                        forceAction=True,
                    )

        try:
            agent_state = self._get_agent_state()
            success, message = self.int_ctrl.execute_interaction(
                action, self.controller, agent_state
            )
            if (not success) and "[SAFETY_BLOCK]" in message.upper():
                return self._recover_from_safety_block(action, message)
            return success, message
        except Exception as e:
            logger.error(f"[V3] Interaction exception: {e}")
            return False, f"Interaction error: {e}"

    def _ensure_holding_knife(self, slice_target: str) -> Tuple[bool, str]:
        """Ensure agent holds a Knife before a slice action.

        If already holding a Knife/ButcherKnife — do nothing.
        If holding another object — place it on CounterTop first (never drop,
        physics drops are unreliable), then pick up the Knife.
        """
        held = self.int_ctrl.get_held_object(self.controller)
        if held:
            held_type = held.get("objectType", "").lower()
            if held_type in ("knife", "butcherknife"):
                return True, "already holding knife"
            # Place held object on CounterTop before picking Knife
            logger.info(f"[V3] Holding {held_type}, placing on CounterTop before picking Knife")
            nav_ok, nav_msg, _ = self._do_navigate({"intent": "navigate", "target": "CounterTop"})
            if not nav_ok:
                return False, f"Cannot reach CounterTop to put down {held_type}: {nav_msg}"
            place_ok, place_msg = self.int_ctrl.execute_interaction(
                {"intent": "place", "object": held.get("objectType"), "receptacle": "CounterTop"},
                self.controller,
                self._get_agent_state(),
            )
            if not place_ok:
                return False, f"Could not place {held_type} on CounterTop: {place_msg}"

        # Navigate to Knife and pick it up
        knife_types = ["Knife", "ButcherKnife"]
        picked = False
        for knife_type in knife_types:
            knife_obj = self.int_ctrl.find_object_by_type(
                knife_type, self.controller, allow_invisible=True
            )
            if knife_obj is None:
                continue
            nav_ok, nav_msg, _ = self._do_navigate({"intent": "navigate", "target": knife_type})
            if not nav_ok:
                continue
            pick_ok, pick_msg = self.int_ctrl.execute_interaction(
                {"intent": "pick", "object": knife_type},
                self.controller,
                self._get_agent_state(),
            )
            if pick_ok:
                logger.info(f"[V3] Picked up {knife_type} for slicing")
                picked = True
                break

        if not picked:
            return False, "No Knife found in scene"

        # Navigate back to the slice target
        nav_ok, nav_msg, _ = self._do_navigate({"intent": "navigate", "target": slice_target})
        if not nav_ok:
            return False, f"Cannot reach {slice_target} after picking Knife: {nav_msg}"
        return True, "ready to slice"

    def _auto_slice_if_needed(self, object_type: str) -> Tuple[bool, str]:
        """Before cooking, slice the object if it is sliceable and not yet sliced.

        Returns (changed, effective_object_name).
        changed=False means no slicing was done (object already sliced or not sliceable).
        """
        obj = self.int_ctrl.find_object_by_type(
            object_type, self.controller, allow_invisible=True
        )
        if obj is None:
            return False, object_type
        if not obj.get("sliceable", False) or obj.get("isSliced", False):
            return False, object_type

        logger.info(f"[V3] Pre-cook: {object_type} is sliceable and unsliced — auto-slicing")

        # Ensure we have a Knife
        knife_ok, knife_msg = self._ensure_holding_knife(object_type)
        if not knife_ok:
            logger.warning(f"[V3] Pre-cook auto-slice skipped (no knife): {knife_msg}")
            return False, object_type

        # Execute slice
        slice_ok, slice_msg = self.int_ctrl.execute_interaction(
            {"intent": "slice", "object": object_type},
            self.controller,
            self._get_agent_state(),
        )
        if not slice_ok:
            logger.warning(f"[V3] Pre-cook slice failed: {slice_msg}")
            return False, object_type

        # AI2-THOR replaces the whole object with a "TypeSliced" object
        sliced_name = object_type + "Sliced"
        logger.info(f"[V3] Pre-cook slice done — object is now '{sliced_name}'")
        return True, sliced_name

    # ── Utility ───────────────────────────────────────────────────────────────

    def _is_near(self, object_type: str) -> Tuple[bool, float]:
        """Check if agent is within interaction range of object."""
        try:
            obj = self.int_ctrl.find_object_by_type(
                object_type, self.controller, allow_invisible=True
            )
            if obj is None:
                return False, float("inf")
            dist = self.int_ctrl.get_object_distance(obj, self.controller)
            return dist <= self.INTERACTION_DISTANCE, dist
        except Exception:
            return False, float("inf")

    def _get_agent_state(self) -> Dict:
        """Return current agent state snapshot."""
        try:
            meta      = self.controller.last_event.metadata
            inv       = meta.get("inventoryObjects", [])
            agent     = meta.get("agent", {})
            holding   = inv[0] if inv else None
            return {
                "holding":        holding,
                "position":       agent.get("position", {}),
                "rotation":       agent.get("rotation", {}),
                "cameraHorizon":  agent.get("cameraHorizon", 0),
                "visible_objects": [
                    o["objectType"] for o in meta.get("objects", [])
                    if o.get("visible", False)
                ],
            }
        except Exception:
            return {"holding": None, "position": {}, "rotation": {},
                    "cameraHorizon": 0, "visible_objects": []}

    def _reset_safety_stats(self):
        self.safety_stats = {
            "unsafe_action_blocks": 0,
            "cooling_waits_inserted": 0,
            "recovery_attempts": 0,
            "recovery_successes": 0,
        }

    def prepare_task_queue(self, task_queue: List[Dict]) -> List[Dict]:
        """
        Normalize common planning gaps before printing or execution.

        Repairs applied are recorded in self.repair_log so callers and
        benchmark code can disclose auto-insertions in reported metrics.
        """
        self.repair_log = []
        prepared: List[Dict] = []
        for task in task_queue:
            prepared.append({
                "task_id": task.get("task_id", len(prepared) + 1),
                "task_name": task.get("task_name", "task"),
                "actions": [dict(action) for action in task.get("actions", [])],
            })

        if not self.enable_queue_repair:
            for idx, task in enumerate(prepared, start=1):
                task["task_id"] = idx
            return prepared

        before = {t["task_name"] for t in prepared}
        prepared = self._ensure_plate_ready_for_serving(prepared)
        after = {t["task_name"] for t in prepared}
        if after - before:
            self.repair_log.append("plate_retrieval_inserted")

        before_actions = sum(len(t.get("actions", [])) for t in prepared)
        prepared = self._ensure_faucet_shutdown_after_clean(prepared)
        after_actions = sum(len(t.get("actions", [])) for t in prepared)
        if after_actions > before_actions:
            self.repair_log.append("faucet_shutdown_inserted")

        before_actions = sum(len(t.get("actions", [])) for t in prepared)
        prepared = self._ensure_brewed_mug_is_served(prepared)
        after_actions = sum(len(t.get("actions", [])) for t in prepared)
        if after_actions > before_actions:
            self.repair_log.append("brewed_mug_retrieval_inserted")

        before_actions = sum(len(t.get("actions", [])) for t in prepared)
        prepared = self._inject_safety_steps(prepared)
        after_actions = sum(len(t.get("actions", [])) for t in prepared)
        if after_actions > before_actions:
            self.repair_log.append("safety_wait_inserted")

        if self.repair_log:
            logger.info("[Planner] Queue repairs applied: %s", ", ".join(self.repair_log))

        for idx, task in enumerate(prepared, start=1):
            task["task_id"] = idx
        return prepared

    def _ensure_plate_ready_for_serving(self, task_queue: List[Dict]) -> List[Dict]:
        """Insert a plate-retrieval task before plating cooked food when missing."""
        serve_idx = None
        for idx, task in enumerate(task_queue):
            for action in task.get("actions", []):
                if (
                    action.get("intent") == "place"
                    and action.get("receptacle", "").lower() == "plate"
                    and action.get("object", "").lower() != "plate"
                ):
                    serve_idx = idx
                    break
            if serve_idx is not None:
                break

        if serve_idx is None:
            return task_queue

        for task in task_queue[: serve_idx + 1]:
            for action in task.get("actions", []):
                if action.get("object", "").lower() == "plate" and action.get("intent") in {"pick", "place"}:
                    return task_queue

        prepare_plate = {
            "task_id": 0,
            "task_name": "prepare_plate",
            "actions": [
                {"intent": "navigate", "target": "Cabinet"},
                {"intent": "open", "object": "Cabinet"},
                {"intent": "navigate", "target": "Plate"},
                {"intent": "pick", "object": "Plate"},
                {"intent": "navigate", "target": "Cabinet"},
                {"intent": "close", "object": "Cabinet"},
                {"intent": "navigate", "target": "CounterTop"},
                {"intent": "place", "object": "Plate", "receptacle": "CounterTop"},
            ],
        }
        updated = list(task_queue)
        updated.insert(serve_idx, prepare_plate)
        return updated

    def _ensure_faucet_shutdown_after_clean(self, task_queue: List[Dict]) -> List[Dict]:
        """Ensure wash plans turn the faucet off after cleaning."""
        updated: List[Dict] = []
        for task in task_queue:
            actions = task.get("actions", [])
            new_actions: List[Dict] = []
            for idx, action in enumerate(actions):
                new_actions.append(dict(action))
                if action.get("intent") != "clean":
                    continue

                washed_object = action.get("object", "")
                trailing = actions[idx + 1 :]
                has_faucet_off = any(
                    follow.get("intent") == "toggle" and follow.get("object", "").lower() == "faucet"
                    for follow in trailing[:3]
                )
                if not has_faucet_off:
                    new_actions.append({"intent": "toggle", "object": "Faucet"})

                has_pickup = any(
                    follow.get("intent") == "pick" and follow.get("object", "").lower() == washed_object.lower()
                    for follow in trailing[:5]
                )
                if washed_object and not has_pickup:
                    new_actions.append({"intent": "navigate", "target": washed_object})
                    new_actions.append({"intent": "pick", "object": washed_object})

            updated.append({
                "task_id": task.get("task_id", 0),
                "task_name": task.get("task_name", "task"),
                "actions": new_actions,
            })
        return updated

    def _ensure_brewed_mug_is_served(self, task_queue: List[Dict]) -> List[Dict]:
        """Retrieve the mug after brewing so the final state is visible to the user."""
        updated: List[Dict] = []
        for task in task_queue:
            actions = [dict(action) for action in task.get("actions", [])]
            toggle_idx = None
            brew_object = ""

            for idx, action in enumerate(actions):
                if action.get("intent") == "toggle" and action.get("object", "").lower() == "coffeemachine":
                    toggle_idx = idx
                    for prior in reversed(actions[:idx]):
                        if (
                            prior.get("intent") == "place"
                            and prior.get("receptacle", "").lower() == "coffeemachine"
                        ):
                            brew_object = prior.get("object", "")
                            break
                    break

            if toggle_idx is not None and brew_object:
                has_retrieval = any(
                    action.get("intent") == "pick" and action.get("object", "").lower() == brew_object.lower()
                    for action in actions[toggle_idx + 1 :]
                )
                if not has_retrieval:
                    prefix = actions[: toggle_idx + 1]
                    suffix = actions[toggle_idx + 1 :]
                    waits: List[Dict] = []
                    while suffix and suffix[0].get("intent") == "wait":
                        waits.append(suffix.pop(0))
                    if not waits:
                        waits.append({"intent": "wait", "seconds": 1, "reason": "Allow coffee to brew"})
                    actions = prefix + waits + [
                        {"intent": "navigate", "target": brew_object},
                        {"intent": "pick", "object": brew_object},
                        {"intent": "navigate", "target": "CounterTop"},
                        {"intent": "place", "object": brew_object, "receptacle": "CounterTop"},
                    ] + suffix

            updated.append({
                "task_id": task.get("task_id", 0),
                "task_name": task.get("task_name", "task"),
                "actions": actions,
            })
        return updated

    def _inject_safety_steps(self, task_queue: List[Dict]) -> List[Dict]:
        """Insert short cooling waits after cook actions when none exist."""
        safe_queue: List[Dict] = []
        for task in task_queue:
            new_task = dict(task)
            new_actions: List[Dict] = []
            actions = task.get("actions", [])
            for idx, action in enumerate(actions):
                new_actions.append(dict(action))
                next_action = actions[idx + 1] if idx + 1 < len(actions) else {}
                if action.get("intent") == "cook" and next_action.get("intent") != "wait":
                    new_actions.append({
                        "intent": "wait",
                        "seconds": 1,
                        "reason": "Cooling pause after cook",
                    })
                    self.safety_stats["cooling_waits_inserted"] += 1
            new_task["actions"] = new_actions
            safe_queue.append(new_task)
        return safe_queue

    def _cool_and_recheck(self, object_type: str, max_cycles: int = 3) -> Tuple[bool, str]:
        """Advance the sim with short waits and re-check whether pickup is safe."""
        for cycle in range(max_cycles):
            self.int_ctrl.wait_action(1)
            try:
                self.controller.step(action="Pass")
            except Exception:
                pass
            obj = self.int_ctrl.get_object_metadata(object_type, self.controller, allow_invisible=True)
            safe, _, _ = check_action_safety(
                "pick",
                obj_meta=obj,
                action={"intent": "pick", "object": object_type},
            )
            if safe:
                return True, f"[RECOVERY] {object_type} cooled enough after {cycle + 1} wait cycle(s)"
        return False, f"[SAFETY_BLOCK] {object_type} is still unsafe after cooling retries"

    def _recover_from_safety_block(self, action: Dict, message: str) -> Tuple[bool, str]:
        """Attempt bounded recovery after an execution-time safety block."""
        self.safety_stats["unsafe_action_blocks"] += 1
        self.safety_stats["recovery_attempts"] += 1

        intent = action.get("intent", "")
        object_type = action.get("object", action.get("target", ""))
        knob_match = re.search(r"knob_index=(\d+)", message or "", flags=re.IGNORECASE)
        recommended_knob = int(knob_match.group(1)) if knob_match else None

        if intent == "pick" and object_type:
            cooled, cool_msg = self._cool_and_recheck(object_type)
            if not cooled:
                return False, cool_msg
            agent_state = self._get_agent_state()
            success, retry_msg = self.int_ctrl.execute_interaction(action, self.controller, agent_state)
            if success:
                self.safety_stats["recovery_successes"] += 1
                return True, f"{cool_msg} | {retry_msg}"
            return False, f"{cool_msg} | {retry_msg}"

        if intent == "toggle" and object_type.lower() == "stoveknob" and recommended_knob is not None:
            retry_action = dict(action)
            retry_action["knob_index"] = recommended_knob
            success, retry_msg = self.int_ctrl.execute_interaction(
                retry_action,
                self.controller,
                self._get_agent_state(),
            )
            if success:
                self.safety_stats["recovery_successes"] += 1
                return True, (
                    f"[RECOVERY] Retried StoveKnob with knob_index={recommended_knob} | "
                    f"{retry_msg}"
                )
            return False, (
                f"[RECOVERY] Retried StoveKnob with knob_index={recommended_knob} | "
                f"{retry_msg}"
            )

        if intent == "cook" and recommended_knob is not None:
            nav_ok, nav_msg, _ = self._do_navigate({"intent": "navigate", "target": "StoveKnob"})
            if not nav_ok:
                return False, f"{message} | [RECOVERY] Could not reach StoveKnob: {nav_msg}"
            toggle_ok, toggle_msg = self.int_ctrl.execute_interaction(
                {"intent": "toggle", "object": "StoveKnob", "knob_index": recommended_knob},
                self.controller,
                self._get_agent_state(),
            )
            if not toggle_ok:
                return False, (
                    f"{message} | [RECOVERY] Could not activate knob_index={recommended_knob}: "
                    f"{toggle_msg}"
                )
            success, retry_msg = self.int_ctrl.execute_interaction(
                action,
                self.controller,
                self._get_agent_state(),
            )
            if success:
                self.safety_stats["recovery_successes"] += 1
                return True, (
                    f"[RECOVERY] Activated knob_index={recommended_knob} before cooking | "
                    f"{retry_msg}"
                )
            return False, (
                f"[RECOVERY] Activated knob_index={recommended_knob} before cooking | "
                f"{retry_msg}"
            )

        return False, f"{message} | [RECOVERY] No safe recovery available"

    def get_safety_stats(self) -> Dict[str, int]:
        return dict(self.safety_stats)

    def get_history(self) -> List[QueueReport]:
        return self.queue_history.copy()

    def clear_history(self):
        self.queue_history.clear()
        try:
            self.nav_ctrl.clear_navigation_log()
        except Exception:
            pass
        try:
            self.int_ctrl.clear_interaction_log()
        except Exception:
            pass
