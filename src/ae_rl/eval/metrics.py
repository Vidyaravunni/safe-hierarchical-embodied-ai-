"""
Metrics V2 — Comprehensive task performance tracker.

Tracks 6 metrics:
  1. Success Rate          — % of tasks completed successfully
  2. SPL                   — Success weighted by Path Length (path efficiency)
  3. Parse Fail Rate       — % of NLP commands that produced no executable plan
  4. Object-Not-Found Rate — % of interaction attempts where the target object was absent
  5. Invalid Action Rate   — % of interaction attempts blocked by bad affordance/precondition
  6. Collision Rate        — % of navigation steps that resulted in a collision

History is saved to metrics_v2_history.json for cross-session analysis.

Usage:
    tracker = MetricsTrackerV2()

    # For each command:
    tracker.begin_command("make me a coffee")
    tracker.set_parse_success(True)

    # During execution (called by kitchen_demo_floorplan1.py):
    tracker.add_navigation_event(steps=18, collisions=2)
    tracker.add_interaction_attempt("pick", "Mug", success=True)
    tracker.add_interaction_attempt("pick", "Egg", success=False,
                                    failure_reason="[OBJECT_NOT_FOUND] Egg not found")

    tracker.end_command(success=True)

    # After session:
    tracker.print_report()
    tracker.save()
"""

import json
import os
import datetime
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

METRICS_V2_FILE = "metrics_v2_history.json"


# ─── Per-command record ───────────────────────────────────────────────────────

class CommandRecord:
    """Stores all events for a single NLP command."""

    def __init__(self, command: str):
        self.command          = command
        self.timestamp        = datetime.datetime.now().isoformat(timespec="seconds")
        self.parse_success    = False   # LLM returned a non-empty plan
        self.task_success     = False   # all sub-tasks completed
        self.nav_steps        = 0       # total navigation steps taken
        self.optimal_steps    = None    # SPL denominator (best-seen for this command)
        self.collisions       = 0       # navigation collisions
        self.interactions     = 0       # total interaction attempts
        self.object_not_found = 0       # attempts where object was absent
        self.invalid_actions  = 0       # attempts blocked by bad affordance / precondition
        self.action_failures  = 0       # other failed interaction attempts
        self.unsafe_command_blocks = 0
        self.unsafe_plan_blocks = 0
        self.unsafe_action_blocks = 0
        self.cooling_waits = 0
        self.fragile_protections = 0
        self.recovery_attempts = 0
        self.recovery_successes = 0
        self.wrong_burner_toggle_blocks = 0
        self.empty_burner_toggle_blocks = 0
        self.near_heat_hazard_blocks = 0
        self.unsafe_burner_occupancy_blocks = 0


# ─── Tracker ─────────────────────────────────────────────────────────────────

class MetricsTrackerV2:
    """
    Session + persistent metrics tracker.

    Call begin_command / set_parse_success / add_* / end_command for each task,
    then print_report() and save() at the end of the session.
    """

    def __init__(self, history_file: str = METRICS_V2_FILE, fresh_baselines: bool = False):
        self.history_file = history_file
        self.records: List[CommandRecord] = []
        self._current: Optional[CommandRecord] = None
        # Per-command-type minimum nav_steps on a successful run (for SPL)
        self._step_baselines: dict = {}
        if not fresh_baselines:
            self._load_baselines()
        else:
            logger.info("[MetricsV2] fresh_baselines=True — SPL computed from this run only")

    # ── Command lifecycle ─────────────────────────────────────────────────────

    def begin_command(self, command: str) -> None:
        """Call when a new NLP command is received (before LLM parse)."""
        self._current = CommandRecord(command)

    def set_parse_success(self, success: bool) -> None:
        """
        Call after LLM returns.
        success=True  → non-empty task_queue produced.
        success=False → LLM failed / returned empty plan (metric 3 numerator).
        """
        if self._current:
            self._current.parse_success = success

    def end_command(self, success: bool) -> None:
        """Call after the full task_queue finishes executing."""
        if self._current is None:
            return
        self._current.task_success = success

        # SPL optimal path: prefer ground-truth BFS (set by add_navigation_event)
        # over the "best-seen actual" heuristic.
        if self._current.optimal_steps is None or self._current.optimal_steps == 0:
            # Fallback: best-seen actual steps (legacy behaviour)
            key = self._cmd_key(self._current.command)
            if success and self._current.nav_steps > 0:
                prev = self._step_baselines.get(key, self._current.nav_steps)
                self._step_baselines[key] = min(prev, self._current.nav_steps)
            self._current.optimal_steps = self._step_baselines.get(key)

        self.records.append(self._current)
        self._current = None

    # ── Event recorders ───────────────────────────────────────────────────────

    def add_navigation_event(self, steps: int, collisions: int,
                             optimal_steps: int = 0) -> None:
        """Accumulate navigation steps + collisions for the current command.

        Args:
            steps: Actual navigation steps taken.
            collisions: Number of collisions during navigation.
            optimal_steps: Ground-truth shortest path in grid steps (BFS).
                           Used as l_i denominator in SPL (Anderson et al. 2018).
                           If 0, falls back to best-seen heuristic.
        """
        if self._current:
            self._current.nav_steps  += max(0, steps)
            self._current.collisions += max(0, collisions)
            if optimal_steps > 0:
                # Accumulate ground-truth optimal steps across all nav episodes
                # in this command (a command may navigate to multiple objects)
                if self._current.optimal_steps is None:
                    self._current.optimal_steps = 0
                self._current.optimal_steps += optimal_steps

    def add_interaction_attempt(
        self,
        intent: str,
        object_type: str,
        success: bool,
        failure_reason: str = "",
    ) -> None:
        """
        Record one interaction attempt and classify failures.

        failure_reason is the message string returned by the interaction controller.
        Strings containing [OBJECT_NOT_FOUND] or [INVALID_ACTION] tags are classified
        automatically; other failures go into action_failures.
        """
        if self._current is None:
            return
        self._current.interactions += 1
        r = failure_reason.upper()
        if "[SAFETY_BLOCK]" in r:
            self._current.unsafe_action_blocks += 1
        if success:
            return

        if (
            "[OBJECT_NOT_FOUND]" in r
            or "NOT FOUND IN SCENE" in r
            or "NOT FOUND" in r
            or "NOT VISIBLE AND TOO FAR" in r
        ):
            self._current.object_not_found += 1
        elif (
            "[INVALID_ACTION]" in r
            or "NOT PICKUPABLE" in r
            or "NOT OPENABLE" in r
            or "NOT CLOSEABLE" in r
            or "NOT SLICEABLE" in r
            or "NOT TOGGLEABLE" in r
            or "NOT A RECEPTACLE" in r
            or "NOT BREAKABLE" in r
            or "ALREADY HOLDING" in r
            or "NOT HOLDING" in r
            or "IS NOT PICKUPABLE" in r
            or "IS NOT OPENABLE" in r
            or "IS NOT SLICEABLE" in r
            or "IS NOT TOGGLEABLE" in r
        ):
            self._current.invalid_actions += 1
        else:
            self._current.action_failures += 1

    def add_safety_event(self, event_type: str, success: bool = True) -> None:
        if self._current is None:
            return
        if event_type == "unsafe_command_block":
            self._current.unsafe_command_blocks += 1
        elif event_type == "unsafe_plan_block":
            self._current.unsafe_plan_blocks += 1
        elif event_type == "cooling_wait":
            self._current.cooling_waits += 1
        elif event_type == "fragile_protection":
            self._current.fragile_protections += 1
        elif event_type == "recovery_attempt":
            self._current.recovery_attempts += 1
        elif event_type == "recovery_success":
            self._current.recovery_successes += 1
        elif event_type == "wrong_burner_toggle_block":
            self._current.wrong_burner_toggle_blocks += 1
        elif event_type == "empty_burner_toggle_block":
            self._current.empty_burner_toggle_blocks += 1
        elif event_type == "near_heat_hazard_block":
            self._current.near_heat_hazard_blocks += 1
        elif event_type == "unsafe_burner_occupancy_block":
            self._current.unsafe_burner_occupancy_blocks += 1

    # ── Metric computation ────────────────────────────────────────────────────

    def _spl(self, rec: CommandRecord) -> float:
        """
        SPL for one record.
        SPL = success × (optimal_steps / max(actual_steps, optimal_steps))
        If optimal_steps is unknown, a successful run with any nav contributes SPL=1.
        """
        if not rec.task_success:
            return 0.0
        li = rec.optimal_steps or rec.nav_steps   # fallback: first success = optimal
        pi = rec.nav_steps
        if li == 0:
            return 1.0
        return li / max(pi, li)

    def get_metrics(self) -> dict:
        """Compute and return all 6 metrics plus raw counters."""
        recs = self.records
        n = len(recs)
        if n == 0:
            return {
                "total_commands":         0,
                "success_rate":           0.0,
                "spl":                    0.0,
                "parse_fail_rate":        0.0,
                "object_not_found_rate":  0.0,
                "invalid_action_rate":    0.0,
                "collision_rate":         0.0,
                "_tasks_succeeded":       0,
                "_parse_failures":        0,
                "_object_not_found":      0,
                "_invalid_actions":       0,
                "_total_interactions":    0,
                "_total_nav_steps":       0,
                "_total_collisions":      0,
                "unsafe_command_block_rate": 0.0,
                "unsafe_action_block_rate": 0.0,
                "recovery_success_rate": 0.0,
                "_unsafe_command_blocks": 0,
                "_unsafe_plan_blocks": 0,
                "_unsafe_action_blocks": 0,
                "_cooling_waits": 0,
                "_fragile_protections": 0,
                "_recovery_attempts": 0,
                "_recovery_successes": 0,
                "_wrong_burner_toggle_blocks": 0,
                "_empty_burner_toggle_blocks": 0,
                "_near_heat_hazard_blocks": 0,
                "_unsafe_burner_occupancy_blocks": 0,
            }

        tasks_ok       = sum(1 for r in recs if r.task_success)
        parse_failed   = sum(1 for r in recs if not r.parse_success)
        total_interact = sum(r.interactions     for r in recs)
        total_onf      = sum(r.object_not_found for r in recs)
        total_inv      = sum(r.invalid_actions  for r in recs)
        total_nav      = sum(r.nav_steps        for r in recs)
        total_col      = sum(r.collisions       for r in recs)
        avg_spl        = sum(self._spl(r) for r in recs) / n
        total_unsafe_cmd = sum(r.unsafe_command_blocks for r in recs)
        total_unsafe_plan = sum(r.unsafe_plan_blocks for r in recs)
        total_unsafe_actions = sum(r.unsafe_action_blocks for r in recs)
        total_cooling = sum(r.cooling_waits for r in recs)
        total_fragile = sum(r.fragile_protections for r in recs)
        total_recovery_attempts = sum(r.recovery_attempts for r in recs)
        total_recovery_successes = sum(r.recovery_successes for r in recs)
        total_wrong_burner = sum(r.wrong_burner_toggle_blocks for r in recs)
        total_empty_burner = sum(r.empty_burner_toggle_blocks for r in recs)
        total_near_heat = sum(r.near_heat_hazard_blocks for r in recs)
        total_unsafe_burner = sum(r.unsafe_burner_occupancy_blocks for r in recs)

        return {
            # ── 6 main metrics ──────────────────────────────────────────────
            "total_commands":         n,
            "success_rate":           round(tasks_ok / n * 100,                       1),
            "spl":                    round(avg_spl  * 100,                           1),
            "parse_fail_rate":        round(parse_failed / n * 100,                   1),
            "object_not_found_rate":  round(total_onf / max(total_interact, 1) * 100, 1),
            "invalid_action_rate":    round(total_inv / max(total_interact, 1) * 100, 1),
            "collision_rate":         round(total_col / max(total_nav, 1)      * 100, 1),
            "unsafe_command_block_rate": round(total_unsafe_cmd / n * 100, 1),
            "unsafe_action_block_rate": round(total_unsafe_actions / max(total_interact, 1) * 100, 1),
            "recovery_success_rate": round(total_recovery_successes / max(total_recovery_attempts, 1) * 100, 1),
            # ── raw counters (for display) ───────────────────────────────────
            "_tasks_succeeded":    tasks_ok,
            "_parse_failures":     parse_failed,
            "_object_not_found":   total_onf,
            "_invalid_actions":    total_inv,
            "_total_interactions": total_interact,
            "_total_nav_steps":    total_nav,
            "_total_collisions":   total_col,
            "_unsafe_command_blocks": total_unsafe_cmd,
            "_unsafe_plan_blocks": total_unsafe_plan,
            "_unsafe_action_blocks": total_unsafe_actions,
            "_cooling_waits": total_cooling,
            "_fragile_protections": total_fragile,
            "_recovery_attempts": total_recovery_attempts,
            "_recovery_successes": total_recovery_successes,
            "_wrong_burner_toggle_blocks": total_wrong_burner,
            "_empty_burner_toggle_blocks": total_empty_burner,
            "_near_heat_hazard_blocks": total_near_heat,
            "_unsafe_burner_occupancy_blocks": total_unsafe_burner,
        }

    # ── Display ───────────────────────────────────────────────────────────────

    def print_report(self) -> None:
        """Print a formatted MetricsV2 report to stdout."""
        m = self.get_metrics()
        n = m["total_commands"]
        if n == 0:
            print("  [MetricsV2] No tasks recorded yet.")
            return

        W = 66
        bar = "═" * W
        print(f"\n  ╔{bar}╗")
        print(f"  ║{'METRICS V2  ─  PERFORMANCE REPORT':^{W}}║")
        print(f"  ╠{bar}╣")
        print(f"  ║  Commands evaluated        : {n:<{W - 34}}║")
        print(f"  ╠{bar}╣")

        def _row(num, label, pct, note=""):
            note_str = f"  ({note})" if note else ""
            body = f"  {num}. {label:<26}: {pct:>6}%{note_str}"
            print(f"  ║  {body:<{W - 2}}║")

        _row("1", "Success Rate         ", f"{m['success_rate']:.1f}",
             f"{m['_tasks_succeeded']}/{n} tasks succeeded")
        _row("2", "Path Efficiency (SPL)", f"{m['spl']:.1f}",
             "100% = perfect shortest path")
        _row("3", "Parse Fail Rate      ", f"{m['parse_fail_rate']:.1f}",
             f"{m['_parse_failures']} command(s) produced no plan")
        _row("4", "Object-Not-Found Rate", f"{m['object_not_found_rate']:.1f}",
             f"{m['_object_not_found']}/{m['_total_interactions']} interactions")
        _row("5", "Invalid Action Rate  ", f"{m['invalid_action_rate']:.1f}",
             f"{m['_invalid_actions']}/{m['_total_interactions']} interactions")
        _row("6", "Collision Rate       ", f"{m['collision_rate']:.1f}",
             f"{m['_total_collisions']}/{m['_total_nav_steps']} nav steps")
        _row("S1", "Unsafe Cmd Block Rate", f"{m['unsafe_command_block_rate']:.1f}",
             f"{m['_unsafe_command_blocks']}/{n} commands")
        _row("S2", "Unsafe Act Block Rate", f"{m['unsafe_action_block_rate']:.1f}",
             f"{m['_unsafe_action_blocks']}/{m['_total_interactions']} interactions")
        _row("S3", "Recovery Success Rate", f"{m['recovery_success_rate']:.1f}",
             f"{m['_recovery_successes']}/{m['_recovery_attempts']} recoveries")
        print(f"  ╚{bar}╝")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Append this session's data to metrics_v2_history.json."""
        history: list = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file) as f:
                    history = json.load(f)
            except (json.JSONDecodeError, IOError):
                history = []

        session = {
            "session_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "metrics":           self.get_metrics(),
            "per_command":       [self._rec_dict(r) for r in self.records],
        }
        history.append(session)
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _rec_dict(self, r: CommandRecord) -> dict:
        return {
            "command":          r.command,
            "timestamp":        r.timestamp,
            "parse_success":    r.parse_success,
            "task_success":     r.task_success,
            "spl":              round(self._spl(r), 4),
            "nav_steps":        r.nav_steps,
            "optimal_steps":    r.optimal_steps,
            "collisions":       r.collisions,
            "interactions":     r.interactions,
            "object_not_found": r.object_not_found,
            "invalid_actions":  r.invalid_actions,
            "action_failures":  r.action_failures,
            "unsafe_command_blocks": r.unsafe_command_blocks,
            "unsafe_plan_blocks": r.unsafe_plan_blocks,
            "unsafe_action_blocks": r.unsafe_action_blocks,
            "cooling_waits": r.cooling_waits,
            "fragile_protections": r.fragile_protections,
            "recovery_attempts": r.recovery_attempts,
            "recovery_successes": r.recovery_successes,
            "wrong_burner_toggle_blocks": r.wrong_burner_toggle_blocks,
            "empty_burner_toggle_blocks": r.empty_burner_toggle_blocks,
            "near_heat_hazard_blocks": r.near_heat_hazard_blocks,
            "unsafe_burner_occupancy_blocks": r.unsafe_burner_occupancy_blocks,
        }

    def _load_baselines(self) -> None:
        """Warm-start SPL baselines from prior history file."""
        if not os.path.exists(self.history_file):
            return
        try:
            with open(self.history_file) as f:
                history = json.load(f)
            for session in history:
                for rec in session.get("per_command", []):
                    if rec.get("task_success") and rec.get("nav_steps", 0) > 0:
                        key  = self._cmd_key(rec["command"])
                        prev = self._step_baselines.get(key, rec["nav_steps"])
                        self._step_baselines[key] = min(prev, rec["nav_steps"])
        except Exception:
            pass

    @staticmethod
    def _cmd_key(command: str) -> str:
        """Normalise command string for baseline dictionary keys."""
        return command.lower().strip()
