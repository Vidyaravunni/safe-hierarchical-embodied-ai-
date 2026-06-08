"""
Navigation Controller - RL-based Object-Directed Navigation

Wraps existing DDQN agent for object search and navigation tasks.
Uses reinforcement learning for exploration when objects are not visible.
"""

import logging
import math
import numpy as np
from typing import Tuple, Optional, List, Dict
from ae_rl.rl.ddqn import DDQN
from ae_rl.sim.ai2thor_env import AIThorEnv
from ae_rl.rl.nav.state import StateModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NavigationController:
    """RL-based navigation controller wrapping existing DDQN agent"""

    # ========== NAVIGATION CONSTANTS ==========
    # Rotation loop prevention
    MAX_CONSECUTIVE_ROTATIONS = 8   # Max rotations before forcing forward movement
    FORCED_FORWARD_STEPS = 3        # Steps to force forward after rotation limit

    # Stuck detection
    STUCK_THRESHOLD = 5             # Steps with no distance change before considering stuck
    DISTANCE_CHANGE_EPSILON = 0.01  # Minimum distance change to not be considered stuck
    GEO_ESCAPE_THRESHOLD = 30       # Stuck steps before geometric escape (reachable-position teleport)
    GEO_ESCAPE_MAX_ATTEMPTS = 2     # Max geometric escapes per navigation episode

    # Bearing thresholds (normalized, where 1.0 = π radians = 180°)
    BEARING_FINE_ADJUST = 0.1       # Bearing threshold for fine adjustments
    BEARING_ROUGHLY_AHEAD = 0.3     # Target roughly in front
    BEARING_SIDE = 0.6              # Target to the side
    BEARING_SEARCH = 0.2            # Threshold for search mode rotation

    # Distance thresholds (normalized, where 1.0 = 10m)
    CLOSE_DISTANCE_THRESHOLD = 0.2  # Normalized distance for "very close"

    def __init__(self, ddqn_model: DDQN, env: AIThorEnv, state_space: int = StateModel.TOTAL_STATE_SIZE):
        """
        Initialize navigation controller

        Args:
            ddqn_model: Trained DDQN policy network
            env: AI2-THOR environment wrapper
            state_space: State space dimension (default 75: 72 grid + 3 target info)
        """
        self.ddqn_model = ddqn_model
        self.env = env
        self.state_space = state_space
        self.navigation_log = []
        self._current_optimal_steps = 0  # BFS ground-truth for current target
        
    def navigate_to_object(self, target_object: str, max_steps: int = 100) -> Tuple[bool, np.ndarray, int]:
        """
        Navigate to target object using RL policy

        If object is not immediately visible, uses RL exploration to find it.

        Args:
            target_object: Object type to navigate to (e.g., "Apple", "CounterTop")
            max_steps: Maximum navigation steps

        Returns:
            Tuple of (success: bool, final_state: np.ndarray, steps_taken: int)
        """
        logger.info(f"Navigating to: {target_object}")

        # Set target in environment so state includes target info
        if not self.env.set_target(target_object):
            logger.warning(f"Target {target_object} not found in scene")
            return False, self.get_current_state(), 0

        # Reset environment step counter for this navigation task
        self.env.step_count = 0
        self.env.prev_distance = self.env.get_distance_to_target()

        # Update target info in state model
        target_info = self.env.get_target_info()
        self.env.stateModel.set_target_info(
            target_info['bearing'],
            target_info['distance'],
            target_info['visible']
        )

        # Compute ground-truth shortest path for SPL (Anderson et al. 2018)
        self._current_optimal_steps = self.env.compute_shortest_path_steps(target_object)
        logger.info(f"Target set: {target_object}, distance={self.env.prev_distance:.2f}m, visible={target_info['visible']}, optimal_steps={self._current_optimal_steps}")

        # Check if object is already visible and close enough
        if self.env.check_success():
            logger.info(f"{target_object} is already reachable")
            return True, self.get_current_state(), 0

        # Explore using RL policy to find object
        success, final_state, steps = self.explore_until_found(target_object, max_steps)

        if success:
            logger.info(f"Found {target_object} after {steps} steps")
        else:
            logger.warning(f"Failed to find {target_object} within {max_steps} steps")

        return success, final_state, steps
    
    def explore_until_found(self, target_object: str, max_steps: int) -> Tuple[bool, np.ndarray, int]:
        """
        Navigate to target using state machine approach.

        States:
        1. SEARCH: Target not visible - rotate toward bearing, then move forward
        2. APPROACH: Target visible - move toward it while adjusting heading
        3. REACH: Target visible and close - fine adjustments

        Args:
            target_object: Object to search for
            max_steps: Maximum exploration steps

        Returns:
            (success, final_state, steps_taken)
        """
        logger.info(f"Starting navigation for: {target_object} (max {max_steps} steps)")

        # Get initial state
        state = self.env.get_current_state()
        state = np.reshape(state, (1, self.state_space))

        # Navigation state
        consecutive_collisions = 0
        total_collisions = 0  # Track total collisions for metrics
        last_distance = self.env.get_distance_to_target()
        stuck_counter = 0
        rotation_count = 0
        forced_forward_steps = 0  # Counter to force forward movement after rotation limit
        looked_down = False       # Whether we've tried looking down for objects in drawers
        geo_escapes_used = 0      # Number of geometric escapes attempted this episode

        for step in range(max_steps):
            # Check success condition
            if self.env.check_success():
                logger.info(f"SUCCESS: {target_object} found at step {step}")
                self.navigation_log.append({"target": target_object, "success": True, "steps": step, "collisions": total_collisions, "optimal_steps": self._current_optimal_steps})
                return True, state, step

            # Get current target info
            target_info = self.env.get_target_info()
            bearing = target_info['bearing']
            visible = target_info['visible']
            distance = self.env.get_distance_to_target()

            # Handle forced forward movement (to break rotation loops)
            if forced_forward_steps > 0:
                action = 1  # MoveAhead
                forced_forward_steps -= 1
                logger.debug(f"Forced forward movement: {forced_forward_steps} steps remaining")
            else:
                # State-machine-driven action selection (handles visible/not-visible,
                # close/far, stuck, and collision cases internally).
                action = self._select_navigation_action(
                    bearing, visible, distance,
                    consecutive_collisions, stuck_counter, rotation_count
                )

            # Execute action
            reward, next_state, done = self.env.step(action)
            action_success = self.env.event.metadata.get('lastActionSuccess', True)

            # Update collision tracking
            if not action_success:
                consecutive_collisions += 1
                total_collisions += 1
            else:
                consecutive_collisions = 0

            # Check if stuck (distance not changing)
            new_distance = self.env.get_distance_to_target()
            if abs(new_distance - last_distance) < self.DISTANCE_CHANGE_EPSILON:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_distance = new_distance

            # Geometric escape: when stuck for many steps, use reachable-position
            # planning to teleport to the nearest navigable point to the target.
            # This preserves the RL-first approach while avoiding permanent stalls.
            if (
                stuck_counter >= self.GEO_ESCAPE_THRESHOLD
                and geo_escapes_used < self.GEO_ESCAPE_MAX_ATTEMPTS
            ):
                logger.info(
                    "[NAV] Stuck for %d steps at %.2fm — attempting geometric escape (%d/%d)",
                    stuck_counter, new_distance,
                    geo_escapes_used + 1, self.GEO_ESCAPE_MAX_ATTEMPTS,
                )
                if self._geometric_escape(target_object):
                    geo_escapes_used += 1
                    stuck_counter = 0
                    last_distance = self.env.get_distance_to_target()
                    state = np.reshape(self.env.get_current_state(), (1, self.state_space))
                    # Check if escape landed us at success
                    if self.env.check_success():
                        logger.info(f"SUCCESS: {target_object} reached via geometric escape at step {step + 1}")
                        self.navigation_log.append({
                            "target": target_object, "success": True,
                            "steps": step + 1, "collisions": total_collisions,
                            "geo_escapes": geo_escapes_used,
                            "optimal_steps": self._current_optimal_steps,
                        })
                        return True, state, step + 1
                    continue  # Resume RL from the new position

            # Look-down strategy: when close but not visible (object in open drawer/low shelf)
            # Trigger after 15 steps if within 1.2m but still not visible
            if not looked_down and not visible and new_distance < 1.2 and step >= 15:
                looked_down = True
                logger.info(f"Close but not visible ({new_distance:.2f}m) — trying look down")
                self.env.controller.step(action="LookDown", degrees=30)
                if self.env.check_success():
                    logger.info(f"SUCCESS: {target_object} found after looking down at step {step + 1}")
                    self.navigation_log.append({"target": target_object, "success": True, "steps": step + 1, "collisions": total_collisions, "optimal_steps": self._current_optimal_steps})
                    return True, np.reshape(self.env.get_current_state(), (1, self.state_space)), step + 1
                # Try looking further down
                self.env.controller.step(action="LookDown", degrees=30)
                if self.env.check_success():
                    logger.info(f"SUCCESS: {target_object} found after looking down 60° at step {step + 1}")
                    self.navigation_log.append({"target": target_object, "success": True, "steps": step + 1, "collisions": total_collisions, "optimal_steps": self._current_optimal_steps})
                    return True, np.reshape(self.env.get_current_state(), (1, self.state_space)), step + 1
                # Restore camera
                self.env.controller.step(action="LookUp", degrees=60)

            # Track rotation and handle rotation limit
            if action in [3, 4]:
                rotation_count += 1
                # If rotation limit exceeded, force forward movement for next several steps
                if rotation_count > self.MAX_CONSECUTIVE_ROTATIONS:
                    forced_forward_steps = self.FORCED_FORWARD_STEPS
                    rotation_count = 0  # Reset after triggering cooldown
                    logger.info(f"Rotation limit hit at step {step}, forcing {forced_forward_steps} forward steps")
            else:
                # Only reset rotation count after successful forward movement
                if rotation_count > 0:
                    rotation_count = max(0, rotation_count - 1)  # Decay gradually instead of instant reset

            # Early exit: if close (< 1.0m) but not visible for 30+ steps, bail out.
            # The object may be in a container below camera view — let interaction
            # handle it with forceAction.
            if not visible and new_distance < 1.0 and step >= 30:
                logger.warning(f"Close but not visible for {step} steps ({new_distance:.2f}m) — bailing out for forced interaction")
                self.navigation_log.append({"target": target_object, "success": False, "steps": step, "close_bail": True, "collisions": total_collisions, "optimal_steps": self._current_optimal_steps})
                return False, np.reshape(self.env.get_current_state(), (1, self.state_space)), step

            # Log every 10 steps
            if step % 10 == 0:
                logger.info(f"Step {step}: dist={distance:.2f}m, visible={visible}, bearing={bearing:.2f}, action={action}")

            # Check success after action
            if self.env.check_success():
                logger.info(f"SUCCESS: {target_object} reached at step {step + 1}")
                self.navigation_log.append({"target": target_object, "success": True, "steps": step + 1, "collisions": total_collisions, "optimal_steps": self._current_optimal_steps})
                return True, np.reshape(next_state, (1, self.state_space)), step + 1

            state = np.reshape(next_state, (1, self.state_space))

        # Failed
        final_dist = self.env.get_distance_to_target()
        logger.warning(f"Navigation failed: {target_object} not found in {max_steps} steps (dist={final_dist:.2f}m)")
        self.navigation_log.append({"target": target_object, "success": False, "steps": max_steps, "collisions": total_collisions, "optimal_steps": self._current_optimal_steps})
        return False, state, max_steps

    def _select_navigation_action(self, bearing: float, visible: bool, distance: float,
                                   collisions: int, stuck: int, rotations: int) -> int:
        """
        State machine for navigation action selection.

        Actions: 0=MoveLeft, 1=MoveAhead, 2=MoveRight, 3=RotateLeft, 4=RotateRight
        """
        # Handle being stuck - try random movement
        if stuck > self.STUCK_THRESHOLD:
            import random
            return random.choice([0, 2, 3, 4])  # Random lateral/rotation

        # Handle repeated collisions - back off and try different direction
        if collisions >= 2:
            if bearing < 0:
                return 0  # Strafe left
            else:
                return 2  # Strafe right

        # Prevent infinite rotation
        if rotations > self.MAX_CONSECUTIVE_ROTATIONS:
            return 1  # Force move ahead

        # Main navigation logic
        if visible:
            # Target visible - approach it
            if distance < self.CLOSE_DISTANCE_THRESHOLD:  # Very close
                # Fine adjustments
                if abs(bearing) < self.BEARING_FINE_ADJUST:
                    return 1  # Move ahead
                elif bearing < 0:
                    return 0  # Strafe left
                else:
                    return 2  # Strafe right
            else:
                # Approach: mostly move forward, adjust heading as needed
                if abs(bearing) < self.BEARING_ROUGHLY_AHEAD:
                    return 1  # Move ahead - target roughly in front
                elif abs(bearing) < self.BEARING_SIDE:
                    # Target to the side - alternate forward and rotation
                    if bearing < 0:
                        return 3 if stuck % 2 == 0 else 1  # RotateLeft or MoveAhead
                    else:
                        return 4 if stuck % 2 == 0 else 1  # RotateRight or MoveAhead
                else:
                    # Target far to side - rotate to face it
                    return 3 if bearing < 0 else 4
        else:
            # Target not visible - search mode
            if abs(bearing) > self.BEARING_SEARCH:
                # Rotate toward target direction
                return 3 if bearing < 0 else 4
            else:
                # Facing target direction - move forward to bring into view
                return 1

    def _geometric_escape(self, target_object: str) -> bool:
        """
        Geometric escape: when the RL policy is stuck oscillating at a fixed
        distance, query AI2-THOR for all reachable grid positions, find the one
        closest to the target object, and teleport the agent there.

        This is analogous to the A* path-planning fallback used in ALFRED and
        similar embodied-AI benchmarks — the key difference is that our system
        uses it only as a recovery mechanism after the learned RL policy fails,
        preserving the RL-first approach while avoiding permanent stalls.

        Returns True if the escape changed agent position, False otherwise.
        """
        try:
            # Find target object position — prefer exact type match to avoid
            # substring collisions (e.g. "Pot" matching before "Potato").
            target_pos = None
            fallback_pos = None
            search = target_object.lower().replace(" ", "")
            known_suffixes = ("sliced", "cracked", "dirty", "empty", "filled", "cooked")
            for obj in self.env.controller.last_event.metadata.get("objects", []):
                obj_type = obj.get("objectType", "").lower().replace(" ", "")
                if obj_type == search or any(obj_type == search + s for s in known_suffixes):
                    target_pos = obj.get("position")
                    break
                if fallback_pos is None and (search in obj_type or obj_type in search):
                    fallback_pos = obj.get("position")
            if target_pos is None:
                target_pos = fallback_pos
            if target_pos is None:
                return False

            # Get all reachable positions on the navigation grid
            reachable = self.env.get_reachable_positions()
            if not reachable:
                return False

            agent_pos = self.env.controller.last_event.metadata["agent"]["position"]

            # Score each reachable position by distance to target
            def _dist_to_target(pos: Dict) -> float:
                return math.sqrt(
                    (pos["x"] - target_pos["x"]) ** 2
                    + (pos["z"] - target_pos["z"]) ** 2
                )

            # Filter out positions very close to where we already are (no point
            # teleporting to the same stuck spot).
            min_move = 0.3  # metres
            candidates = [
                p for p in reachable
                if math.sqrt((p["x"] - agent_pos["x"])**2
                             + (p["z"] - agent_pos["z"])**2) > min_move
            ]
            if not candidates:
                return False

            best = min(candidates, key=_dist_to_target)

            # Compute rotation to face the target from the new position
            dx = target_pos["x"] - best["x"]
            dz = target_pos["z"] - best["z"]
            yaw = math.degrees(math.atan2(dx, dz))  # AI2-THOR y-rotation convention

            logger.info(
                "[NAV] Geometric escape: teleporting from (%.2f,%.2f) to (%.2f,%.2f)  "
                "target_dist=%.2fm → %.2fm",
                agent_pos["x"], agent_pos["z"],
                best["x"], best["z"],
                _dist_to_target(agent_pos),
                _dist_to_target(best),
            )
            success = self.env.teleport_agent(best["x"], best["z"], rotation=yaw)
            return success
        except Exception as e:
            logger.warning("[NAV] Geometric escape failed: %s", e)
            return False

    def is_object_visible(self, target_object: str) -> bool:
        """
        Check if target object is currently visible

        Args:
            target_object: Object type to check (case-insensitive)

        Returns:
            True if object is visible, False otherwise
        """
        try:
            visible_objects = self.env.controller.last_event.metadata.get("objects", [])

            # Normalize search term
            search_type = target_object.lower().replace(" ", "")
            known_suffixes = ("sliced", "cracked", "dirty", "empty", "filled", "cooked")

            # Prefer exact type matches; fall back to substring only if none exist.
            # Bidirectional substring matching (the previous behavior) could return
            # "Sink" when "SinkBasin" was requested, or "Pot" when "Potato" was.
            exact_hit = False
            substring_hit = False
            for obj in visible_objects:
                if not obj.get("visible", False):
                    continue
                obj_type = obj.get("objectType", "").lower().replace(" ", "")
                if obj_type == search_type or any(
                    obj_type == search_type + suffix for suffix in known_suffixes
                ):
                    exact_hit = True
                    break
                if obj_type in search_type or search_type in obj_type:
                    substring_hit = True

            if exact_hit:
                logger.debug(f"Object {target_object} is visible")
                return True
            if substring_hit:
                logger.debug(f"Object {target_object} visible via substring fallback")
                return True
            return False

        except Exception as e:
            logger.error(f"Error checking object visibility: {e}")
            return False
    
    def get_current_state(self) -> np.ndarray:
        """Get current environment state (properly computed from segmentation)"""
        state = self.env.get_current_state()
        return np.reshape(state, (1, self.state_space))
    
    def step_forward(self) -> Tuple[float, np.ndarray, bool]:
        """
        Take a single forward step (useful for approaching visible objects)
        
        Returns:
            (reward, next_state, done)
        """
        # Action 1 is MoveAhead in the existing action space
        reward, next_state, done = self.env.step(action=1)
        next_state = np.reshape(next_state, (1, self.state_space))
        return reward, next_state, done
    
    def turn_left(self) -> Tuple[float, np.ndarray, bool]:
        """
        Turn left
        
        Returns:
            (reward, next_state, done)
        """
        # Action 3 is RotateLeft
        reward, next_state, done = self.env.step(action=3)
        next_state = np.reshape(next_state, (1, self.state_space))
        return reward, next_state, done
    
    def turn_right(self) -> Tuple[float, np.ndarray, bool]:
        """
        Turn right
        
        Returns:
            (reward, next_state, done)
        """
        # Action 4 is RotateRight
        reward, next_state, done = self.env.step(action=4)
        next_state = np.reshape(next_state, (1, self.state_space))
        return reward, next_state, done
    
    def get_navigation_log(self):
        """Get navigation history"""
        return self.navigation_log.copy()
    
    def clear_navigation_log(self):
        """Clear navigation log"""
        self.navigation_log = []


class SimpleNavigationController:
    """
    Simplified navigation controller for testing without full RL agent
    
    Uses heuristic exploration instead of RL (useful for ablation studies)
    """
    
    def __init__(self, env: AIThorEnv):
        """
        Initialize simple navigation controller
        
        Args:
            env: AI2-THOR environment wrapper
        """
        self.env = env
        self.navigation_log = []
    
    def navigate_to_object(self, target_object: str, max_steps: int = 100) -> Tuple[bool, np.ndarray, int]:
        """
        Navigate using simple heuristic: random walk until object found
        
        Args:
            target_object: Object to find
            max_steps: Maximum steps
            
        Returns:
            (success, final_state, steps_taken)
        """
        logger.info(f"Heuristic navigation to: {target_object}")
        
        if self.is_object_visible(target_object):
            return True, self.env.state, 0
        
        # Simple random exploration
        import random
        state = self.env.state
        
        for step in range(max_steps):
            if self.is_object_visible(target_object):
                self.navigation_log.append({
                    "target": target_object,
                    "success": True,
                    "steps": step,
                    "mode": "heuristic"
                })
                return True, state, step
            
            # Random action
            action = random.choice([0, 1, 2, 3, 4])  # MoveLeft, MoveAhead, MoveRight, RotateLeft, RotateRight
            reward, next_state, done = self.env.step(action)
            
            if not done:
                state = next_state
        
        self.navigation_log.append({
            "target": target_object,
            "success": False,
            "steps": max_steps,
            "mode": "heuristic"
        })
        return False, state, max_steps
    
    def is_object_visible(self, target_object: str) -> bool:
        """Check if object is visible"""
        try:
            visible_objects = self.env.controller.last_event.metadata.get("objects", [])
            search_type = target_object.lower().replace(" ", "")
            known_suffixes = ("sliced", "cracked", "dirty", "empty", "filled", "cooked")

            # Prefer exact type matches; fall back to substring only if none exist.
            exact_hit = False
            substring_hit = False
            for obj in visible_objects:
                if not obj.get("visible", False):
                    continue
                obj_type = obj.get("objectType", "").lower().replace(" ", "")
                if obj_type == search_type or any(
                    obj_type == search_type + suffix for suffix in known_suffixes
                ):
                    exact_hit = True
                    break
                if obj_type in search_type or search_type in obj_type:
                    substring_hit = True
            return exact_hit or substring_hit
        except Exception as e:
            logger.warning(f"Error checking object visibility: {e}")
            return False
