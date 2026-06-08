"""
AI2-THOR Environment Wrapper for Kitchen Navigation Training

Updated for AI2-THOR 5.x API (2024+)

Key features:
- Target-aware navigation (knows what object to find)
- Progress-based rewards (distance reduction)
- Non-terminal collisions (agent learns to recover)
- Success condition (target visible + distance <= SUCCESS_DISTANCE)
"""

import random
import math
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from ae_rl.rl.nav.state import StateModel
from ae_rl.sim.nav_reward import RewardModel
import cv2


class AIThorEnv:
    """
    AI2-THOR environment wrapper for kitchen navigation training.

    Compatible with AI2-THOR 5.x API.

    Key features:
    - Target-aware navigation (knows what object to find)
    - Progress-based rewards (distance reduction)
    - Non-terminal collisions (agent learns to recover)
    - Success condition (target visible + distance <= SUCCESS_DISTANCE)
    """

    # ========== CONFIGURATION CONSTANTS ==========
    # Navigation thresholds
    SUCCESS_DISTANCE = 1.2          # meters - distance to consider target reached
    INTERACTION_DISTANCE = 1.5      # meters - max distance for object interaction
    MAX_DISTANCE_NORMALIZATION = 10.0  # meters - for normalizing distance to [0, 1]

    # Episode limits
    DEFAULT_MAX_STEPS = 200         # Maximum steps per episode
    GRID_SIZE = 0.25                # Grid size for movement (meters)
    ROTATION_DEGREES = 45           # Degrees per rotation action
    LOOK_DOWN_DEGREES = 30          # Initial camera pitch

    # Render settings
    RENDER_WIDTH = 300
    RENDER_HEIGHT = 300
    FIELD_OF_VIEW = 90

    # Kitchen scenes in AI2-THOR (FloorPlan1-30 are kitchens)
    KITCHEN_SCENES = [f"FloorPlan{i}" for i in range(1, 11)]  # FloorPlan1-10

    # Common kitchen objects for navigation targets
    # Includes pickable objects + receptacles needed for interaction tasks
    KITCHEN_TARGETS = [
        # Pickable objects (food)
        "Apple", "Bread", "Lettuce", "Tomato", "Potato", "Egg",
        # Pickable objects (containers/dishes)
        "Cup", "Bowl", "Plate", "Mug", "Pot", "Pan",
        # Pickable objects (utensils)
        "Knife", "Fork", "Spoon", "Spatula", "ButterKnife",
        # Pickable objects (other)
        "SoapBottle", "SaltShaker", "PepperShaker",
        # Receptacles (appliances)
        "Fridge", "Microwave", "Toaster", "CoffeeMachine", "StoveBurner",
        # Receptacles (surfaces)
        "CounterTop", "DiningTable", "Sink", "SinkBasin",
        # Receptacles (storage)
        "Cabinet", "Drawer", "GarbageCan"
    ]

    def __init__(self, numStates, test=0, headless=False):
        """
        Initialize AI2-THOR environment.

        Args:
            numStates: Number of grid rows for state representation
            test: 0 for training scenes (FloorPlan1-5), 1 for test scenes (FloorPlan6-10)
            headless: If True, run without display (for server/training)
        """
        # State tracking
        self.collisionCounter = 0
        self.event = None
        self.done = None
        self.state = None
        self.stateModel = StateModel(numStates)
        self.rewardModel = RewardModel()

        self.counter = 0
        self.captureNo = 0

        # Scene configuration
        if test == 0:
            self.scenes = self.KITCHEN_SCENES[:5]  # Train on FloorPlan1-5
        else:
            self.scenes = self.KITCHEN_SCENES[5:10]  # Test on FloorPlan6-10

        self.sceneNo = 0

        # Target navigation
        self.target_object = None
        self.target_object_id = None
        self.prev_distance = None
        self.success_distance = self.SUCCESS_DISTANCE

        # Episode tracking
        self.step_count = 0
        self.max_steps = self.DEFAULT_MAX_STEPS
        self.episode_successes = 0
        self.episode_collisions = 0

        # Initialize Controller for AI2-THOR 5.x
        # Simplified for headless reliability - reduce render complexity
        #
        # Platform options:
        # - "CloudRendering": GPU-accelerated headless (requires NVIDIA)
        # - "Linux64": Standard Linux build with display
        # - None: Auto-detect (may cause version jump during startup)
        #
        # Using explicit platform avoids the "old version → new version" jump
        # that occurs when AI2-THOR auto-detects and potentially retries platforms
        # AI2-THOR 5.x expects a platform class here, not the string name.
        selected_platform = CloudRendering if headless else None

        self.controller = Controller(
            width=self.RENDER_WIDTH,
            height=self.RENDER_HEIGHT,
            fieldOfView=self.FIELD_OF_VIEW,
            renderDepthImage=False,  # Disable depth to speed up initialization
            renderInstanceSegmentation=True,
            snapToGrid=False,
            visibilityDistance=self.INTERACTION_DISTANCE,
            headless=headless,
            agentMode="default",
            server_timeout=600.0,  # Increased timeout for slower systems
            platform=selected_platform,
        )
        # Start the scene
        self.event = self.controller.reset(scene=self.scenes[self.sceneNo],
                                           gridSize=self.GRID_SIZE)
        # Initial camera setup - look down slightly
        self.event = self.controller.step(
            action='LookDown',
            degrees=self.LOOK_DOWN_DEGREES
        )

        self.gridMaxNum = numStates
        # Actions: 0=MoveLeft, 1=MoveAhead, 2=MoveRight, 3=RotateLeft, 4=RotateRight
        # Removed MoveBack to encourage forward exploration
        self.actionCommands = ['MoveLeft', 'MoveAhead', 'MoveRight', 'RotateLeft', 'RotateRight']
        self.action_space = len(self.actionCommands)

    def get_objects_in_scene(self):
        """Get list of all interactable objects in current scene."""
        objects = []
        for obj in self.event.metadata['objects']:
            objects.append({
                'objectType': obj['objectType'],
                'objectId': obj['objectId'],
                'position': obj['position'],
                'visible': obj['visible'],
                'distance': obj.get('distance', None)
            })
        return objects

    def select_random_target(self):
        """Select a random navigable target from the scene."""
        available_targets = []
        for obj in self.event.metadata['objects']:
            if obj['objectType'] in self.KITCHEN_TARGETS:
                available_targets.append(obj)

        if not available_targets:
            # Fallback: use any pickupable object
            for obj in self.event.metadata['objects']:
                if obj.get('pickupable', False):
                    available_targets.append(obj)

        if available_targets:
            target = random.choice(available_targets)
            self.target_object = target['objectType']
            self.target_object_id = target['objectId']
            return True
        return False

    def set_target(self, target_type):
        """Set a specific target object type."""
        matches = [
            obj for obj in self.event.metadata['objects']
            if obj['objectType'] == target_type
        ]
        if not matches:
            return False

        agent_pos = self.event.metadata['agent']['position']

        def _sort_key(obj):
            pos = obj.get('position', {})
            dist = math.sqrt(
                (agent_pos['x'] - pos.get('x', 0.0)) ** 2 +
                (agent_pos['z'] - pos.get('z', 0.0)) ** 2
            )
            return (
                0 if obj.get('visible', False) else 1,
                dist,
                obj.get('objectId', ''),
            )

        target = sorted(matches, key=_sort_key)[0]
        self.target_object = target['objectType']
        self.target_object_id = target['objectId']
        return True

    def get_distance_to_target(self):
        """Calculate Euclidean distance from agent to target object."""
        if self.target_object_id is None:
            return None

        agent_pos = self.event.metadata['agent']['position']

        for obj in self.event.metadata['objects']:
            if obj['objectId'] == self.target_object_id:
                target_pos = obj['position']
                distance = math.sqrt(
                    (agent_pos['x'] - target_pos['x']) ** 2 +
                    (agent_pos['z'] - target_pos['z']) ** 2
                )
                return distance
        return None

    def get_target_bearing(self):
        """
        Calculate bearing angle from agent's heading to target.

        Returns:
            Angle in range [-pi, pi] where:
            - 0 = target directly ahead
            - positive = target to the right
            - negative = target to the left
            Returns 0 if target not found.
        """
        if self.target_object_id is None:
            return 0.0

        agent_pos = self.event.metadata['agent']['position']
        agent_rot = self.event.metadata['agent']['rotation']['y']  # degrees

        for obj in self.event.metadata['objects']:
            if obj['objectId'] == self.target_object_id:
                target_pos = obj['position']

                # Vector from agent to target
                dx = target_pos['x'] - agent_pos['x']
                dz = target_pos['z'] - agent_pos['z']

                # Angle to target (in world coordinates)
                angle_to_target = math.atan2(dx, dz)  # radians

                # Agent's heading (convert to radians)
                agent_heading = math.radians(agent_rot)

                # Relative bearing (target angle - agent heading)
                bearing = angle_to_target - agent_heading

                # Normalize to [-pi, pi]
                while bearing > math.pi:
                    bearing -= 2 * math.pi
                while bearing < -math.pi:
                    bearing += 2 * math.pi

                return bearing
        return 0.0

    def get_target_info(self):
        """
        Get target information for state representation.

        Returns:
            dict with:
            - bearing: normalized angle to target [-1, 1]
            - distance: normalized distance [0, 1]
            - visible: 1 if visible, 0 otherwise
        """
        bearing = self.get_target_bearing()
        distance = self.get_distance_to_target()
        visible = self.is_target_visible()

        # Normalize bearing to [-1, 1] (divide by pi)
        bearing_norm = bearing / math.pi if bearing else 0.0

        # Normalize distance to [0, 1] (cap at MAX_DISTANCE_NORMALIZATION meters)
        distance_norm = min(distance / self.MAX_DISTANCE_NORMALIZATION, 1.0) if distance else 1.0

        return {
            'bearing': bearing_norm,
            'distance': distance_norm,
            'visible': 1.0 if visible else 0.0
        }

    def is_target_visible(self):
        """Check if target object is currently visible."""
        if self.target_object_id is None:
            return False

        for obj in self.event.metadata['objects']:
            if obj['objectId'] == self.target_object_id:
                return obj['visible']
        return False

    def check_success(self):
        """Check if navigation goal is achieved."""
        if self.target_object_id is None:
            return False

        distance = self.get_distance_to_target()
        visible = self.is_target_visible()

        # Standard success: visible and close enough
        if visible and distance is not None and distance <= self.success_distance:
            return True

        # Fallback for surfaces/receptacles below camera view — distance-only check
        # CounterTop/tables use 1.2m (large surfaces); floor receptacles use 0.8m
        SURFACE_THRESHOLDS = {
            'StoveBurner': 0.8, 'SinkBasin': 0.8, 'Bathtub': 0.8, 'BathtubBasin': 0.8,
            'CounterTop': 1.2, 'DiningTable': 1.2, 'CoffeeTable': 1.2, 'Desk': 1.2,
            'Faucet': 1.5, 'Sink': 1.5,
        }
        threshold = SURFACE_THRESHOLDS.get(self.target_object)
        if threshold is not None and distance is not None and distance <= threshold:
            return True

        return False

    def get_agent_position(self):
        """Get agent's current position."""
        pos = self.event.metadata['agent']['position']
        return (pos['x'], pos['z'])

    def reset(self, target_type=None):
        """
        Reset environment for new episode.

        Args:
            target_type: Specific object type to navigate to (optional)

        Returns:
            Initial state
        """
        # Cycle through scenes for variety
        self.sceneNo = (self.sceneNo + 1) % len(self.scenes)

        # Reset the scene using modern API
        self.event = self.controller.reset(scene=self.scenes[self.sceneNo])
        print(f"SCENE: {self.scenes[self.sceneNo]}")

        # Randomize agent starting rotation
        degrees = random.choice([-90, -45, 0, 45, 90])
        self.event = self.controller.step(action='RotateRight', degrees=degrees)
        self.event = self.controller.step(action='LookDown', degrees=self.LOOK_DOWN_DEGREES)

        # Select target object
        if target_type and self.set_target(target_type):
            pass
        else:
            self.select_random_target()

        print(f"TARGET: {self.target_object}")

        # Reset tracking
        self.collisionCounter = 0
        self.step_count = 0
        self.prev_distance = self.get_distance_to_target()
        self.rewardModel.reset_episode()

        # Set initial target info
        target_info = self.get_target_info()
        self.stateModel.set_target_info(
            target_info['bearing'],
            target_info['distance'],
            target_info['visible']
        )
        self.state = self.stateModel.resetState()

        return self.state

    def performAction(self, actionCommand, degree=None):
        """Execute an action in the environment."""
        if degree is None:
            self.event = self.controller.step(action=actionCommand)
        else:
            self.event = self.controller.step(action=actionCommand, degrees=degree)

    def step(self, action, prevState=None, prevReward=0, back=False):
        """
        Execute action and return new state, reward, done.

        Args:
            action: Action index (0-4)
            prevState: Previous state (unused in new system)
            prevReward: Previous reward (unused in new system)
            back: Whether to move back first (collision recovery)

        Returns:
            reward, state, done
        """
        self.step_count += 1

        # Collision recovery: step back before new action
        if back:
            self.performAction('MoveBack')

        # Execute the action
        degree = self.ROTATION_DEGREES if action in [3, 4] else None
        self.performAction(self.actionCommands[action], degree)

        # Get new state and compute reward
        state, reward, done = self.run_frame(action)

        return reward, state, done

    def run_frame(self, action):
        """
        Process frame after action, compute reward.

        Returns:
            state, reward, done
        """
        # Update target info in state model FIRST
        target_info = self.get_target_info()
        self.stateModel.set_target_info(
            target_info['bearing'],
            target_info['distance'],
            target_info['visible']
        )

        # Get visual state from instance segmentation (modern API)
        segImg = self.event.instance_segmentation_frame
        if segImg is None:
            self.event = self.controller.step(action="Pass")
            segImg = self.event.instance_segmentation_frame
        if segImg is None:
            return (np.zeros(self.stateModel.arrSize * self.stateModel.gridMaxNum + 3),
                    0, False)
        segmap = self.convertToSegmap(segImg)
        state = self.stateModel.getStateFlat(segmap)  # Returns 75-dim state

        # Check conditions
        collision = not self.event.metadata['lastActionSuccess']
        curr_distance = self.get_distance_to_target()
        target_visible = self.is_target_visible()
        reached_target = self.check_success()
        agent_position = self.get_agent_position()

        # Compute reward using new system
        reward, reward_breakdown = self.rewardModel.compute_navigation_reward(
            prev_distance=self.prev_distance,
            curr_distance=curr_distance,
            target_visible=target_visible,
            action=action,
            collision=collision,
            agent_position=agent_position,
            reached_target=reached_target
        )

        # Update tracking
        self.prev_distance = curr_distance
        if collision:
            self.collisionCounter += 1

        # Determine if episode is done
        done = False

        # Success condition
        if reached_target:
            done = True
            print(f"SUCCESS! Reached {self.target_object} in {self.step_count} steps")

        # Max steps reached
        elif self.step_count >= self.max_steps:
            done = True
            print(f"MAX STEPS reached. Distance to target: {curr_distance:.2f}m")

        # NOTE: Collision does NOT terminate episode anymore
        # This allows agent to learn recovery behavior

        return state, reward, done

    def convertToSegmap(self, segImg):
        """Convert segmentation image to object map."""
        # Modern API uses instance_colors instead of color_to_object_id
        color_map = self.event.instance_masks

        # Fallback to color mapping if instance_masks not available
        if color_map is None:
            color_map = getattr(self.event, 'color_to_object_id', {})

        segmap = [["" for x in range(segImg.shape[0])] for j in range(segImg.shape[1])]

        for i in range(segImg.shape[0]):
            for j in range(segImg.shape[1]):
                try:
                    pixel_color = tuple(segImg[i][j][:3])  # RGB only, ignore alpha if present
                    # Try to find object ID from color
                    if hasattr(self.event, 'color_to_object_id') and self.event.color_to_object_id:
                        segmap[i][j] = self.event.color_to_object_id.get(pixel_color, "Wall")
                    else:
                        segmap[i][j] = "Unknown"
                except Exception:
                    segmap[i][j] = "Wall"
        return np.array(segmap)

    def get_current_state(self):
        """
        Get current visual state from segmentation (without taking an action).

        Returns:
            75-dim state array (72 grid + 3 target info)
        """
        # Update target info
        target_info = self.get_target_info()
        self.stateModel.set_target_info(
            target_info['bearing'],
            target_info['distance'],
            target_info['visible']
        )

        # Sync with controller's latest event — interactions call controller.step()
        # directly so self.event can become stale between tasks
        self.event = self.controller.last_event

        # Get visual state from current frame (modern API)
        segImg = self.event.instance_segmentation_frame
        if segImg is None:
            # Frame not ready yet — take a Pass step to force a render
            self.event = self.controller.step(action="Pass")
            segImg = self.event.instance_segmentation_frame
        if segImg is None:
            return np.zeros(self.stateModel.arrSize * self.stateModel.gridMaxNum + 3)
        segmap = self.convertToSegmap(segImg)
        state = self.stateModel.getStateFlat(segmap)

        return state

    def get_episode_stats(self):
        """Get statistics for current episode."""
        return {
            'steps': self.step_count,
            'collisions': self.collisionCounter,
            'target': self.target_object,
            'scene': self.scenes[self.sceneNo],
            'final_distance': self.get_distance_to_target(),
            'target_visible': self.is_target_visible(),
            'success': self.check_success()
        }

    def get_reachable_positions(self):
        """Get all reachable positions in current scene."""
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata["actionReturn"]

    def compute_shortest_path_steps(self, target_object: str) -> int:
        """
        Compute the geodesic shortest path (in grid steps) from the agent's
        current position to the nearest reachable position adjacent to the
        target object, using BFS on the reachable-position grid.

        This is used as the ground-truth optimal path length (l_i) for SPL
        computation per Anderson et al. 2018.

        Returns:
            Number of grid steps on the shortest path, or 0 if unreachable.
        """
        from collections import deque

        positions = self.get_reachable_positions()
        if not positions:
            return 0

        grid = self.controller.last_event.metadata.get("gridSize", 0.25)

        def _snap(x, z):
            return (round(round(x / grid) * grid, 3),
                    round(round(z / grid) * grid, 3))

        # Build adjacency set for BFS (4-connected grid)
        pos_set = set()
        for p in positions:
            pos_set.add(_snap(p["x"], p["z"]))

        # Agent start position
        agent = self.controller.last_event.metadata["agent"]["position"]
        start = _snap(agent["x"], agent["z"])

        # Find target object position — prefer exact type match to avoid
        # substring collisions (e.g. "Sink" matching before "SinkBasin",
        # or "Pot" matching before "Potato"), which would corrupt SPL.
        target_pos = None
        fallback_pos = None
        search = target_object.lower().replace(" ", "")
        known_suffixes = ("sliced", "cracked", "dirty", "empty", "filled", "cooked")
        for obj in self.controller.last_event.metadata.get("objects", []):
            obj_type = obj.get("objectType", "").lower().replace(" ", "")
            if obj_type == search or any(obj_type == search + s for s in known_suffixes):
                target_pos = obj.get("position")
                break
            if fallback_pos is None and (search in obj_type or obj_type in search):
                fallback_pos = obj.get("position")
        if target_pos is None:
            target_pos = fallback_pos
        if target_pos is None:
            return 0

        # Goal: any reachable position within interaction distance of target
        # Use the same thresholds as check_success()
        tx, tz = target_pos["x"], target_pos["z"]
        interaction_dist = 1.5  # metres — matches nav success threshold

        goal_set = set()
        for pos in pos_set:
            dx = pos[0] - tx
            dz = pos[1] - tz
            if (dx * dx + dz * dz) <= interaction_dist * interaction_dist:
                goal_set.add(pos)

        if not goal_set:
            # No reachable position near target — use closest reachable
            best = min(pos_set, key=lambda p: (p[0]-tx)**2 + (p[1]-tz)**2)
            goal_set = {best}

        if start in goal_set:
            return 0

        # BFS on 4-connected grid
        directions = [(grid, 0), (-grid, 0), (0, grid), (0, -grid)]
        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            (cx, cz), steps = queue.popleft()
            for dx, dz in directions:
                nx, nz = round(cx + dx, 3), round(cz + dz, 3)
                npos = (nx, nz)
                if npos in goal_set:
                    return steps + 1
                if npos in pos_set and npos not in visited:
                    visited.add(npos)
                    queue.append((npos, steps + 1))

        return 0  # Unreachable

    def teleport_agent(self, x, z, rotation=0):
        """Teleport agent to specific position."""
        self.event = self.controller.step(
            action="Teleport",
            position=dict(x=x, y=0.9, z=z),
            rotation=dict(x=0, y=rotation, z=0)
        )
        return self.event.metadata['lastActionSuccess']

    def get_object_metadata(self, object_id):
        """Get detailed metadata for a specific object."""
        for obj in self.event.metadata['objects']:
            if obj['objectId'] == object_id:
                return obj
        return None

    def close(self):
        """Clean up controller."""
        self.controller.stop()


# Convenience function to create environment
def make_env(numStates=6, test=0, headless=False):
    """
    Factory function to create AI2-THOR environment.

    Args:
        numStates: Number of grid rows for state representation
        test: 0 for training, 1 for testing
        headless: Run without display

    Returns:
        AIThorEnv instance
    """
    return AIThorEnv(numStates, test=test, headless=headless)
