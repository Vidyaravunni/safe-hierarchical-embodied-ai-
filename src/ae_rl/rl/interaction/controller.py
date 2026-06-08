"""
Interaction RL Controller — DDQN-based object interaction for AI2-THOR.

Replaces the hardcoded affordance-based InteractionController with a full
Reinforcement Learning approach. A DDQN agent learns the best micro-strategy
to successfully execute each interaction (pick, place, open, close, toggle,
slice, cook, break) in the AI2-THOR kitchen environment.

Architecture:
  - State:   ~26-dim vector (object affordances, agent state, intent context)
  - Actions: 8 micro-strategies (direct, forced, move_closer, step_back,
             look_down, look_up, rotate_left, rotate_right)
  - Reward:  +1.0 success / -0.1 per step / -0.5 persistent failure
  - Learning: online — trains from every interaction attempt

This class has the SAME public interface as InteractionController so it is
a drop-in replacement in task_executor.py and kitchen_demo_floorplan1.py.
"""

import os
import time
import random
import logging
import pickle
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional, List
from ae_rl.execution.safety import check_action_safety
from ae_rl.execution.kitchen_safety import (
    build_kitchen_snapshot,
    check_cook_scene_safety,
    check_place_scene_safety,
    check_toggle_scene_safety,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Lazy-import TensorFlow so the module can be imported even without GPU ────
_tf   = None
_keras = None

def _get_tf():
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        import keras
        _tf   = tf
        _keras = keras
    return _tf, _keras


# ─────────────────────────────────────────────────────────────────────────────
#  State encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

# All intents handled by this controller
INTERACTION_INTENTS = ["pick", "place", "open", "close", "toggle", "slice", "cook", "break"]
INTENT_TO_IDX = {intent: i for i, intent in enumerate(INTERACTION_INTENTS)}

# Common object type vocabulary (top-50 AI2-THOR kitchen objects)
OBJECT_VOCAB = [
    "Apple","AppleSliced","Bread","BreadSliced","Egg","EggCracked",
    "Lettuce","LettuceSliced","Potato","PotatoSliced","Tomato","TomatoSliced",
    "Mug","Cup","Bowl","Plate","Pan","Pot","WineBottle","Bottle",
    "Knife","ButterKnife","Fork","Spoon","Spatula","Ladle",
    "Fridge","Microwave","Toaster","CoffeeMachine","StoveBurner","StoveKnob",
    "SinkBasin","Faucet","CounterTop","DiningTable","Cabinet","Drawer","Shelf",
    "Lamp","LightSwitch","GarbageCan","SoapBottle","SaltShaker","PepperShaker",
    "Book","CellPhone","RemoteControl","Statue","Vase",
]
OBJ_TO_IDX = {obj.lower(): i for i, obj in enumerate(OBJECT_VOCAB)}
OBJ_VOCAB_SIZE = len(OBJECT_VOCAB)

# Micro-action definitions
MICRO_ACTIONS = [
    "direct",          # 0: execute action directly (standard params)
    "forced",          # 1: execute with forceAction=True
    "move_closer",     # 2: MoveAhead × 2 then execute
    "step_back",       # 3: MoveBack × 1 then execute
    "look_down",       # 4: LookDown 30° then execute
    "look_up",         # 5: LookUp 30° then execute
    "rotate_left",     # 6: RotateLeft 45° then execute
    "rotate_right",    # 7: RotateRight 45° then execute
]
NUM_MICRO_ACTIONS = len(MICRO_ACTIONS)

# State vector size
#  intent one-hot:     8
#  target obj one-hot: 50
#  held obj one-hot:   50
#  affordances:        8  (pickupable,openable,receptacle,toggleable,sliceable,cookable,breakable,fillable)
#  object state:       6  (visible, distance_norm, is_open, is_cooked, is_sliced, is_toggled)
#  agent state:        3  (is_holding, holding_matches_target, step_norm)
#  last_success:       1
INTERACTION_STATE_SIZE = 8 + OBJ_VOCAB_SIZE + OBJ_VOCAB_SIZE + 8 + 6 + 3 + 1


# ─────────────────────────────────────────────────────────────────────────────
#  Interaction DDQN
# ─────────────────────────────────────────────────────────────────────────────

class InteractionDDQN:
    """
    Double DQN for interaction micro-strategy selection.

    Learns which micro-action (direct, forced, move_closer, …) achieves the
    highest success rate for each (intent × object × context) combination.
    """

    def __init__(
        self,
        state_size: int = INTERACTION_STATE_SIZE,
        action_size: int = NUM_MICRO_ACTIONS,
        learning_rate: float = 0.0005,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.10,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
    ):
        self.state_size   = state_size
        self.action_size  = action_size
        self.lr           = learning_rate
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size   = batch_size

        self.model      = self._build_model()
        self.target_net = self._build_model()
        self.update_target()

    def _build_model(self):
        tf, keras = _get_tf()
        inp = keras.Input(shape=(self.state_size,))
        x   = keras.layers.Dense(256, activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(1e-4))(inp)
        x   = keras.layers.Dense(128, activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x   = keras.layers.Dense(64,  activation="relu")(x)
        out = keras.layers.Dense(self.action_size, activation="linear")(x)
        m   = keras.Model(inputs=inp, outputs=out)
        m.compile(loss="mse",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        return m

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return int(np.argmax(q))

    def replay(self, minibatch: list):
        """Double DQN experience replay."""
        states      = np.array([t[0] for t in minibatch])
        actions     = np.array([t[1] for t in minibatch])
        rewards     = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones       = np.array([t[4] for t in minibatch])

        q      = self.model.predict(states,      verbose=0)
        next_q = self.model.predict(next_states, verbose=0)
        tgt_q  = self.target_net.predict(next_states, verbose=0)

        for i in range(len(minibatch)):
            if dones[i]:
                q[i, actions[i]] = rewards[i]
            else:
                best = int(np.argmax(next_q[i]))
                q[i, actions[i]] = rewards[i] + self.gamma * tgt_q[i, best]

        self.model.fit(states, q, epochs=1, verbose=0)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.set_weights(self.model.get_weights())

    def save(self, path: str = "outputs/weights/interaction_rl"):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save_weights(path)

    def load(self, path: str = "outputs/weights/interaction_rl"):
        status = self.model.load_weights(path)
        if hasattr(status, "expect_partial"):
            status.expect_partial()
        status2 = self.target_net.load_weights(path)
        if hasattr(status2, "expect_partial"):
            status2.expect_partial()
        self.epsilon = self.epsilon_min
        logger.info(f"InteractionDDQN weights loaded from {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  State builder
# ─────────────────────────────────────────────────────────────────────────────

def _obj_one_hot(obj_type: Optional[str]) -> np.ndarray:
    v = np.zeros(OBJ_VOCAB_SIZE, dtype=np.float32)
    if obj_type:
        idx = OBJ_TO_IDX.get(obj_type.lower().strip(), -1)
        if idx >= 0:
            v[idx] = 1.0
    return v

def build_interaction_state(
    intent: str,
    target_type: Optional[str],
    receptacle_type: Optional[str],
    obj_meta: Optional[Dict],
    controller,
    step: int = 0,
    max_steps: int = 10,
    last_success: bool = False,
) -> np.ndarray:
    """
    Build the interaction state vector for the DDQN.

    Components (total = INTERACTION_STATE_SIZE):
      [0:8]      intent one-hot
      [8:58]     target object type one-hot
      [58:108]   held object type one-hot
      [108:116]  object affordances
      [116:122]  object state flags
      [122:125]  agent state
      [125]      last_action_success
    """
    # 1. Intent one-hot (8)
    intent_vec = np.zeros(8, dtype=np.float32)
    idx = INTENT_TO_IDX.get(intent, -1)
    if idx >= 0:
        intent_vec[idx] = 1.0

    # 2. Target object one-hot (50)
    target_vec = _obj_one_hot(target_type)

    # 3. Held object one-hot (50)
    held_type = None
    try:
        inv = controller.last_event.metadata.get("inventoryObjects", [])
        if inv:
            held_id = inv[0]["objectId"]
            for o in controller.last_event.metadata.get("objects", []):
                if o["objectId"] == held_id:
                    held_type = o["objectType"]
                    break
    except Exception:
        pass
    held_vec = _obj_one_hot(held_type)

    # 4. Object affordances (8)
    afford = np.zeros(8, dtype=np.float32)
    if obj_meta:
        afford[0] = float(obj_meta.get("pickupable",  False))
        afford[1] = float(obj_meta.get("openable",    False))
        afford[2] = float(obj_meta.get("receptacle",  False))
        afford[3] = float(obj_meta.get("toggleable",  False))
        afford[4] = float(obj_meta.get("sliceable",   False))
        afford[5] = float(obj_meta.get("cookable",    False))
        afford[6] = float(obj_meta.get("breakable",   False))
        afford[7] = float(obj_meta.get("canFillWithLiquid", False))

    # 5. Object state flags (6)
    obj_state = np.zeros(6, dtype=np.float32)
    if obj_meta:
        try:
            agent_pos = controller.last_event.metadata["agent"]["position"]
            obj_pos   = obj_meta.get("position", {})
            dx = agent_pos["x"] - obj_pos.get("x", 0)
            dy = agent_pos["y"] - obj_pos.get("y", 0)
            dz = agent_pos["z"] - obj_pos.get("z", 0)
            dist = float(np.sqrt(dx*dx + dy*dy + dz*dz))
        except Exception:
            dist = 5.0
        obj_state[0] = float(obj_meta.get("visible", False))
        obj_state[1] = min(dist / 10.0, 1.0)
        obj_state[2] = float(obj_meta.get("isOpen",    False))
        obj_state[3] = float(obj_meta.get("isCooked",  False))
        obj_state[4] = float(obj_meta.get("isSliced",  False))
        obj_state[5] = float(obj_meta.get("isToggled", False))

    # 6. Agent state (3)
    agent_state = np.zeros(3, dtype=np.float32)
    agent_state[0] = float(held_type is not None)
    agent_state[1] = float(
        held_type is not None and target_type is not None and
        held_type.lower() == target_type.lower()
    )
    agent_state[2] = min(step / max(max_steps, 1), 1.0)

    # 7. Last success flag (1)
    last_succ = np.array([float(last_success)], dtype=np.float32)

    return np.concatenate([
        intent_vec, target_vec, held_vec,
        afford, obj_state, agent_state, last_succ
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Main RL Interaction Controller
# ─────────────────────────────────────────────────────────────────────────────

class InteractionRLController:
    """
    RL-based interaction controller.

    For each interaction request (pick/place/open/close/toggle/slice/cook/break),
    the DDQN agent selects a micro-strategy to execute, learns from the result,
    and improves over time via online experience replay.

    Public interface is identical to InteractionController so it is a
    drop-in replacement everywhere in the codebase.
    """

    WEIGHTS_PATH     = "outputs/weights/interaction_rl"
    BUFFER_PATH      = "outputs/weights/interaction_rl_buffer.pkl"
    MAX_DISTANCE     = 1.5     # metres — max interaction reach
    MAX_MICRO_STEPS  = 4       # max repositioning attempts per interaction
    REPLAY_AFTER_N   = 10      # train after every N interactions
    MIN_BUFFER       = 16      # minimum experiences before first replay
    BATCH_SIZE       = 32
    TARGET_UPDATE_N  = 50      # sync target net every N interactions

    def __init__(
        self,
        max_interaction_distance: float = 1.5,
        action_delay_sec: float = 0.5,
        online_learning: bool = True,
        load_weights: bool = True,
        eval_mode: bool = False,
    ):
        self.max_distance     = max_interaction_distance
        self.action_delay_sec = max(0.0, float(action_delay_sec))
        self.online_learning  = online_learning
        self.eval_mode        = eval_mode
        self.interaction_log  = []
        self.safety_events    = []

        # DDQN agent
        self.ddqn = InteractionDDQN()

        # Experience replay buffer
        self.replay_buffer  = deque(maxlen=10_000)
        self.interactions_n = 0
        self.successes_n    = 0

        # Load saved weights + buffer
        if load_weights:
            self._load()

        # Eval mode must always be greedy, even if no weights/buffer were loaded.
        if self.eval_mode:
            self.ddqn.epsilon = 0.0

        logger.info(
            f"InteractionRLController ready — "
            f"state_size={INTERACTION_STATE_SIZE}  "
            f"actions={NUM_MICRO_ACTIONS}  "
            f"epsilon={self.ddqn.epsilon:.2f}  "
            f"buffer={len(self.replay_buffer)}"
        )

    # ── Public interface (same as InteractionController) ──────────────────────

    def execute_interaction(self, action: Dict, controller, agent_state: Dict) -> Tuple[bool, str]:
        """Main dispatcher — mirrors InteractionController.execute_interaction()."""
        intent = action.get("intent")
        if intent == "pick":
            return self.pick_object(action.get("object"), controller)
        elif intent == "place":
            return self.place_object(action.get("object"), action.get("receptacle"), controller)
        elif intent == "open":
            return self.open_object(action.get("object"), controller)
        elif intent == "close":
            return self.close_object(action.get("object"), controller)
        elif intent == "toggle":
            return self.toggle_object(
                action.get("object"), controller,
                knob_index=action.get("knob_index"),
                knob_id=action.get("knob_id"),
            )
        elif intent == "slice":
            return self.slice_object(action.get("object"), controller)
        elif intent == "cook":
            return self.cook_object(action.get("object"), controller, action=action)
        elif intent == "break":
            return self.break_object(action.get("object"), controller)
        elif intent == "clean":
            return self.clean_object(action.get("object"), controller)
        elif intent == "pour":
            return self.pour_object(action.get("object"), action.get("receptacle"), controller)
        elif intent == "wait":
            return self.wait_action(action.get("seconds", 3))
        elif intent == "drop":
            return self.drop_object(controller)
        elif intent == "throw":
            return self.throw_object(action.get("object"), controller)
        else:
            return False, f"Unknown intent: {intent}"

    def pick_object(self, object_type: str, controller, auto_open_containers: bool = True) -> Tuple[bool, str]:
        """Pick object using RL micro-strategy selection."""
        logger.info(f"[IRL] pick {object_type}")

        # Check already holding
        if self.is_agent_holding_object(controller):
            held = self.get_held_object(controller)
            return False, f"[INVALID_ACTION] Already holding {held['objectType'] if held else 'something'}"

        # Auto-open containers if needed
        if auto_open_containers:
            is_acc, _, closed = self.is_object_accessible(object_type, controller)
            if not is_acc and closed:
                ok, msg, _ = self.ensure_object_accessible(object_type, controller)
                if not ok:
                    return False, f"Cannot access {object_type}: {msg}"

        obj = self.find_object_by_type(object_type, controller)
        force_pick = False
        if obj is None:
            obj = self.find_object_by_type(object_type, controller, allow_invisible=True)
            if obj is None:
                return False, f"[OBJECT_NOT_FOUND] {object_type} not found in scene"
            dist = self.get_object_distance(obj, controller)
            if dist > self.max_distance:
                return False, f"[OBJECT_NOT_FOUND] {object_type} not visible and too far ({dist:.2f}m)"
            force_pick = True

        if not obj.get("pickupable", False):
            return False, f"[INVALID_ACTION] {object_type} is not pickupable"

        safe, safety_msg, safety_meta = check_action_safety(
            "pick",
            obj_meta=obj,
            action={"intent": "pick", "object": object_type},
        )
        if not safe:
            self._record_safety_event("pick", object_type, safety_msg, safety_meta, success=False)
            return False, safety_msg

        # force_pick=True when object was not visible (e.g. on lower fridge shelf).
        # Pass it into the lambda so every micro-strategy attempt also uses forceAction.
        _fp = force_pick
        success, message = self._rl_execute(
            intent="pick",
            target_type=object_type,
            receptacle_type=None,
            obj_meta=obj,
            controller=controller,
            exec_fn=lambda force: controller.step(
                action="PickupObject",
                objectId=obj["objectId"],
                forceAction=force or _fp,
                manualInteract=False,
            ),
        )

        # Wall-cabinet fallback: if RL pick failed for a visible object in a Cabinet,
        # try LookUp so the high shelf is in frame, then force-pick, then restore.
        if not success and not force_pick:
            parents_ids = obj.get("parentReceptacles") or []
            if any("Cabinet" in p or "Drawer" in p for p in parents_ids):
                total_up = 0
                for look_deg in (30, 45):
                    controller.step(action="LookUp", degrees=look_deg)
                    total_up += look_deg
                    event = controller.step(
                        action="PickupObject",
                        objectId=obj["objectId"],
                        forceAction=True,
                        manualInteract=False,
                    )
                    if event.metadata["lastActionSuccess"]:
                        success = True
                        message = f"Pick {object_type} from cabinet after LookUp {total_up}°"
                        break
                controller.step(action="LookDown", degrees=total_up)

        # Fridge/container fallback: if RL pick failed and the object was inside
        # a container, try LookDown 45° then force-pick, then restore camera.
        if not success and force_pick:
            total_down = 0
            for look_deg in (45, 60):
                controller.step(action="LookDown", degrees=look_deg)
                total_down += look_deg
                event = controller.step(
                    action="PickupObject",
                    objectId=obj["objectId"],
                    forceAction=True,
                    manualInteract=False,
                )
                if event.metadata["lastActionSuccess"]:
                    success = True
                    message = f"Pick {object_type} succeeded after LookDown {total_down}°"
                    logger.info(f"[IRL] {message}")
                    break
            controller.step(action="LookUp", degrees=total_down)

        if not success:
            fb_success, fb_message = self._try_direct_pick_fallback(
                object_type,
                controller,
                preferred_obj=obj,
            )
            if fb_success:
                success = True
                message = fb_message

        if safety_msg:
            self._record_safety_event("pick", object_type, safety_msg, safety_meta, success=True)
            if success:
                return success, f"{safety_msg} | {message}"
        return success, message

    # Receptacles that use the aggressive look-down + PlaceObjectAtPoint fallback.
    # Cabinet/Drawer are above eye level so LookDown attempts fail fast, but
    # PlaceObjectAtPoint (last resort) places deterministically inside the container.
    _VESSEL_TYPES = frozenset({
        'Pot', 'Pan', 'Bowl', 'Cup', 'Mug', 'WineBottle', 'Plate',
        'Sink', 'SinkBasin', 'CoffeeMachine',
        'StoveBurner', 'Toaster', 'ToasterOven', 'Microwave',
        'Cabinet', 'Drawer', 'Safe', 'Box', 'Fridge',
    })

    def place_object(self, object_type: str, receptacle_type: str, controller) -> Tuple[bool, str]:
        """Place held object using RL micro-strategy, with vessel-specific fallback."""
        logger.info(f"[IRL] place {object_type} on {receptacle_type}")

        if not self.is_agent_holding_object(controller):
            return False, "[INVALID_ACTION] Not holding any object"
        held = self.get_held_object(controller)
        safe, safety_msg, safety_meta = check_action_safety(
            "place",
            obj_meta=held,
            action={"intent": "place", "object": object_type, "receptacle": receptacle_type},
            held_obj=held,
        )
        scene_msg = ""
        scene_meta: Dict = {}
        rec = None
        if receptacle_type.lower().replace(" ", "") == "stoveburner":
            snapshot = build_kitchen_snapshot(controller)
            scene_safe, scene_msg, scene_meta = check_place_scene_safety(
                {"intent": "place", "object": object_type, "receptacle": receptacle_type},
                snapshot,
            )
            if not scene_safe:
                self._record_safety_event(
                    "place",
                    object_type,
                    scene_msg,
                    {**(safety_meta or {}), **(scene_meta or {})},
                    success=False,
                )
                return False, scene_msg
            burner_id = scene_meta.get("selected_burner_id")
            if burner_id:
                rec = self.find_object_by_id(burner_id, controller)
        if not safe:
            self._record_safety_event(
                "place",
                object_type,
                safety_msg,
                {**(safety_meta or {}), **(scene_meta or {})},
                success=False,
            )
            return False, safety_msg

        if rec is None:
            rec = self.find_object_by_type(receptacle_type, controller, allow_invisible=True)
        if rec is None:
            return False, f"[OBJECT_NOT_FOUND] Receptacle {receptacle_type} not found"
        if not rec.get("receptacle", False):
            return False, f"[INVALID_ACTION] {receptacle_type} is not a receptacle"

        # Auto-open: ensure openable receptacles are open before placing
        if rec.get("openable", False) and not rec.get("isOpen", False):
            logger.info(f"[IRL] Auto-opening {receptacle_type} before placement")
            controller.step(action="OpenObject", objectId=rec["objectId"], forceAction=True)

        success, message = self._rl_execute(
            intent="place",
            target_type=object_type,
            receptacle_type=receptacle_type,
            obj_meta=rec,
            controller=controller,
            exec_fn=lambda force: controller.step(
                action="PutObject",
                objectId=rec["objectId"],
                forceAction=force,
                placeStationary=True,
            ),
        )

        if success:
            combined_meta = {**(safety_meta or {}), **(scene_meta or {})}
            notes = [msg for msg in (scene_msg, safety_msg) if msg]
            if notes:
                combined_msg = " | ".join(notes)
                self._record_safety_event("place", object_type, combined_msg, combined_meta, success=True)
                return success, f"{combined_msg} | {message}"
            return success, message

        # ── Vessel fallback: aggressive look-down + PlaceObjectAtPoint ────────
        if receptacle_type in self._VESSEL_TYPES:
            logger.info(f"[IRL] Vessel fallback for {receptacle_type} — trying deep look-down")
            held = self.get_held_object(controller)
            if held is None:
                return False, "Object was dropped during RL attempts"

            # Re-fetch receptacle (position may have updated)
            rec = self.find_object_by_type(receptacle_type, controller, allow_invisible=True) or rec

            # Track how much we look down so we can restore
            total_down = 0

            for look_deg in (45, 60):
                controller.step(action="LookDown", degrees=look_deg)
                total_down += look_deg
                event = controller.step(
                    action="PutObject",
                    objectId=rec["objectId"],
                    forceAction=True,
                    placeStationary=True,
                )
                if event.metadata["lastActionSuccess"]:
                    logger.info(f"[IRL] Vessel placement succeeded at LookDown {total_down}°")
                    self.interaction_log.append({
                        "action": "place", "object": object_type,
                        "receptacle": receptacle_type, "success": True,
                    })
                    # Restore camera
                    controller.step(action="LookUp", degrees=total_down)
                    self._apply_action_delay()
                    return True, f"Placed {object_type} in {receptacle_type} (vessel fallback)"

            # Restore camera before PlaceObjectAtPoint attempt
            if total_down > 0:
                controller.step(action="LookUp", degrees=total_down)

            # Last resort: PlaceObjectAtPoint — deterministic, no physics bounce
            logger.info(f"[IRL] Vessel fallback: trying PlaceObjectAtPoint for {receptacle_type}")
            vessel_pos = rec["position"]
            event = controller.step(
                action="PlaceObjectAtPoint",
                objectId=held["objectId"],
                position=vessel_pos,
            )
            if event.metadata["lastActionSuccess"]:
                logger.info(f"[IRL] PlaceObjectAtPoint succeeded for {receptacle_type}")
                self.interaction_log.append({
                    "action": "place", "object": object_type,
                    "receptacle": receptacle_type, "success": True,
                })
                self._apply_action_delay()
                return True, f"Placed {object_type} in {receptacle_type} (PlaceObjectAtPoint)"

            error = event.metadata.get("errorMessage", "unknown error")
            logger.warning(f"[IRL] All vessel placement strategies failed: {error}")
            self.interaction_log.append({
                "action": "place", "object": object_type,
                "receptacle": receptacle_type, "success": False,
            })

        # ── Last resort: direct forceAction PutObject for any receptacle ─────
        held = self.get_held_object(controller)
        if held is not None:
            logger.info(f"[IRL] Last-resort forceAction PutObject for {receptacle_type}")
            # Re-fetch receptacle
            rec = self.find_object_by_type(receptacle_type, controller, allow_invisible=True) or rec
            # Ensure open
            if rec.get("openable", False) and not rec.get("isOpen", False):
                controller.step(action="OpenObject", objectId=rec["objectId"], forceAction=True)
            event = controller.step(
                action="PutObject",
                objectId=rec["objectId"],
                forceAction=True,
                placeStationary=True,
            )
            if event.metadata["lastActionSuccess"]:
                logger.info(f"[IRL] Last-resort placement succeeded for {receptacle_type}")
                self.interaction_log.append({
                    "action": "place", "object": object_type,
                    "receptacle": receptacle_type, "success": True,
                })
                self._apply_action_delay()
                return True, f"Placed {object_type} in {receptacle_type} (last-resort force)"

        # ── Multi-candidate fallback: try other instances of the same type ────
        held = self.get_held_object(controller)
        if held is not None:
            tried_ids = {rec["objectId"]} if rec else set()
            all_candidates = self.find_all_objects_by_type(
                receptacle_type, controller, allow_invisible=True,
            )
            for alt_rec in all_candidates:
                if alt_rec.get("objectId", "") in tried_ids:
                    continue
                if not alt_rec.get("receptacle", False):
                    continue
                tried_ids.add(alt_rec["objectId"])
                logger.info(
                    "[IRL] Trying alternate %s: %s",
                    receptacle_type, alt_rec["objectId"],
                )
                if alt_rec.get("openable", False) and not alt_rec.get("isOpen", False):
                    controller.step(
                        action="OpenObject",
                        objectId=alt_rec["objectId"],
                        forceAction=True,
                    )
                event = controller.step(
                    action="PutObject",
                    objectId=alt_rec["objectId"],
                    forceAction=True,
                    placeStationary=True,
                )
                if event.metadata["lastActionSuccess"]:
                    logger.info(
                        "[IRL] Alternate %s placement succeeded: %s",
                        receptacle_type, alt_rec["objectId"],
                    )
                    self.interaction_log.append({
                        "action": "place", "object": object_type,
                        "receptacle": receptacle_type, "success": True,
                    })
                    self._apply_action_delay()
                    return True, f"Placed {object_type} in {receptacle_type} (alternate instance)"

        # ── Semantic receptacle fallback: when all instances of the target
        # receptacle reject the object (e.g. every Cabinet in FP3 is full or
        # has incompatible geometry), try semantically equivalent storage types.
        # This mirrors how humans adapt: "the cabinet is full, put it on the
        # shelf / counter instead."
        _STORAGE_FALLBACKS = {
            "Cabinet": ["Shelf", "Drawer", "CounterTop"],
            "Shelf":   ["Cabinet", "Drawer", "CounterTop"],
            "Drawer":  ["Cabinet", "Shelf", "CounterTop"],
        }
        fallback_types = _STORAGE_FALLBACKS.get(receptacle_type, [])
        held = self.get_held_object(controller)
        for fb_type in fallback_types:
            if held is None:
                break
            fb_recs = self.find_all_objects_by_type(
                fb_type, controller, allow_invisible=True,
            )
            for fb_rec in fb_recs:
                if not fb_rec.get("receptacle", False):
                    continue
                logger.info(
                    "[IRL] Semantic fallback: trying %s (%s) for %s",
                    fb_type, fb_rec["objectId"], object_type,
                )
                if fb_rec.get("openable", False) and not fb_rec.get("isOpen", False):
                    controller.step(
                        action="OpenObject",
                        objectId=fb_rec["objectId"],
                        forceAction=True,
                    )
                event = controller.step(
                    action="PutObject",
                    objectId=fb_rec["objectId"],
                    forceAction=True,
                    placeStationary=True,
                )
                if event.metadata["lastActionSuccess"]:
                    logger.info(
                        "[IRL] Semantic fallback succeeded: %s → %s (%s)",
                        object_type, fb_type, fb_rec["objectId"],
                    )
                    self.interaction_log.append({
                        "action": "place", "object": object_type,
                        "receptacle": fb_type, "success": True,
                    })
                    self._apply_action_delay()
                    return True, f"Placed {object_type} in {fb_type} (semantic fallback from {receptacle_type})"

        return False, message

    def open_object(self, object_type: str, controller) -> Tuple[bool, str]:
        """Open container using RL micro-strategy."""
        logger.info(f"[IRL] open {object_type}")
        obj = self.find_object_by_type(object_type, controller)
        if obj is None:
            # Fallback: find even if not in visibility cone — let RL move into range
            obj = self.find_object_by_type(object_type, controller, allow_invisible=True)
        if obj is None:
            return False, f"[OBJECT_NOT_FOUND] {object_type} not found in scene"
        if not obj.get("openable", False):
            return False, f"[INVALID_ACTION] {object_type} is not openable"
        if obj.get("isOpen", False):
            return True, f"{object_type} already open"

        return self._rl_execute(
            intent="open",
            target_type=object_type,
            receptacle_type=None,
            obj_meta=obj,
            controller=controller,
            exec_fn=lambda force: controller.step(
                action="OpenObject",
                objectId=obj["objectId"],
                forceAction=force,
            ),
        )

    def close_object(self, object_type: str, controller) -> Tuple[bool, str]:
        """Close container using RL micro-strategy."""
        logger.info(f"[IRL] close {object_type}")
        obj = self.find_object_by_type(object_type, controller)
        if obj is None:
            # Fallback: find even if not in visibility cone — let RL move into range
            obj = self.find_object_by_type(object_type, controller, allow_invisible=True)
        if obj is None:
            return False, f"[OBJECT_NOT_FOUND] {object_type} not found in scene"
        if not obj.get("openable", False):
            return False, f"[INVALID_ACTION] {object_type} is not closeable"
        if not obj.get("isOpen", True):
            return True, f"{object_type} already closed"

        return self._rl_execute(
            intent="close",
            target_type=object_type,
            receptacle_type=None,
            obj_meta=obj,
            controller=controller,
            exec_fn=lambda force: controller.step(
                action="CloseObject",
                objectId=obj["objectId"],
                forceAction=force,
            ),
        )

    def toggle_object(
        self, object_type: str, controller,
        knob_index: Optional[int] = None,
        knob_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Toggle appliance using RL micro-strategy.

        For StoveKnob, pass knob_index (0-3) or knob_id (e.g. "StoveKnob_0")
        from the action JSON to target the correct burner knob.
        """
        logger.info(f"[IRL] toggle {object_type} knob_index={knob_index} knob_id={knob_id}")

        # Resolve the specific StoveKnob by knob_id or knob_index if given
        obj = None
        scene_msg = ""
        scene_meta: Dict = {}
        if object_type.lower() == "stoveknob":
            snapshot = build_kitchen_snapshot(controller)
            scene_safe, scene_msg, scene_meta = check_toggle_scene_safety(
                {
                    "intent": "toggle",
                    "object": object_type,
                    "knob_index": knob_index,
                    "knob_id": knob_id,
                },
                snapshot,
            )
            if not scene_safe:
                # Auto-redirect to correct knob if the safety check recommends one
                rec_knob_idx = scene_meta.get("recommended_knob_index")
                if rec_knob_idx is not None:
                    logger.info(
                        "[IRL] toggle: redirecting from empty burner to recommended knob_index=%s",
                        rec_knob_idx,
                    )
                    snapshot = build_kitchen_snapshot(controller)
                    scene_safe, scene_msg, scene_meta = check_toggle_scene_safety(
                        {
                            "intent": "toggle",
                            "object": object_type,
                            "knob_index": int(rec_knob_idx),
                        },
                        snapshot,
                    )
                if not scene_safe:
                    self._record_safety_event("toggle", object_type, scene_msg, scene_meta, success=False)
                    return False, scene_msg
            selected_knob_id = scene_meta.get("selected_knob_id")
            if selected_knob_id:
                obj = self.find_object_by_id(selected_knob_id, controller)
            if knob_index is None and scene_meta.get("selected_knob_index", -1) >= 0:
                knob_index = int(scene_meta["selected_knob_index"])

        if object_type.lower() == "stoveknob" and (knob_id or knob_index is not None):
            all_objs = controller.last_event.metadata.get("objects", [])
            knobs = [o for o in all_objs if "stoveknob" in o.get("objectType","").lower()]
            if knob_id:
                # Match by exact objectId suffix or name
                for k in knobs:
                    if knob_id.lower() in k.get("objectId","").lower() or \
                       knob_id.lower() == k.get("objectType","").lower():
                        obj = k
                        break
            if obj is None and knob_index is not None and 0 <= knob_index < len(knobs):
                # Sort knobs by objectId for deterministic ordering
                knobs_sorted = sorted(knobs, key=lambda o: o.get("objectId",""))
                obj = knobs_sorted[knob_index]

        if obj is None:
            obj = self.find_object_by_type(object_type, controller)
        if obj is None:
            obj = self.find_object_by_type(object_type, controller, allow_invisible=True)
        if obj is None:
            return False, f"[OBJECT_NOT_FOUND] {object_type} not found in scene"
        if not obj.get("toggleable", False):
            return False, f"[INVALID_ACTION] {object_type} is not toggleable"

        current  = obj.get("isToggled", False)
        act_name = "ToggleObjectOff" if current else "ToggleObjectOn"
        _obj_id  = obj["objectId"]  # capture before lambda

        if scene_msg:
            return self._toggle_with_scene_note(
                object_type,
                obj,
                act_name,
                _obj_id,
                controller,
                scene_msg,
                scene_meta,
            )

        success, message = self._rl_execute(
            intent="toggle",
            target_type=object_type,
            receptacle_type=None,
            obj_meta=obj,
            controller=controller,
            exec_fn=lambda force: controller.step(
                action=act_name,
                objectId=_obj_id,
                forceAction=force,
            ),
        )
        if not success:
            fb_success, fb_message = self._try_direct_toggle_fallback(
                object_type,
                act_name,
                controller,
                preferred_obj=obj,
            )
            if fb_success:
                return True, fb_message
        return success, message

    def _toggle_with_scene_note(
        self,
        object_type: str,
        obj: Dict,
        act_name: str,
        obj_id: str,
        controller,
        scene_msg: str,
        scene_meta: Dict,
    ) -> Tuple[bool, str]:
        success, message = self._rl_execute(
            intent="toggle",
            target_type=object_type,
            receptacle_type=None,
            obj_meta=obj,
            controller=controller,
            exec_fn=lambda force: controller.step(
                action=act_name,
                objectId=obj_id,
                forceAction=force,
            ),
        )
        if not success:
            fb_success, fb_message = self._try_direct_toggle_fallback(
                object_type,
                act_name,
                controller,
                preferred_obj=obj,
            )
            if fb_success:
                success = True
                message = fb_message
        if scene_msg:
            self._record_safety_event("toggle", object_type, scene_msg, scene_meta, success=success)
            if success:
                return success, f"{scene_msg} | {message}"
        return success, message

    def slice_object(self, object_type: str, controller) -> Tuple[bool, str]:
        """Slice object using RL micro-strategy (requires holding knife)."""
        logger.info(f"[IRL] slice {object_type}")

        # AI2-THOR requires the agent to hold a Knife or ButcherKnife to slice.
        held = self.get_held_object(controller)
        if not held or held.get("objectType", "").lower() not in ("knife", "butcherknife"):
            held_name = held.get("objectType", "nothing") if held else "nothing"
            return False, (
                f"[INVALID_ACTION] Must be holding a Knife to slice "
                f"(currently holding {held_name}) — pick up Knife first"
            )

        obj = self.find_object_by_type(object_type, controller)
        if obj is None:
            obj = self.find_object_by_type(object_type, controller, allow_invisible=True)
        if obj is None:
            return False, f"[OBJECT_NOT_FOUND] {object_type} not found in scene"
        if not obj.get("sliceable", False):
            return False, f"[INVALID_ACTION] {object_type} is not sliceable"
        if obj.get("isSliced", False):
            return True, f"{object_type} already sliced"

        return self._rl_execute(
            intent="slice",
            target_type=object_type,
            receptacle_type=None,
            obj_meta=obj,
            controller=controller,
            exec_fn=lambda force: controller.step(
                action="SliceObject",
                objectId=obj["objectId"],
                forceAction=force,
            ),
        )

    def cook_object(self, object_type: str, controller, action: dict | None = None) -> Tuple[bool, str]:
        """Cook object using RL micro-strategy."""
        logger.info(f"[IRL] cook {object_type}")
        snapshot = build_kitchen_snapshot(controller)
        cook_action = dict(action) if action else {}
        cook_action.update({"intent": "cook", "object": object_type})
        scene_safe, scene_msg, scene_meta = check_cook_scene_safety(
            cook_action,
            snapshot,
        )
        if not scene_safe:
            self._record_safety_event("cook", object_type, scene_msg, scene_meta, success=False)
            return False, scene_msg

        # Search all objects including invisible (might be in Microwave/Pot)
        all_objs = controller.last_event.metadata.get("objects", [])
        search   = object_type.lower().replace(" ", "")
        variants = {search}
        if search == "egg":
            variants.add("eggcracked")

        candidates = [
            o for o in all_objs
            if any(v in o.get("objectType","").lower() for v in variants)
        ]
        if not candidates:
            return False, f"[OBJECT_NOT_FOUND] {object_type} not found"

        # Already cooked?
        for c in candidates:
            if c.get("isCooked", False):
                return True, f"{object_type} already cooked"

        obj = next((c for c in candidates if c.get("cookable", False)), candidates[0])

        success, message = self._rl_execute(
            intent="cook",
            target_type=object_type,
            receptacle_type=None,
            obj_meta=obj,
            controller=controller,
            exec_fn=lambda force: controller.step(
                action="CookObject",
                objectId=obj["objectId"],
                forceAction=True,
            ),
        )
        if scene_msg:
            self._record_safety_event("cook", object_type, scene_msg, scene_meta, success=success)
            if success:
                return success, f"{scene_msg} | {message}"
        return success, message

    def break_object(self, object_type: str, controller) -> Tuple[bool, str]:
        """Break/crack object using RL micro-strategy."""
        logger.info(f"[IRL] break {object_type}")
        obj = self.find_object_by_type(object_type, controller, allow_invisible=True)
        if obj is None:
            return False, f"[OBJECT_NOT_FOUND] {object_type} not found"
        if not obj.get("breakable", False):
            return False, f"[INVALID_ACTION] {object_type} is not breakable"
        if obj.get("isBroken", False):
            return True, f"{object_type} already broken"

        return self._rl_execute(
            intent="break",
            target_type=object_type,
            receptacle_type=None,
            obj_meta=obj,
            controller=controller,
            exec_fn=lambda force: controller.step(
                action="BreakObject",
                objectId=obj["objectId"],
                forceAction=True,
            ),
        )

    def clean_object(self, object_type: str, controller) -> Tuple[bool, str]:
        """Clean object via AI2-THOR CleanObject (requires object in SinkBasin under water)."""
        logger.info(f"[IRL] clean {object_type}")
        obj = self.find_object_by_type(object_type, controller, allow_invisible=True)
        if obj is None:
            return False, f"[OBJECT_NOT_FOUND] {object_type} not found"

        # If already clean, succeed immediately (no action needed)
        if not obj.get("isDirty", True):
            logger.info(f"[IRL] {object_type} is already clean")
            return True, f"{object_type} is already clean"

        # CleanObject with forceAction bypasses position/visibility requirements.
        # Bypass _rl_execute (which repositions agent) to avoid moving away from SinkBasin.
        event = controller.step(
            action="CleanObject",
            objectId=obj["objectId"],
            forceAction=True,
        )
        if event.metadata["lastActionSuccess"]:
            self.interaction_log.append({"action": "clean", "object": object_type, "success": True})
            return True, f"Cleaned {object_type}"

        error = event.metadata.get("errorMessage", "unknown error")
        # If the sim says "already clean", treat as success
        if "clean" in error.lower() and ("already" in error.lower() or "not dirty" in error.lower()):
            logger.info(f"[IRL] {object_type} reported already clean: {error}")
            return True, f"{object_type} is already clean"

        self.interaction_log.append({"action": "clean", "object": object_type, "success": False})
        return False, f"Clean failed: {error}"

    def pour_object(self, object_type: str, receptacle_type: str, controller) -> Tuple[bool, str]:
        """Pour liquid from held object into receptacle (FillObjectWithLiquid)."""
        logger.info(f"[IRL] pour {object_type} → {receptacle_type}")
        if not self.is_agent_holding_object(controller):
            return False, "[INVALID_ACTION] Not holding any object to pour"

        rec = self.find_object_by_type(receptacle_type, controller, allow_invisible=True)
        if rec is None:
            # If no receptacle target, empty the held object (EmptyLiquid)
            return self._rl_execute(
                intent="pour",
                target_type=object_type,
                receptacle_type=receptacle_type,
                obj_meta=None,
                controller=controller,
                exec_fn=lambda force: controller.step(
                    action="EmptyLiquid",
                    forceAction=force,
                ),
            )

        return self._rl_execute(
            intent="pour",
            target_type=object_type,
            receptacle_type=receptacle_type,
            obj_meta=rec,
            controller=controller,
            exec_fn=lambda force: controller.step(
                action="FillObjectWithLiquid",
                objectId=rec["objectId"],
                fillLiquid="water",
                forceAction=force,
            ),
        )

    def wait_action(self, seconds: int = 3) -> Tuple[bool, str]:
        """Wait for a specified number of seconds (for cooking timers, etc.)."""
        seconds = max(1, int(seconds))
        logger.info(f"[IRL] wait {seconds}s")
        time.sleep(seconds)
        self.interaction_log.append({"action": "wait", "seconds": seconds, "success": True})
        return True, f"Waited {seconds}s"

    def drop_object(self, controller) -> Tuple[bool, str]:
        """Drop held object (no RL needed — deterministic)."""
        if not self.is_agent_holding_object(controller):
            return False, "[INVALID_ACTION] Not holding any object"
        held = self.get_held_object(controller)
        obj_type = held["objectType"] if held else "unknown"
        safe, safety_msg, safety_meta = check_action_safety(
            "drop",
            obj_meta=held,
            action={"intent": "drop", "object": obj_type},
            held_obj=held,
        )
        if not safe:
            self._record_safety_event("drop", obj_type, safety_msg, safety_meta, success=False)
            return False, safety_msg
        event = controller.step(action="DropHandObject")
        success = event.metadata["lastActionSuccess"]
        msg = f"Dropped {obj_type}" if success else event.metadata.get("errorMessage", "Drop failed")
        self.interaction_log.append({"action": "drop", "object": obj_type, "success": success})
        if safety_msg:
            self._record_safety_event("drop", obj_type, safety_msg, safety_meta, success=success)
            if success:
                msg = f"{safety_msg} | {msg}"
        return success, msg

    def throw_object(self, object_type: str, controller, force: float = 100.0) -> Tuple[bool, str]:
        """Throw held object (no RL needed — deterministic)."""
        if not self.is_agent_holding_object(controller):
            return False, "[INVALID_ACTION] Not holding any object"
        held = self.get_held_object(controller)
        obj_type = held["objectType"] if held else "unknown"
        safe, safety_msg, safety_meta = check_action_safety(
            "throw",
            obj_meta=held,
            action={"intent": "throw", "object": obj_type},
            held_obj=held,
        )
        if not safe:
            self._record_safety_event("throw", obj_type, safety_msg, safety_meta, success=False)
            return False, safety_msg
        event = controller.step(action="ThrowObject", moveMagnitude=force)
        success = event.metadata["lastActionSuccess"]
        msg = f"Threw {obj_type}" if success else event.metadata.get("errorMessage", "Throw failed")
        self.interaction_log.append({"action": "throw", "object": obj_type, "success": success})
        if safety_msg:
            self._record_safety_event("throw", obj_type, safety_msg, safety_meta, success=success)
            if success:
                msg = f"{safety_msg} | {msg}"
        return success, msg

    # ── Core RL execution engine ──────────────────────────────────────────────

    def _rl_execute(
        self,
        intent: str,
        target_type: Optional[str],
        receptacle_type: Optional[str],
        obj_meta: Optional[Dict],
        controller,
        exec_fn,
    ) -> Tuple[bool, str]:
        """
        Use DDQN to select a micro-strategy, execute it, observe result,
        store experience, and train periodically.

        exec_fn(force: bool) -> AI2-THOR event
        """
        last_success = False
        state = build_interaction_state(
            intent, target_type, receptacle_type,
            obj_meta, controller,
            step=0, max_steps=self.MAX_MICRO_STEPS,
            last_success=last_success,
        )

        final_success = False
        final_message = f"{intent} {target_type} failed"

        for step in range(self.MAX_MICRO_STEPS):
            # DDQN selects micro-action
            micro_action = self.ddqn.act(state)
            micro_name   = MICRO_ACTIONS[micro_action]

            # Execute repositioning moves for the chosen strategy
            self._apply_micro_repositioning(micro_name, controller)

            # Execute the actual interaction
            use_force = (micro_action == 1)  # "forced" strategy
            try:
                event = exec_fn(use_force)
                success = event.metadata["lastActionSuccess"]
                error   = event.metadata.get("errorMessage", "")
            except Exception as e:
                success = False
                error   = str(e)

            # Compute reward
            if success:
                reward = 1.0
            elif step == self.MAX_MICRO_STEPS - 1:
                reward = -0.5   # final failure
            else:
                reward = -0.1   # time penalty

            # Build next state
            try:
                # Refresh obj_meta after the step
                refreshed = self._refresh_obj_meta(target_type, controller, obj_meta)
            except Exception:
                refreshed = obj_meta

            next_state = build_interaction_state(
                intent, target_type, receptacle_type,
                refreshed, controller,
                step=step+1, max_steps=self.MAX_MICRO_STEPS,
                last_success=success,
            )

            done = success or (step == self.MAX_MICRO_STEPS - 1)

            # Store experience
            if self.online_learning:
                self.replay_buffer.append((state, micro_action, reward, next_state, done))

            logger.info(
                f"[IRL] {intent} {target_type} | step={step} | "
                f"micro={micro_name} | {'OK' if success else 'FAIL'} {error}"
            )

            # Log to interaction log
            self.interaction_log.append({
                "action": intent, "object": target_type,
                "micro_action": micro_name, "success": success,
                "step": step,
            })

            if success:
                final_success = True
                final_message = f"{intent.capitalize()} {target_type} successfully ({micro_name})"
                self._apply_action_delay()
                break

            state        = next_state
            last_success = False

        # Update counters and train
        self.interactions_n += 1
        if final_success:
            self.successes_n += 1

        if self.online_learning:
            self._maybe_train()
            self.ddqn.decay_epsilon()

        return final_success, final_message

    def _apply_micro_repositioning(self, micro_name: str, controller):
        """Apply repositioning moves before the main interaction attempt."""
        try:
            if micro_name == "move_closer":
                controller.step(action="MoveAhead")
                controller.step(action="MoveAhead")
            elif micro_name == "step_back":
                controller.step(action="MoveBack")
            elif micro_name == "look_down":
                controller.step(action="LookDown", degrees=30)
            elif micro_name == "look_up":
                controller.step(action="LookUp", degrees=30)
            elif micro_name == "rotate_left":
                controller.step(action="RotateLeft", degrees=45)
            elif micro_name == "rotate_right":
                controller.step(action="RotateRight", degrees=45)
            # "direct" and "forced" — no repositioning
        except Exception as e:
            logger.warning(f"[IRL] Repositioning failed for {micro_name}: {e}")

    def _refresh_obj_meta(self, obj_type: Optional[str], controller, fallback: Optional[Dict]) -> Optional[Dict]:
        """Get updated object metadata after an action step."""
        if obj_type is None:
            return fallback
        obj = self.find_object_by_type(obj_type, controller, allow_invisible=True)
        return obj if obj else fallback

    def _maybe_train(self):
        """Train DDQN if buffer is large enough and replay interval reached."""
        if len(self.replay_buffer) < self.MIN_BUFFER:
            return
        if self.interactions_n % self.REPLAY_AFTER_N != 0:
            return

        batch = random.sample(list(self.replay_buffer), min(self.BATCH_SIZE, len(self.replay_buffer)))
        self.ddqn.replay(batch)

        if self.interactions_n % self.TARGET_UPDATE_N == 0:
            self.ddqn.update_target()

        if self.interactions_n % 100 == 0:
            self._save()
            sr = self.successes_n / max(self.interactions_n, 1) * 100
            logger.info(
                f"[IRL] Progress: {self.interactions_n} interactions | "
                f"success rate {sr:.1f}% | eps={self.ddqn.epsilon:.3f}"
            )

    def _apply_action_delay(self):
        if self.action_delay_sec > 0:
            time.sleep(self.action_delay_sec)

    def _sorted_interaction_candidates(self, objects: List[Dict], controller) -> List[Dict]:
        """Prefer visible, nearby objects to reduce duplicate-type mis-targeting."""
        def _sort_key(obj: Dict) -> Tuple[int, float, str]:
            try:
                dist = self.get_object_distance(obj, controller)
            except Exception:
                dist = float("inf")
            return (
                0 if obj.get("visible", False) else 1,
                dist,
                obj.get("objectId", ""),
            )

        return sorted(objects, key=_sort_key)

    def _collect_action_candidates(
        self,
        object_type: str,
        controller,
        allow_invisible: bool = True,
        preferred_obj: Optional[Dict] = None,
    ) -> List[Dict]:
        """Return de-duplicated candidates, preserving a preferred object first."""
        candidates: List[Dict] = []
        seen: set[str] = set()

        def _push(obj: Optional[Dict]) -> None:
            if not obj:
                return
            obj_id = obj.get("objectId")
            if not obj_id or obj_id in seen:
                return
            seen.add(obj_id)
            live = self.find_object_by_id(obj_id, controller) or obj
            candidates.append(live)

        _push(preferred_obj)
        for obj in self.find_all_objects_by_type(object_type, controller, allow_invisible=allow_invisible):
            _push(obj)
        return candidates

    def _try_direct_pick_fallback(
        self,
        object_type: str,
        controller,
        preferred_obj: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """Try a deterministic pickup on the best nearby matching candidates."""
        for obj in self._collect_action_candidates(
            object_type,
            controller,
            allow_invisible=True,
            preferred_obj=preferred_obj,
        ):
            if not obj.get("pickupable", False):
                continue
            try:
                dist = self.get_object_distance(obj, controller)
            except Exception:
                dist = float("inf")
            if dist > self.max_distance + 0.35:
                continue

            attempts = [(False, "direct"), (True, "forced")]
            if not obj.get("visible", False):
                attempts = [(True, "forced")]

            for force, label in attempts:
                event = controller.step(
                    action="PickupObject",
                    objectId=obj["objectId"],
                    forceAction=force,
                    manualInteract=False,
                )
                if event.metadata.get("lastActionSuccess", False):
                    self.interaction_log.append({
                        "action": "pick",
                        "object": object_type,
                        "micro_action": f"{label}_fallback",
                        "success": True,
                    })
                    self._apply_action_delay()
                    return True, (
                        f"Picked {object_type} via {label} fallback "
                        f"({obj.get('objectType', object_type)} @ {dist:.2f}m)"
                    )

        return False, f"pick {object_type} failed"

    def _try_direct_toggle_fallback(
        self,
        object_type: str,
        action_name: str,
        controller,
        preferred_obj: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """Try a deterministic toggle on the resolved object before giving up."""
        for obj in self._collect_action_candidates(
            object_type,
            controller,
            allow_invisible=True,
            preferred_obj=preferred_obj,
        ):
            if not obj.get("toggleable", False):
                continue
            try:
                dist = self.get_object_distance(obj, controller)
            except Exception:
                dist = float("inf")
            if dist > self.max_distance + 0.5:
                continue

            attempts = [(False, "direct"), (True, "forced")]
            if not obj.get("visible", False):
                attempts = [(True, "forced")]

            for force, label in attempts:
                event = controller.step(
                    action=action_name,
                    objectId=obj["objectId"],
                    forceAction=force,
                )
                if event.metadata.get("lastActionSuccess", False):
                    self.interaction_log.append({
                        "action": "toggle",
                        "object": object_type,
                        "micro_action": f"{label}_fallback",
                        "success": True,
                    })
                    self._apply_action_delay()
                    return True, (
                        f"Toggled {object_type} via {label} fallback "
                        f"({obj.get('objectId', object_type)})"
                    )

        return False, f"toggle {object_type} failed"

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            self.ddqn.save(self.WEIGHTS_PATH)
            data = list(self.replay_buffer)[-5000:]
            with open(self.BUFFER_PATH, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"[IRL] Saved weights + {len(data)} experiences")
        except Exception as e:
            logger.warning(f"[IRL] Save failed: {e}")

    def _load(self):
        # Weights
        if os.path.exists(self.WEIGHTS_PATH + ".index"):
            try:
                self.ddqn.load(self.WEIGHTS_PATH)
            except Exception as e:
                logger.warning(f"[IRL] Could not load weights: {e}")
        # Replay buffer
        if os.path.exists(self.BUFFER_PATH):
            try:
                with open(self.BUFFER_PATH, "rb") as f:
                    data = pickle.load(f)
                self.replay_buffer.extend(data)
                logger.info(f"[IRL] Loaded {len(data)} replay experiences")
                # Warm epsilon based on buffer size (skipped in eval_mode for determinism)
                if self.eval_mode:
                    self.ddqn.epsilon = 0.0
                    logger.info("[IRL] eval_mode=True — epsilon forced to 0.0 (greedy)")
                elif len(self.replay_buffer) > 500:
                    self.ddqn.epsilon = max(self.ddqn.epsilon_min, 0.3)
                elif len(self.replay_buffer) > 100:
                    self.ddqn.epsilon = 0.6
            except Exception as e:
                logger.warning(f"[IRL] Could not load replay buffer: {e}")

    # ── Utility methods (same as InteractionController) ───────────────────────

    def find_object_by_type(self, object_type: str, controller, allow_invisible: bool = False) -> Optional[Dict]:
        """Find the best matching object, preferring visible and nearby candidates."""
        matches = self.find_all_objects_by_type(
            object_type,
            controller,
            allow_invisible=allow_invisible,
        )
        return matches[0] if matches else None

    def find_all_objects_by_type(self, object_type: str, controller, allow_invisible: bool = False) -> List[Dict]:
        """Find all objects matching type, ranked for interaction quality."""
        objects = controller.last_event.metadata.get("objects", [])
        search_type = object_type.lower().replace(" ", "")
        variants = {"egg": ["egg", "eggcracked"]}.get(search_type, [search_type])
        known_suffixes = ("sliced", "cracked", "dirty", "empty", "filled", "cooked")

        exact_matches: List[Dict] = []
        substring_matches: List[Dict] = []
        for obj in objects:
            if not allow_invisible and not obj.get("visible", False):
                continue
            obj_type = obj.get("objectType", "").lower().replace(" ", "")
            if obj_type in variants:
                exact_matches.append(obj)
                continue
            if any(obj_type == search_type + suffix for suffix in known_suffixes):
                exact_matches.append(obj)
                continue
            if obj_type in search_type and obj_type != search_type:
                # Substring fallback (e.g. "Sink" matching a "SinkBasin" query).
                # Only used when no exact-type match exists in the scene, to
                # avoid returning a non-receptacle "Sink" when "SinkBasin" is
                # what the caller actually wants.
                substring_matches.append(obj)

        matches = exact_matches if exact_matches else substring_matches

        if not allow_invisible:
            matches = [obj for obj in matches if obj.get("visible", False)]

        return self._sorted_interaction_candidates(matches, controller)

    def find_object_by_id(self, object_id: str, controller) -> Optional[Dict]:
        for obj in controller.last_event.metadata.get("objects", []):
            if obj.get("objectId") == object_id:
                return obj
        return None

    def get_object_distance(self, obj: Dict, controller) -> float:
        agent_pos = controller.last_event.metadata["agent"]["position"]
        obj_pos   = obj["position"]
        return float(np.sqrt(
            (agent_pos["x"] - obj_pos["x"])**2 +
            (agent_pos["y"] - obj_pos["y"])**2 +
            (agent_pos["z"] - obj_pos["z"])**2
        ))

    def get_object_affordances(self, obj: Dict) -> Dict[str, bool]:
        return {
            "pickupable": obj.get("pickupable", False),
            "openable":   obj.get("openable",   False),
            "receptacle": obj.get("receptacle", False),
            "toggleable": obj.get("toggleable", False),
            "sliceable":  obj.get("sliceable",  False),
            "cookable":   obj.get("cookable",   False),
            "breakable":  obj.get("breakable",  False),
            "fillable":   obj.get("canFillWithLiquid", False),
        }

    def is_agent_holding_object(self, controller) -> bool:
        return len(controller.last_event.metadata.get("inventoryObjects", [])) > 0

    def get_held_object(self, controller) -> Optional[Dict]:
        inv = controller.last_event.metadata.get("inventoryObjects", [])
        if not inv:
            return None
        held_id = inv[0]["objectId"]
        for obj in controller.last_event.metadata.get("objects", []):
            if obj["objectId"] == held_id:
                return obj
        return None

    def is_object_visible(self, object_type: str, controller) -> bool:
        return self.find_object_by_type(object_type, controller) is not None

    def get_visible_objects(self, controller) -> List[Dict]:
        objects = controller.last_event.metadata.get("objects", [])
        return [
            {
                "objectId": o["objectId"],
                "objectType": o["objectType"],
                "position": o["position"],
                "affordances": self.get_object_affordances(o),
            }
            for o in objects if o.get("visible", False)
        ]

    # ── Container accessibility (same API as InteractionController) ───────────

    def get_parent_receptacles(self, obj: Dict, controller) -> List[Dict]:
        parent_ids = obj.get("parentReceptacles") or []
        return [
            self.find_object_by_id(pid, controller)
            for pid in parent_ids
            if self.find_object_by_id(pid, controller) is not None
        ]

    def get_closed_parent_containers(self, obj: Dict, controller) -> List[Dict]:
        parents = self.get_parent_receptacles(obj, controller)
        closed  = [p for p in parents if p.get("openable", False) and not p.get("isOpen", True)]
        return list(reversed(closed))

    def is_object_accessible(self, object_type: str, controller) -> Tuple[bool, str, List[Dict]]:
        obj = self.find_object_by_type(object_type, controller, allow_invisible=True)
        if obj is None:
            return False, f"{object_type} not found", []
        closed = self.get_closed_parent_containers(obj, controller)
        if closed:
            names = [c.get("objectType","?") for c in closed]
            return False, f"{object_type} in closed containers: {names}", closed
        return True, f"{object_type} accessible", []

    def ensure_object_accessible(self, object_type: str, controller) -> Tuple[bool, str, List[str]]:
        is_acc, msg, closed = self.is_object_accessible(object_type, controller)
        if is_acc:
            return True, msg, []
        opened = []
        for container in closed:
            ctype = container.get("objectType", "?")
            # Use horizontal-only (x, z) distance — wall-mounted cabinets are
            # above head height so 3-D Euclidean distance is always > 1.5 m
            # even when the agent is standing directly below/in front of them.
            agent_pos = controller.last_event.metadata["agent"]["position"]
            c_pos     = container.get("position", {})
            xz_dist   = float(np.sqrt(
                (agent_pos["x"] - c_pos.get("x", 0))**2 +
                (agent_pos["z"] - c_pos.get("z", 0))**2
            ))
            if xz_dist > self.max_distance:
                return False, f"{ctype} too far ({xz_dist:.2f}m xz)", opened
            event = controller.step(
                action="OpenObject",
                objectId=container["objectId"],
                forceAction=True,   # force-open to bypass small visibility issues
            )
            if event.metadata["lastActionSuccess"]:
                opened.append(ctype)
            else:
                err = event.metadata.get("errorMessage", "?")
                return False, f"Failed to open {ctype}: {err}", opened
        return True, f"Opened {len(opened)} container(s)", opened

    def get_object_containment_info(self, object_type: str, controller) -> Dict:
        obj = self.find_object_by_type(object_type, controller, allow_invisible=True)
        if obj is None:
            return {"found": False, "object_type": object_type, "message": "not found"}
        parents = self.get_parent_receptacles(obj, controller)
        closed  = self.get_closed_parent_containers(obj, controller)
        return {
            "found":             True,
            "object_type":       object_type,
            "object_id":         obj.get("objectId"),
            "visible":           obj.get("visible", False),
            "pickupable":        obj.get("pickupable", False),
            "parent_receptacles":[p.get("objectType") for p in parents],
            "closed_containers": [c.get("objectType") for c in closed],
            "is_accessible":     len(closed) == 0,
            "position":          obj.get("position"),
        }

    def get_object_metadata(self, object_type: str, controller, allow_invisible: bool = True) -> Optional[Dict]:
        return self.find_object_by_type(object_type, controller, allow_invisible=allow_invisible)

    def _record_safety_event(
        self,
        intent: str,
        object_type: str,
        message: str,
        metadata: Optional[Dict] = None,
        success: bool = False,
    ) -> None:
        event = {
            "action": intent,
            "object": object_type,
            "success": success,
            "safety_message": message,
            "safety_metadata": metadata or {},
        }
        self.safety_events.append(event)
        self.interaction_log.append(event)

    def get_interaction_log(self) -> List[Dict]:
        return self.interaction_log.copy()

    def clear_interaction_log(self):
        self.interaction_log = []
        self.safety_events = []

    def get_safety_events(self) -> List[Dict]:
        return self.safety_events.copy()

    def get_rl_stats(self) -> Dict:
        """Return current RL training statistics."""
        return {
            "interactions":   self.interactions_n,
            "successes":      self.successes_n,
            "success_rate":   self.successes_n / max(self.interactions_n, 1),
            "epsilon":        self.ddqn.epsilon,
            "buffer_size":    len(self.replay_buffer),
        }
