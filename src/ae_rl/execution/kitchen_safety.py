"""
Kitchen relational safety graph utilities.

This module builds a scene-level stove-top safety snapshot from AI2-THOR
metadata so interaction policies can reason about:
  - which burner a stove knob controls
  - what objects are placed on each burner
  - whether nearby hazards make burner activation unsafe
  - whether cookware is on an active, correct burner before cooking
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ae_rl.execution.safety import classify_semantic_category, normalize_object_name


SAFE_STOVETOP_COOKWARE = {"pan", "pot"}
ALLOWED_BURNER_CONTENT_CATEGORIES = {"food", "cookware"}
NEAR_HEAT_HAZARD_CATEGORIES = {
    "document_card",
    "valuable_small_item",
    "electronic",
    "hazardous_electronic",
    "flammable",
    "chemical_liquid",
    "flammable_liquid",
}
NEAR_HEAT_HAZARD_TYPES = {
    "book",
    "newspaper",
    "cloth",
    "tissuebox",
    "toiletpaper",
    "papertowel",
    "cellphone",
    "laptop",
    "creditcard",
    "debitcard",
    "wallet",
    "keychain",
}


def _position(obj: Dict[str, Any]) -> Dict[str, float]:
    return obj.get("position") or {"x": 0.0, "y": 0.0, "z": 0.0}


def _distance_2d(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    pa = _position(a)
    pb = _position(b)
    return math.sqrt((pa["x"] - pb["x"]) ** 2 + (pa["z"] - pb["z"]) ** 2)


def _layout_sort(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(objects, key=lambda obj: (-_position(obj)["z"], _position(obj)["x"]))


def _extract_parent_ids(obj: Dict[str, Any]) -> List[str]:
    return [str(parent) for parent in (obj.get("parentReceptacles") or []) if parent]


def _is_safe_cookware(obj: Dict[str, Any]) -> bool:
    obj_type = normalize_object_name(obj.get("objectType"))
    return obj_type in SAFE_STOVETOP_COOKWARE or classify_semantic_category(obj_type) == "cookware"


def _is_allowed_burner_object(obj: Dict[str, Any]) -> bool:
    obj_type = normalize_object_name(obj.get("objectType"))
    category = classify_semantic_category(obj_type)
    if obj_type in SAFE_STOVETOP_COOKWARE:
        return True
    return category in ALLOWED_BURNER_CONTENT_CATEGORIES


def _describe_objects(objects: List[Dict[str, Any]]) -> str:
    names = [obj.get("objectType", "Object") for obj in objects]
    return ", ".join(sorted(names)) if names else "nothing"


def infer_stove_knob_burner_map(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    burners = [obj for obj in objects if obj.get("objectType") == "StoveBurner"]
    knobs = [obj for obj in objects if obj.get("objectType") == "StoveKnob"]
    knob_to_burner: Dict[str, str] = {}
    burner_to_knob: Dict[str, str] = {}

    burners_sorted = _layout_sort(burners)
    knobs_sorted = sorted(knobs, key=lambda obj: (_position(obj)["x"], _position(obj)["z"]))
    confidence = "low"

    if len(burners_sorted) == 4 and len(knobs_sorted) == 4:
        # Typical AI2-THOR 4-burner stove layout:
        # burners_sorted = [front-left, front-right, back-left, back-right]
        burner_order = [
            burners_sorted[0],  # leftmost knob -> front-left
            burners_sorted[2],  # second knob   -> back-left
            burners_sorted[3],  # third knob    -> back-right
            burners_sorted[1],  # rightmost knob-> front-right
        ]
        confidence = "high"
    else:
        burner_order = sorted(burners, key=lambda obj: (_position(obj)["x"], -_position(obj)["z"]))
        confidence = "medium" if burners_sorted and knobs_sorted and len(burners_sorted) == len(knobs_sorted) else "low"

    if len(knobs_sorted) == len(burner_order):
        for knob, burner in zip(knobs_sorted, burner_order):
            knob_id = knob.get("objectId", "")
            burner_id = burner.get("objectId", "")
            if knob_id and burner_id:
                knob_to_burner[knob_id] = burner_id
                burner_to_knob[burner_id] = knob_id

    return {
        "knob_to_burner": knob_to_burner,
        "burner_to_knob": burner_to_knob,
        "sorted_knob_ids": [obj.get("objectId", "") for obj in knobs_sorted if obj.get("objectId")],
        "sorted_burner_ids": [obj.get("objectId", "") for obj in burners_sorted if obj.get("objectId")],
        "confidence": confidence,
    }


def build_kitchen_snapshot(controller) -> Dict[str, Any]:
    objects = list(controller.last_event.metadata.get("objects", []))
    burners = [obj for obj in objects if obj.get("objectType") == "StoveBurner"]
    knobs = [obj for obj in objects if obj.get("objectType") == "StoveKnob"]
    mapping = infer_stove_knob_burner_map(objects)

    burner_objects: Dict[str, List[Dict[str, Any]]] = {
        burner.get("objectId", ""): [] for burner in burners if burner.get("objectId")
    }
    for obj in objects:
        obj_id = obj.get("objectId", "")
        obj_type = obj.get("objectType", "")
        if obj_type in {"StoveBurner", "StoveKnob"} or not obj_id:
            continue
        for parent_id in _extract_parent_ids(obj):
            if parent_id in burner_objects:
                burner_objects[parent_id].append(obj)

    snapshot = {
        "objects": objects,
        "burners": burners,
        "knobs": knobs,
        "burners_by_id": {
            burner.get("objectId", ""): burner for burner in burners if burner.get("objectId")
        },
        "knobs_by_id": {
            knob.get("objectId", ""): knob for knob in knobs if knob.get("objectId")
        },
        "knob_to_burner": mapping["knob_to_burner"],
        "burner_to_knob": mapping["burner_to_knob"],
        "sorted_knob_ids": mapping["sorted_knob_ids"],
        "sorted_burner_ids": mapping["sorted_burner_ids"],
        "mapping_confidence": mapping["confidence"],
        "burner_objects": burner_objects,
    }

    active_burners: List[str] = []
    for knob_id, burner_id in snapshot["knob_to_burner"].items():
        knob = snapshot["knobs_by_id"].get(knob_id)
        if knob and knob.get("isToggled", False):
            active_burners.append(burner_id)
    snapshot["active_burners"] = active_burners

    hazards_by_burner = {
        burner_id: get_nearby_hazards(snapshot, burner_id)
        for burner_id in burner_objects.keys()
    }
    snapshot["hazards_by_burner"] = hazards_by_burner
    return snapshot


def get_objects_on_burner(snapshot: Dict[str, Any], burner_id: str) -> List[Dict[str, Any]]:
    return list(snapshot.get("burner_objects", {}).get(burner_id, []))


def get_nearby_hazards(
    snapshot: Dict[str, Any],
    burner_id: str,
    radius: float = 0.35,
) -> List[Dict[str, Any]]:
    burner = snapshot.get("burners_by_id", {}).get(burner_id)
    if burner is None:
        return []

    hazards: List[Dict[str, Any]] = []
    burner_stack_ids = {obj.get("objectId", "") for obj in get_objects_on_burner(snapshot, burner_id)}
    burner_stack_ids.add(burner_id)
    knob_id = snapshot.get("burner_to_knob", {}).get(burner_id)
    if knob_id:
        burner_stack_ids.add(knob_id)

    for obj in snapshot.get("objects", []):
        obj_id = obj.get("objectId", "")
        if not obj_id or obj_id in burner_stack_ids:
            continue
        if obj.get("objectType") in {"StoveBurner", "StoveKnob"}:
            continue
        if _distance_2d(obj, burner) > radius:
            continue

        obj_type = normalize_object_name(obj.get("objectType"))
        category = classify_semantic_category(obj_type)
        if category in NEAR_HEAT_HAZARD_CATEGORIES or obj_type in NEAR_HEAT_HAZARD_TYPES:
            hazards.append({
                "objectId": obj_id,
                "objectType": obj.get("objectType", "Object"),
                "category": category,
                "distance": round(_distance_2d(obj, burner), 3),
            })
    return hazards


def find_corresponding_knob(
    snapshot: Dict[str, Any],
    burner_id: str,
) -> Optional[Dict[str, Any]]:
    knob_id = snapshot.get("burner_to_knob", {}).get(burner_id)
    if knob_id:
        return snapshot.get("knobs_by_id", {}).get(knob_id)
    return None


def resolve_toggle_target(
    action: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    knob_id = action.get("knob_id")
    knob_index = action.get("knob_index")
    knob_obj = None
    meta: Dict[str, Any] = {"auto_resolved": False}

    if knob_id:
        for candidate_id, knob in snapshot.get("knobs_by_id", {}).items():
            if knob_id.lower() in candidate_id.lower():
                knob_obj = knob
                break
    elif knob_index is not None:
        knob_ids = snapshot.get("sorted_knob_ids", [])
        if 0 <= int(knob_index) < len(knob_ids):
            knob_obj = snapshot.get("knobs_by_id", {}).get(knob_ids[int(knob_index)])

    occupied_burners = [
        burner_id
        for burner_id in snapshot.get("sorted_burner_ids", [])
        if any(_is_safe_cookware(obj) for obj in get_objects_on_burner(snapshot, burner_id))
    ]

    if knob_obj is None:
        active_knobs = [
            snapshot["knobs_by_id"][kid]
            for kid in snapshot.get("sorted_knob_ids", [])
            if snapshot["knobs_by_id"].get(kid, {}).get("isToggled", False)
        ]
        if len(active_knobs) == 1:
            knob_obj = active_knobs[0]
            meta["auto_resolved"] = True
        elif len(occupied_burners) == 1:
            knob_obj = find_corresponding_knob(snapshot, occupied_burners[0])
            meta["auto_resolved"] = knob_obj is not None

    if knob_obj is None:
        return None, None, meta

    burner_id = snapshot.get("knob_to_burner", {}).get(knob_obj.get("objectId", ""))
    burner_obj = snapshot.get("burners_by_id", {}).get(burner_id) if burner_id else None
    if knob_obj:
        meta["selected_knob_id"] = knob_obj.get("objectId")
        if knob_obj.get("objectId") in snapshot.get("sorted_knob_ids", []):
            meta["selected_knob_index"] = snapshot["sorted_knob_ids"].index(knob_obj["objectId"])
    if burner_obj:
        meta["selected_burner_id"] = burner_obj.get("objectId")
    return knob_obj, burner_obj, meta


def select_preferred_stove_burner(snapshot: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    candidates = []
    for burner_id in snapshot.get("sorted_burner_ids", []):
        burner = snapshot.get("burners_by_id", {}).get(burner_id)
        if burner is None:
            continue
        burner_objs = get_objects_on_burner(snapshot, burner_id)
        hazards = snapshot.get("hazards_by_burner", {}).get(burner_id, [])
        knob = find_corresponding_knob(snapshot, burner_id)
        if burner_objs:
            continue
        if hazards:
            continue
        if knob and knob.get("isToggled", False):
            continue
        candidates.append(burner)

    if not candidates:
        return None, {"reason": "no_safe_empty_burner"}

    burner = candidates[0]
    knob = find_corresponding_knob(snapshot, burner.get("objectId", ""))
    meta = {
        "selected_burner_id": burner.get("objectId"),
        "selected_knob_id": knob.get("objectId") if knob else "",
        "selected_burner_index": snapshot.get("sorted_burner_ids", []).index(burner.get("objectId"))
        if burner.get("objectId") in snapshot.get("sorted_burner_ids", [])
        else -1,
    }
    if knob and knob.get("objectId") in snapshot.get("sorted_knob_ids", []):
        meta["selected_knob_index"] = snapshot["sorted_knob_ids"].index(knob["objectId"])
    return burner, meta


def check_toggle_scene_safety(
    action: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    knob_obj, burner_obj, meta = resolve_toggle_target(action, snapshot)
    if knob_obj is None or burner_obj is None:
        return False, "[SAFETY_BLOCK] Stove knob target is ambiguous; specify or infer a burner safely", {
            **meta,
            "rule": "ambiguous_stove_knob",
            "mapping_confidence": snapshot.get("mapping_confidence", "low"),
        }

    burner_id = burner_obj.get("objectId", "")
    burner_objs = get_objects_on_burner(snapshot, burner_id)
    hazards = snapshot.get("hazards_by_burner", {}).get(burner_id, [])
    safe_cookware = [obj for obj in burner_objs if _is_safe_cookware(obj)]
    active = bool(knob_obj.get("isToggled", False))

    enriched = {
        **meta,
        "rule": "",
        "burner_objects": [obj.get("objectType", "Object") for obj in burner_objs],
        "hazards": hazards,
    }

    if active:
        return True, "", enriched

    occupied_burners = [
        b_id
        for b_id in snapshot.get("sorted_burner_ids", [])
        if any(_is_safe_cookware(obj) for obj in get_objects_on_burner(snapshot, b_id))
    ]

    if not burner_objs:
        enriched["rule"] = "empty_burner_toggle"
        # Find any occupied burner whose knob differs from the one selected
        redirect_candidates = [
            b_id for b_id in occupied_burners if b_id != burner_id
        ]
        if redirect_candidates:
            correct_knob = find_corresponding_knob(snapshot, redirect_candidates[0])
            if correct_knob and correct_knob.get("objectId") in snapshot.get("sorted_knob_ids", []):
                knob_index = snapshot["sorted_knob_ids"].index(correct_knob["objectId"])
                enriched["recommended_knob_index"] = knob_index
                enriched["recommended_burner_id"] = redirect_candidates[0]
                return False, (
                    f"[SAFETY_BLOCK] Wrong stove knob for occupied burner; use knob_index={knob_index}"
                ), enriched
        return False, "[SAFETY_BLOCK] Cannot turn on an empty stove burner safely", enriched

    if not safe_cookware:
        enriched["rule"] = "unsafe_burner_occupancy"
        return False, (
            f"[SAFETY_BLOCK] Burner contains unsafe object(s): {_describe_objects(burner_objs)}"
        ), enriched

    disallowed = [obj for obj in burner_objs if not _is_allowed_burner_object(obj)]
    if disallowed:
        enriched["rule"] = "unsafe_burner_occupancy"
        return False, (
            f"[SAFETY_BLOCK] Burner contains non-cookware object(s): {_describe_objects(disallowed)}"
        ), enriched

    if hazards:
        enriched["rule"] = "near_heat_hazard"
        return False, (
            f"[SAFETY_BLOCK] Hazard near active burner: {_describe_objects(hazards)}"
        ), enriched

    if meta.get("auto_resolved"):
        return True, (
            f"[SAFETY_WARN] Auto-selected knob_index={meta.get('selected_knob_index', -1)} "
            "for the occupied burner"
        ), enriched
    return True, "", enriched


def check_place_scene_safety(
    action: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    receptacle = normalize_object_name(action.get("receptacle"))
    if receptacle != "stoveburner":
        return True, "", {"rule": ""}

    obj_name = action.get("object", "")
    obj_norm = normalize_object_name(obj_name)
    obj_cat = classify_semantic_category(obj_norm)

    if obj_norm not in SAFE_STOVETOP_COOKWARE and obj_cat != "cookware":
        return False, (
            f"[SAFETY_BLOCK] {obj_name or 'Object'} is not safe cookware for a stove burner"
        ), {"rule": "unsafe_burner_placement", "object_category": obj_cat}

    burner, meta = select_preferred_stove_burner(snapshot)
    if burner is None:
        return False, "[SAFETY_BLOCK] No safe empty stove burner is available", {
            **meta,
            "rule": "no_safe_burner_available",
        }

    return True, (
        f"[SAFETY_WARN] Selected burner_index={meta.get('selected_burner_index', -1)} "
        "as the safe stove placement target"
    ), {
        **meta,
        "rule": "safe_burner_selected",
    }


def check_cook_scene_safety(
    action: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    object_name = normalize_object_name(action.get("object"))
    device_name = normalize_object_name(action.get("device"))
    if not object_name:
        return True, "", {"rule": ""}

    candidates = [
        obj for obj in snapshot.get("objects", [])
        if object_name in normalize_object_name(obj.get("objectType")) or
        normalize_object_name(obj.get("objectType")) in object_name
    ]
    if not candidates:
        return True, "", {"rule": "object_not_found_for_scene_check"}

    obj = next((
        candidate for candidate in candidates
        if any(parent in snapshot.get("burners_by_id", {}) for parent in _extract_parent_ids(candidate))
    ), None)
    if obj is None:
        obj = next((
            candidate for candidate in candidates
            if any(
                normalize_object_name(parent.split("|")[0]) in {"pan", "pot", "microwave"}
                for parent in _extract_parent_ids(candidate)
            )
        ), None)
    if obj is None:
        obj = next((candidate for candidate in candidates if candidate.get("cookable", False)), candidates[0])
    parents = _extract_parent_ids(obj)

    # Fallback: if the cooked object has no parents (e.g. EggCracked floating inside
    # a Pan after a break action), try to resolve via the action's "container" field.
    # Find the container object and use its parents instead.
    if not parents:
        container_name = normalize_object_name((action.get("container") or ""))
        if container_name:
            container_obj = next((
                o for o in snapshot.get("objects", [])
                if normalize_object_name(o.get("objectType", "")) == container_name
            ), None)
            if container_obj:
                parents = _extract_parent_ids(container_obj)

    burner_parent = next((parent for parent in parents if parent in snapshot.get("burners_by_id", {})), None)
    if burner_parent:
        burner_objs = get_objects_on_burner(snapshot, burner_parent)
        if not any(_is_safe_cookware(parent_obj) for parent_obj in burner_objs):
            return False, (
                f"[SAFETY_BLOCK] {obj.get('objectType', 'Object')} is on a burner without cookware support"
            ), {"rule": "cook_without_cookware", "burner_id": burner_parent}

        hazards = snapshot.get("hazards_by_burner", {}).get(burner_parent, [])
        if hazards:
            return False, (
                f"[SAFETY_BLOCK] Hazard near burner during cook: {_describe_objects(hazards)}"
            ), {"rule": "near_heat_hazard", "burner_id": burner_parent, "hazards": hazards}

        knob = find_corresponding_knob(snapshot, burner_parent)
        if knob is None:
            return False, "[SAFETY_BLOCK] Could not determine the stove knob for the active burner", {
                "rule": "unknown_burner_knob_mapping",
                "burner_id": burner_parent,
            }
        if not knob.get("isToggled", False):
            knob_index = snapshot.get("sorted_knob_ids", []).index(knob["objectId"]) \
                if knob.get("objectId") in snapshot.get("sorted_knob_ids", []) else -1
            return False, (
                f"[SAFETY_BLOCK] Burner under {obj.get('objectType', 'Object')} is not active; "
                f"toggle knob_index={knob_index} first"
            ), {
                "rule": "inactive_required_burner",
                "burner_id": burner_parent,
                "required_knob_index": knob_index,
            }
        return True, "", {"rule": "safe_active_burner", "burner_id": burner_parent}

    parent_types = {normalize_object_name(parent.split("|")[0]) for parent in parents}
    if any(h in parent_types for h in ("microwave", "pan", "pot", "toaster")):
        return True, "", {"rule": "supported_heat_container"}

    # Some AI2-THOR states lag parent linkage updates after a successful place
    # into countertop appliances like the Toaster or Microwave. When the planner
    # explicitly requests one of those appliances as the cook device, avoid
    # false safety blocks and let execution validate the actual cook action.
    if device_name in {"microwave", "toaster"}:
        has_device = any(
            normalize_object_name(obj.get("objectType", "")) == device_name
            for obj in snapshot.get("objects", [])
        )
        if has_device:
            return True, "", {
                "rule": "supported_heat_device_hint",
                "device": device_name,
            }

    return False, (
        f"[SAFETY_BLOCK] {obj.get('objectType', 'Object')} is not on a valid heat source for cooking"
    ), {"rule": "missing_heat_source"}
