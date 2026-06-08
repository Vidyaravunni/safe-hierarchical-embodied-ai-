"""
Shared safety helpers for semantic, thermal, and material-aware execution.

This module centralizes the hybrid safety layer used by:
  - nlp_parser_api.py      (command / plan validation)
  - llm_parser.py          (parse metadata handling)
  - interaction_rl_controller.py (execution-time checks)
  - task_executor_v3.py    (recovery + safety-aware queue edits)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


HEATING_APPLIANCES = {
    "microwave",
    "toaster",
    "oven",
    "stoveburner",
    "stoveknob",
    "coffeemachine",
    "airfryer",
}

ELECTRICAL_APPLIANCES = {
    "microwave",
    "toaster",
    "coffeemachine",
    "television",
    "lightswitch",
    "lamp",
    "floorlamp",
    "remotecontrol",
}

SAFE_RECEPTACLES = {
    "countertop",
    "diningtable",
    "sidetable",
    "plate",
    "bowl",
}

COMMAND_ALIASES: Dict[str, str] = {
    "debit card": "debitcard",
    "credit card": "creditcard",
    "id card": "idcard",
    "bank card": "creditcard",
    "cell phone": "cellphone",
    "mobile phone": "cellphone",
    "smart phone": "cellphone",
    "remote control": "remotecontrol",
    "key chain": "keychain",
    "paper bag": "paperbag",
    "plastic bag": "plasticbag",
    "tissue box": "tissuebox",
    "toilet paper": "toiletpaper",
    "paper towel": "papertowel",
    "soap bottle": "soapbottle",
    "spray bottle": "spraybottle",
    "wine bottle": "winebottle",
    "glass bottle": "glassbottle",
    "ceramic plate": "ceramicplate",
    "metal pan": "metalpan",
    "coffee machine": "coffeemachine",
    "stove burner": "stoveburner",
    "stove knob": "stoveknob",
    "air fryer": "airfryer",
}

SEMANTIC_CATEGORIES: Dict[str, str] = {
    "apple": "food",
    "bread": "food",
    "breadsliced": "food",
    "egg": "food",
    "eggcracked": "food",
    "lettuce": "food",
    "lettucesliced": "food",
    "potato": "food",
    "potatosliced": "food",
    "tomato": "food",
    "tomatosliced": "food",
    "mug": "vessel",
    "cup": "vessel",
    "bowl": "vessel",
    "plate": "vessel",
    "pan": "cookware",
    "pot": "cookware",
    "winebottle": "vessel",
    "glassbottle": "fragile_container",
    "bottle": "vessel",
    "ceramicplate": "fragile_item",
    "metalpan": "conductive_cookware",
    "knife": "sharp",
    "butterknife": "sharp",
    "butcherknife": "sharp",
    "fork": "sharp",
    "scissors": "sharp",
    "keychain": "valuable_small_item",
    "wallet": "valuable_small_item",
    "keys": "valuable_small_item",
    "charger": "electronic",
    "battery": "hazardous_electronic",
    "laptop": "electronic",
    "debitcard": "document_card",
    "creditcard": "document_card",
    "idcard": "document_card",
    "card": "document_card",
    "paper": "flammable",
    "paperbag": "flammable",
    "plasticbag": "flammable",
    "bag": "flammable",
    "book": "flammable",
    "newspaper": "flammable",
    "cloth": "flammable",
    "tissue": "flammable",
    "tissuebox": "flammable",
    "toiletpaper": "flammable",
    "papertowel": "flammable",
    "cellphone": "electronic",
    "phone": "electronic",
    "remotecontrol": "electronic",
    "television": "electronic",
    "lamp": "electronic",
    "lightswitch": "electronic",
    "water": "liquid",
    "coffee": "liquid",
    "milk": "liquid",
    "soap": "liquid",
    "bleach": "chemical_liquid",
    "detergent": "chemical_liquid",
    "oil": "flammable_liquid",
    "soapbottle": "liquid_container",
    "spraybottle": "liquid_container",
    "microwave": "heating_appliance",
    "toaster": "heating_appliance",
    "oven": "heating_appliance",
    "stoveburner": "heating_appliance",
    "stoveknob": "heating_appliance",
    "coffeemachine": "electrical_appliance",
    "faucet": "water_source",
}

AI2THOR_KNOWN_OBJECT_TYPES: Set[str] = {
    "CreditCard",
    "CellPhone",
    "RemoteControl",
    "Laptop",
    "KeyChain",
    "Book",
    "Newspaper",
    "TissueBox",
    "ToiletPaper",
    "SoapBottle",
    "SprayBottle",
    "WineBottle",
    "Bottle",
    "Plate",
    "Pan",
    "Pot",
    "Mug",
    "Cup",
    "Bowl",
}

MATERIAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "glass": {
        "fragile": True,
        "heat_conductivity": "low",
        "handling_mode": "slow",
    },
    "ceramic": {
        "fragile": True,
        "heat_conductivity": "medium",
        "handling_mode": "slow",
    },
    "metal": {
        "fragile": False,
        "heat_conductivity": "high",
        "handling_mode": "cautious",
    },
    "aluminum": {
        "fragile": False,
        "heat_conductivity": "high",
        "handling_mode": "cautious",
    },
    "steel": {
        "fragile": False,
        "heat_conductivity": "high",
        "handling_mode": "cautious",
    },
    "plastic": {
        "fragile": False,
        "heat_conductivity": "low",
        "handling_mode": "normal",
    },
    "rubber": {
        "fragile": False,
        "heat_conductivity": "low",
        "handling_mode": "normal",
    },
    "wood": {
        "fragile": False,
        "heat_conductivity": "low",
        "handling_mode": "normal",
    },
    "paper": {
        "fragile": True,
        "heat_conductivity": "low",
        "handling_mode": "slow",
    },
    "default": {
        "fragile": False,
        "heat_conductivity": "medium",
        "handling_mode": "normal",
    },
}

TEMPERATURE_MEMBERSHIPS: Dict[str, Dict[str, float]] = {
    "cold": {"cool": 1.0, "warm": 0.0, "hot": 0.0},
    "roomtemp": {"cool": 0.6, "warm": 0.15, "hot": 0.0},
    "hot": {"cool": 0.0, "warm": 0.35, "hot": 1.0},
}

RISKY_COMMAND_PATTERNS = {
    "place": re.compile(
        r"(?:put|place|keep|insert|store)\s+(?:the\s+|a\s+|an\s+)?(?P<object>.+?)\s+"
        r"(?:in|into|inside|on|onto)\s+(?:the\s+|a\s+|an\s+)?(?P<target>[^,.!?]+)",
        re.IGNORECASE,
    ),
    "pour": re.compile(
        r"(?:pour|spill|spray)\s+(?:the\s+|a\s+|an\s+)?(?P<object>.+?)\s+"
        r"(?:in|into|inside|on|onto)\s+(?:the\s+|a\s+|an\s+)?(?P<target>[^,.!?]+)",
        re.IGNORECASE,
    ),
    "throw": re.compile(
        r"(?:throw|drop|toss|hurl)\s+(?:the\s+|a\s+|an\s+)?(?P<object>[^,.!?]+)",
        re.IGNORECASE,
    ),
    "heat": re.compile(
        r"(?:heat|cook|microwave|toast|burn)\s+(?:the\s+|a\s+|an\s+)?(?P<object>.+?)"
        r"(?:\s+(?:in|inside|on|using|with)\s+(?:the\s+|a\s+|an\s+)?(?P<target>[^,.!?]+))?$",
        re.IGNORECASE,
    ),
}

RISKY_COMMAND_TOKENS = {
    "microwave", "oven", "toaster", "stove", "burner",
    "put", "place", "insert", "store", "pour", "spill", "spray",
    "throw", "drop", "toss", "hurl", "heat", "cook", "burn",
}


def normalize_object_name(name: Optional[str]) -> str:
    if not name:
        return ""
    text = str(name).lower().strip()
    for alias, canonical in sorted(COMMAND_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(alias, canonical)
    return re.sub(r"[^a-z0-9]+", "", text)


def extract_current_task_text(command: Optional[str]) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    match = re.search(r"current task:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    return re.sub(r"\s+", " ", text).strip()


def classify_semantic_category(name: Optional[str]) -> str:
    norm = normalize_object_name(name)
    if not norm:
        return "unknown"
    if norm in SEMANTIC_CATEGORIES:
        return SEMANTIC_CATEGORIES[norm]
    if any(token in norm for token in ("card", "document", "wallet", "passport")):
        return "document_card"
    if any(token in norm for token in ("phone", "remote", "tv", "laptop", "charger", "battery", "cable")):
        return "electronic"
    if any(token in norm for token in ("knife", "fork", "scissor", "blade", "needle")):
        return "sharp"
    if any(token in norm for token in ("water", "coffee", "milk", "juice", "liquid", "bleach", "detergent", "soap")):
        return "liquid"
    if any(token in norm for token in ("microwave", "toaster", "oven", "stove", "burner")):
        return "heating_appliance"
    if any(token in norm for token in ("plasticbag", "paper", "bag", "newspaper", "cloth", "tissue")):
        return "flammable"
    if any(token in norm for token in ("glass", "ceramic", "vase")):
        return "fragile_item"
    if any(token in norm for token in ("metal", "aluminum", "steel")):
        return "conductive_item"
    return "unknown"


def _contains_any(text: str, tokens: List[str]) -> bool:
    return any(token in text for token in tokens)


_PRONOUNS_AND_STOPWORDS = {
    "it", "them", "this", "that", "these", "those", "one", "ones",
    "here", "there", "which", "what", "something", "anything", "nothing",
}

def _matches_known_object(name: str, scene_objects: Optional[List[str]] = None) -> bool:
    norm = normalize_object_name(name)
    if not norm:
        return True
    # Pronouns and stopwords are not real objects — do not flag as unknown
    if norm in _PRONOUNS_AND_STOPWORDS:
        return True
    if classify_semantic_category(norm) != "unknown":
        return True

    scene_objects = scene_objects or []
    for candidate in list(AI2THOR_KNOWN_OBJECT_TYPES) + list(scene_objects):
        known = normalize_object_name(candidate)
        if not known:
            continue
        if norm == known or norm in known or known in norm:
            return True
    return False


def _extract_categories(text: str) -> Set[str]:
    categories: Set[str] = set()
    normalized = normalize_object_name(text)
    for token in re.findall(r"[a-z0-9]+", normalized):
        cat = classify_semantic_category(token)
        if cat != "unknown":
            categories.add(cat)
    for alias in COMMAND_ALIASES.values():
        if alias in normalized:
            cat = classify_semantic_category(alias)
            if cat != "unknown":
                categories.add(cat)
    category_keywords = {
        "document_card": ["wallet", "card", "passport", "license", "receipt"],
        "valuable_small_item": ["keys", "key", "wallet"],
        "electronic": ["charger", "battery", "laptop", "phone", "remote"],
        "flammable": ["paper", "newspaper", "cloth", "tissue", "napkin", "bag"],
        "chemical_liquid": ["bleach", "detergent", "cleaner", "soap"],
        "flammable_liquid": ["oil", "alcohol", "sanitizer"],
        "fragile_item": ["glass", "ceramic", "vase"],
        "conductive_item": ["metal", "aluminum", "steel"],
    }
    raw_lower = text.lower()
    for category, tokens in category_keywords.items():
        if _contains_any(raw_lower, tokens):
            categories.add(category)
    return categories


def needs_open_vocab_safety_review(
    command: str,
    scene_objects: Optional[List[str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    effective = extract_current_task_text(command)
    lowered = effective.lower()
    meta: Dict[str, Any] = {
        "effective_command": effective,
        "risky_context": False,
        "unknown_entities": [],
        "matched_entities": [],
        "pattern": "",
    }

    if not effective or not _contains_any(lowered, list(RISKY_COMMAND_TOKENS)):
        return False, meta

    meta["risky_context"] = True

    for pattern_name, pattern in RISKY_COMMAND_PATTERNS.items():
        match = pattern.search(effective)
        if not match:
            continue

        meta["pattern"] = pattern_name
        entities: List[Dict[str, Any]] = []
        for key in ("object", "target"):
            value = (match.groupdict().get(key) or "").strip()
            if not value:
                continue
            known = _matches_known_object(value, scene_objects=scene_objects)
            category = classify_semantic_category(value)
            entity = {
                "role": key,
                "text": value,
                "normalized": normalize_object_name(value),
                "known": known,
                "category": category,
            }
            entities.append(entity)
        meta["matched_entities"] = entities
        meta["unknown_entities"] = [e["text"] for e in entities if not e["known"]]
        return bool(meta["unknown_entities"]), meta

    return False, meta


def _normalize_typos(text: str) -> str:
    """Normalize common misspellings of appliance/object names so safety
    rules cannot be bypassed by typos (e.g. 'microven' → 'microwave')."""
    import re as _re
    typo_map = {
        r"\bmicrow?a?ve?\w{0,2}\b": "microwave",  # microven, microwav, microwve, etc.
        r"\btoaste?r?\b": "toaster",               # toaste, toastr
        r"\bove?n\b": "oven",                      # ovn
        r"\bstov?e?\b": "stove",                   # stov, stve
    }
    for pattern, replacement in typo_map.items():
        # Only replace if the original token is NOT already the correct word
        text = _re.sub(pattern, replacement, text)
    return text


def check_command_safety(command: str) -> Tuple[bool, str, Dict[str, Any]]:
    effective = extract_current_task_text(command)
    text = _normalize_typos(effective.lower().strip())
    meta: Dict[str, Any] = {
        "rule": "",
        "matched_terms": [],
        "matched_categories": sorted(_extract_categories(text)),
        "effective_command": effective,
    }

    heat_tokens = ["microwave", "toaster", "oven", "stove", "burner"]
    electric_tokens = ["microwave", "toaster", "coffee machine", "coffeemachine", "tv", "television", "laptop", "charger"]
    liquid_tokens = ["water", "coffee", "milk", "juice", "soap", "bleach", "detergent", "oil"]
    sharp_tokens = ["knife", "fork", "scissors", "blade", "needle"]
    non_food_heat_tokens = [
        "debit card", "credit card", "id card", "card", "phone", "remote",
        "book", "wallet", "keys", "key", "charger", "battery", "laptop",
        "newspaper", "cloth", "tissue", "bleach", "detergent", "oil",
    ]
    flammable_tokens = ["plastic bag", "paper bag", "paper", "book", "bag", "card", "newspaper", "cloth", "tissue", "napkin"]
    fragile_tokens = ["glass", "ceramic", "vase", "wine bottle", "glass bottle", "plate", "cup", "mug", "bowl"]
    throw_tokens = ["throw", "toss", "hurl", "drop"]
    place_tokens = ["put", "place", "keep", "insert"]

    categories = set(meta["matched_categories"])

    if (_contains_any(text, liquid_tokens) or categories.intersection({"liquid", "chemical_liquid", "flammable_liquid"})) and _contains_any(text, electric_tokens):
        meta["rule"] = "liquid_on_electrical"
        meta["matched_terms"] = [tok for tok in liquid_tokens + electric_tokens if tok in text]
        return False, "Unsafe command: do not pour liquid on electrical appliances", meta

    if (_contains_any(text, non_food_heat_tokens) or categories.intersection({
        "document_card", "valuable_small_item", "electronic", "flammable",
        "chemical_liquid", "flammable_liquid", "hazardous_electronic",
    })) and _contains_any(text, heat_tokens):
        meta["rule"] = "non_food_heat_appliance"
        meta["matched_terms"] = [tok for tok in non_food_heat_tokens + heat_tokens if tok in text]
        return False, "Unsafe command: non-food or document items must not be placed in heating appliances", meta

    if (_contains_any(text, sharp_tokens) or "sharp" in categories) and _contains_any(text, throw_tokens):
        meta["rule"] = "throw_sharp_object"
        meta["matched_terms"] = [tok for tok in sharp_tokens + throw_tokens if tok in text]
        return False, "Unsafe command: sharp objects must not be thrown", meta

    if (_contains_any(text, flammable_tokens) or categories.intersection({"flammable", "flammable_liquid"})) and _contains_any(text, heat_tokens):
        meta["rule"] = "flammable_near_heat"
        meta["matched_terms"] = [tok for tok in flammable_tokens + heat_tokens if tok in text]
        return False, "Unsafe command: flammable items must not be placed near heating appliances", meta

    if (_contains_any(text, fragile_tokens) or categories.intersection({"fragile_item", "fragile_container"})) and _contains_any(text, throw_tokens):
        meta["rule"] = "throw_fragile_object"
        meta["matched_terms"] = [tok for tok in fragile_tokens + throw_tokens if tok in text]
        return False, "Unsafe command: fragile glass or ceramic objects must not be thrown or dropped", meta

    if (_contains_any(text, ["metal pan", "metal plate", "aluminum foil", "aluminium foil"]) or categories.intersection({"conductive_item", "conductive_cookware"})) and "microwave" in text and _contains_any(text, place_tokens):
        meta["rule"] = "conductive_in_microwave"
        meta["matched_terms"] = [tok for tok in ["metal pan", "metal plate", "aluminum foil", "aluminium foil", "microwave"] if tok in text]
        return False, "Unsafe command: metal or highly conductive objects must not be placed in the microwave", meta

    # Sharp/metal objects (knife, fork, blade) in microwave — metal sparks and is a fire hazard.
    if (_contains_any(text, sharp_tokens) or "sharp" in categories) and "microwave" in text and _contains_any(text, place_tokens + ["inside", "into", "in the"]):
        meta["rule"] = "sharp_metal_in_microwave"
        meta["matched_terms"] = [tok for tok in sharp_tokens + ["microwave"] if tok in text]
        return False, "Unsafe command: sharp metal objects (knife, fork, blade) must not be placed in the microwave", meta

    # Damaging/destroying documents, cards, or valuables with a sharp tool.
    document_tokens = ["credit card", "creditcard", "debit card", "debitcard", "id card", "passport", "document", "bank card"]
    damage_tokens   = ["cut", "slice", "break", "smash", "burn", "destroy", "damage", "shred"]
    if _contains_any(text, document_tokens) and _contains_any(text, damage_tokens):
        meta["rule"] = "damage_document"
        meta["matched_terms"] = [tok for tok in document_tokens + damage_tokens if tok in text]
        return False, "Unsafe command: do not damage or destroy documents, cards, or valuables", meta

    return True, "", meta


def validate_task_queue_safety(task_queue: List[Dict[str, Any]]) -> Tuple[bool, str, List[Dict[str, Any]]]:
    for task in task_queue:
        for action in task.get("actions", []):
            intent = action.get("intent", "")
            obj = action.get("object") or action.get("target") or ""
            rec = action.get("receptacle", "")
            obj_cat = classify_semantic_category(obj)
            rec_cat = classify_semantic_category(rec)
            rec_norm = normalize_object_name(rec)

            if intent == "place" and rec_norm in HEATING_APPLIANCES:
                if obj_cat in {
                    "document_card", "valuable_small_item", "electronic", "hazardous_electronic",
                    "flammable", "chemical_liquid", "flammable_liquid", "unknown",
                } and normalize_object_name(obj) not in {
                    "mug",
                    "cup",
                    "bowl",
                    "pan",
                    "pot",
                    "bread",
                    "breadsliced",
                    "potato",
                    "potatosliced",
                    "egg",
                    "eggcracked",
                }:
                    return False, f"Unsafe plan: cannot place {obj or 'item'} in {rec}", task_queue

            # Sharp metal objects (knife, fork) in microwave → fire/sparking hazard
            if intent == "place" and rec_norm == "microwave" and obj_cat in {"sharp", "conductive_item", "conductive_cookware"}:
                return False, f"Unsafe plan: cannot place metal/sharp object {obj} in Microwave", task_queue

            if intent == "pour" and rec_cat in {"electrical_appliance", "heating_appliance"}:
                return False, f"Unsafe plan: cannot pour liquid into or onto {rec}", task_queue

            if intent == "throw" and obj_cat in {"sharp", "fragile_item", "fragile_container"}:
                return False, f"Unsafe plan: cannot throw unsafe object {obj}", task_queue

            if intent == "drop" and obj_cat in {"fragile_item", "fragile_container"}:
                return False, f"Unsafe plan: cannot drop fragile object {obj}", task_queue

    return True, "", task_queue


def get_fuzzy_temperature(obj_meta: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not obj_meta:
        return dict(TEMPERATURE_MEMBERSHIPS["roomtemp"])
    raw_temp = normalize_object_name(
        obj_meta.get("ObjectTemperature")
        or obj_meta.get("temperature")
        or "RoomTemp"
    )
    return dict(TEMPERATURE_MEMBERSHIPS.get(raw_temp, TEMPERATURE_MEMBERSHIPS["roomtemp"]))


def get_material_profile(obj_meta: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    if not obj_meta:
        return "unknown", dict(MATERIAL_PROFILES["default"])
    materials = obj_meta.get("salientMaterials", []) or []
    for material in materials:
        norm = normalize_object_name(material)
        if norm in MATERIAL_PROFILES:
            return material, dict(MATERIAL_PROFILES[norm])
    return "unknown", dict(MATERIAL_PROFILES["default"])


def _material_is_fragile(profile: Dict[str, Any], obj_meta: Optional[Dict[str, Any]]) -> bool:
    if profile.get("fragile", False):
        return True
    return bool(obj_meta and obj_meta.get("breakable", False))


def check_action_safety(
    intent: str,
    obj_meta: Optional[Dict[str, Any]],
    action: Optional[Dict[str, Any]] = None,
    held_obj: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    target = obj_meta or held_obj or {}
    fuzzy = get_fuzzy_temperature(target)
    material, profile = get_material_profile(target)
    fragile = _material_is_fragile(profile, target)
    obj_type = target.get("objectType") or (action or {}).get("object") or "object"
    obj_cat = classify_semantic_category(obj_type)
    meta = {
        "temperature": target.get("ObjectTemperature", "RoomTemp"),
        "fuzzy_temperature": fuzzy,
        "material": material,
        "fragile": fragile,
        "handling_mode": profile.get("handling_mode", "normal"),
        "object_type": obj_type,
    }

    if intent == "pick":
        if fuzzy["hot"] >= 0.8:
            # Exception 1: a hot mug/cup coming directly out of a CoffeeMachine is
            # intentionally hot — blocking it would prevent every coffee task.
            if obj_type.lower() in ("mug", "cup"):
                parents = (obj_meta or {}).get("parentReceptacles") or []
                if any("CoffeeMachine" in p for p in parents):
                    return True, (
                        f"[SAFETY_WARN] {obj_type} is hot from CoffeeMachine; handle carefully"
                    ), meta
            # Exception 2: food that is isCooked=True is intentionally hot from
            # cooking (e.g. EggCracked after frying). Blocking it would break
            # every serve step that follows a cook step.
            if (obj_meta or {}).get("isCooked", False):
                return True, (
                    f"[SAFETY_WARN] {obj_type} is hot from cooking; handle carefully"
                ), meta
            # Exception 3: food items coming out of a Toaster are intentionally
            # hot (toasting doesn't set isCooked in AI2-THOR).  Blocking pickup
            # would prevent every toast task from completing.
            if obj_cat == "food":
                parents = (obj_meta or {}).get("parentReceptacles") or []
                if any("Toaster" in p for p in parents):
                    return True, (
                        f"[SAFETY_WARN] {obj_type} is hot from toasting; handle carefully"
                    ), meta
            if fragile:
                return False, f"[SAFETY_BLOCK] {obj_type} is hot and fragile ({material})", meta
            if profile.get("heat_conductivity") == "high":
                return False, f"[SAFETY_BLOCK] {obj_type} is hot and highly conductive ({material})", meta
            return False, f"[SAFETY_BLOCK] {obj_type} is too hot to pick safely", meta
        if fuzzy["warm"] >= 0.5:
            return True, f"[SAFETY_WARN] {obj_type} is warm; use cautious handling", meta

    if intent in ("drop", "throw"):
        if fragile:
            verb = "dropped" if intent == "drop" else "thrown"
            return False, f"[SAFETY_BLOCK] {obj_type} is fragile and cannot be {verb} safely", meta
        if intent == "throw" and obj_cat == "sharp":
            return False, f"[SAFETY_BLOCK] Sharp object {obj_type} cannot be thrown", meta
        if fuzzy["hot"] >= 0.8:
            return False, f"[SAFETY_BLOCK] {obj_type} is too hot to {intent} safely", meta

    if intent == "place" and fuzzy["hot"] >= 0.8 and fragile:
        return True, f"[SAFETY_WARN] Place hot fragile {obj_type} only on a stable safe surface", meta

    return True, "", meta


def classify_failure_reason(message: str) -> str:
    text = (message or "").upper()
    if "[SAFETY_BLOCK]" in text:
        return "safety_block"
    if "[SAFETY_WARN]" in text:
        return "safety_warn"
    if "[OBJECT_NOT_FOUND]" in text:
        return "object_not_found"
    if "[INVALID_ACTION]" in text:
        return "invalid_action"
    if "[RECOVERY]" in text:
        return "recovery"
    return "other"


def summarize_scene_safety(objects: List[Dict[str, Any]]) -> str:
    hot_items: List[str] = []
    warm_items: List[str] = []
    fragile_items: List[str] = []

    for obj in objects:
        obj_type = obj.get("objectType", "Object")
        fuzzy = get_fuzzy_temperature(obj)
        material, profile = get_material_profile(obj)
        if fuzzy["hot"] >= 0.8:
            hot_items.append(f"{obj_type}({material or 'unknown'})")
        elif fuzzy["warm"] >= 0.5:
            warm_items.append(f"{obj_type}({material or 'unknown'})")
        if _material_is_fragile(profile, obj):
            fragile_items.append(obj_type)

    parts = []
    if hot_items:
        parts.append("hot=" + ", ".join(hot_items[:8]))
    if warm_items:
        parts.append("warm=" + ", ".join(warm_items[:8]))
    if fragile_items:
        parts.append("fragile=" + ", ".join(sorted(set(fragile_items))[:8]))
    return " | ".join(parts) if parts else "no notable thermal/material hazards detected"
