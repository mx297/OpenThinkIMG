import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

NUMBER_PATTERN = r'-?(?:0|[1-9]\d*)(?:\.\d+)?'
IMAGE_REF_RE = re.compile(r'^img_[1-9]\d*$')
SEGMENT_PARAM_RE = re.compile(rf'^x="{NUMBER_PATTERN}" y="{NUMBER_PATTERN}"$')
DRAW_Y_PARAM_RE = re.compile(rf'^y={NUMBER_PATTERN}$')
DRAW_X_PARAM_RE = re.compile(rf'^x={NUMBER_PATTERN}$')

ALLOWED_TOOL_NAMES = (
    "OCR",
    "Point",
    "ZoomInSubfigure",
    "SegmentRegionAroundPoint",
    "DrawHorizontalLineByY",
    "DrawVerticalLineByX",
    "Terminate",
)

TOOL_ARGUMENT_KEYS = {
    "OCR": {"image"},
    "Point": {"image", "param"},
    "ZoomInSubfigure": {"image", "param"},
    "SegmentRegionAroundPoint": {"image", "param"},
    "DrawHorizontalLineByY": {"image", "param"},
    "DrawVerticalLineByX": {"image", "param"},
    "Terminate": {"ans"},
}


@dataclass
class ValidationResult:
    is_valid: bool
    score: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    parsed_data: Optional[Dict[str, Any]] = None
    action_name: Optional[str] = None
    action_arguments: Optional[Dict[str, Any]] = None

    def as_api_config(self) -> Optional[Dict[str, Any]]:
        if not self.is_valid or self.action_name is None or self.action_arguments is None:
            return None
        return {"API_name": self.action_name, "API_params": self.action_arguments}


def _make_result(
    is_valid: bool,
    score: float,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    parsed_data: Optional[Dict[str, Any]] = None,
    action_name: Optional[str] = None,
    action_arguments: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    return ValidationResult(
        is_valid=is_valid,
        score=score,
        error_type=error_type,
        error_message=error_message,
        parsed_data=parsed_data,
        action_name=action_name,
        action_arguments=action_arguments,
    )


def _ensure_clean_string(value: Any, field_name: str) -> Optional[str]:
    if not isinstance(value, str):
        return f"{field_name} must be a JSON string."
    if value != value.strip():
        return f"{field_name} must not contain leading or trailing whitespace."
    if not value:
        return f"{field_name} must not be empty."
    if "\n" in value or "\r" in value:
        return f"{field_name} must be a single-line string."
    return None


def _validate_image_ref(arguments: Dict[str, Any]) -> Optional[str]:
    error = _ensure_clean_string(arguments.get("image"), "arguments.image")
    if error is not None:
        return error
    if not IMAGE_REF_RE.fullmatch(arguments["image"]):
        return 'arguments.image must exactly match the pattern img_<number>, for example "img_1".'
    return None


def _validate_argument_values(tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
    if tool_name != "Terminate":
        image_error = _validate_image_ref(arguments)
        if image_error is not None:
            return image_error

    if tool_name == "OCR":
        return None

    if tool_name in {"Point", "ZoomInSubfigure"}:
        return _ensure_clean_string(arguments.get("param"), "arguments.param")

    if tool_name == "SegmentRegionAroundPoint":
        error = _ensure_clean_string(arguments.get("param"), "arguments.param")
        if error is not None:
            return error
        if not SEGMENT_PARAM_RE.fullmatch(arguments["param"]):
            return 'SegmentRegionAroundPoint.arguments.param must exactly match x="<number>" y="<number>", for example x="21.5" y="28.5".'
        return None

    if tool_name == "DrawHorizontalLineByY":
        error = _ensure_clean_string(arguments.get("param"), "arguments.param")
        if error is not None:
            return error
        if not DRAW_Y_PARAM_RE.fullmatch(arguments["param"]):
            return 'DrawHorizontalLineByY.arguments.param must exactly match y=<number>, for example y=28.5.'
        return None

    if tool_name == "DrawVerticalLineByX":
        error = _ensure_clean_string(arguments.get("param"), "arguments.param")
        if error is not None:
            return error
        if not DRAW_X_PARAM_RE.fullmatch(arguments["param"]):
            return 'DrawVerticalLineByX.arguments.param must exactly match x=<number>, for example x=21.5.'
        return None

    if tool_name == "Terminate":
        error = _ensure_clean_string(arguments.get("ans"), "arguments.ans")
        if error is not None:
            return error
        return None

    return f"Unsupported tool schema for {tool_name}."


def validate_tool_message(output_text: str) -> ValidationResult:
    if not isinstance(output_text, str) or not output_text.strip():
        return _make_result(
            False,
            0.0,
            "PARSER_ERROR",
            'response must be a non-empty JSON object with keys "thought" and "actions".',
        )

    stripped_text = output_text.strip()
    try:
        parsed_data = json.loads(stripped_text)
    except Exception:
        return _make_result(
            False,
            0.0,
            "PARSER_ERROR",
            'response must be a single JSON object and nothing else.',
        )

    score = 0.2
    if not isinstance(parsed_data, dict):
        return _make_result(
            False,
            score,
            "PARSER_ERROR",
            'response must decode to a JSON object, not a list or scalar value.',
        )

    expected_top_level_keys = {"thought", "actions"}
    if set(parsed_data.keys()) != expected_top_level_keys:
        return _make_result(
            False,
            score,
            "TOOL_SCHEMA_ERROR",
            'top-level keys must be exactly {"thought", "actions"}.',
            parsed_data=parsed_data,
        )

    if not isinstance(parsed_data.get("thought"), str):
        return _make_result(
            False,
            score,
            "TOOL_SCHEMA_ERROR",
            '"thought" must be a JSON string.',
            parsed_data=parsed_data,
        )

    if not isinstance(parsed_data.get("actions"), list):
        return _make_result(
            False,
            score,
            "TOOL_SCHEMA_ERROR",
            '"actions" must be a JSON list.',
            parsed_data=parsed_data,
        )

    score = 0.35
    actions = parsed_data["actions"]
    if len(actions) == 0:
        return _make_result(
            True,
            1.0,
            parsed_data=parsed_data,
            action_name="",
            action_arguments={},
        )

    if len(actions) != 1:
        return _make_result(
            False,
            score,
            "TOOL_SCHEMA_ERROR",
            '"actions" must contain exactly one action per response.',
            parsed_data=parsed_data,
        )

    action = actions[0]
    if not isinstance(action, dict):
        return _make_result(
            False,
            score,
            "TOOL_SCHEMA_ERROR",
            'each action must be a JSON object with keys "name" and "arguments".',
            parsed_data=parsed_data,
        )

    if set(action.keys()) != {"name", "arguments"}:
        return _make_result(
            False,
            score,
            "TOOL_SCHEMA_ERROR",
            'action keys must be exactly {"name", "arguments"}.',
            parsed_data=parsed_data,
        )

    if not isinstance(action.get("name"), str):
        return _make_result(
            False,
            score,
            "TOOL_SCHEMA_ERROR",
            'action.name must be a JSON string.',
            parsed_data=parsed_data,
        )

    if not isinstance(action.get("arguments"), dict):
        return _make_result(
            False,
            score,
            "TOOL_SCHEMA_ERROR",
            'action.arguments must be a JSON object.',
            parsed_data=parsed_data,
            action_name=action.get("name"),
        )

    score = 0.5
    action_name = action["name"]
    action_arguments = action["arguments"]

    if action_name not in ALLOWED_TOOL_NAMES:
        return _make_result(
            False,
            score,
            "TOOL_NAME_ERROR",
            f"tool name must exactly match one of: {', '.join(ALLOWED_TOOL_NAMES)}.",
            parsed_data=parsed_data,
            action_name=action_name,
            action_arguments=action_arguments,
        )

    score = 0.65
    expected_argument_keys = TOOL_ARGUMENT_KEYS[action_name]
    if set(action_arguments.keys()) != expected_argument_keys:
        return _make_result(
            False,
            score,
            "TOOL_SCHEMA_ERROR",
            f"{action_name}.arguments keys must be exactly {sorted(expected_argument_keys)}.",
            parsed_data=parsed_data,
            action_name=action_name,
            action_arguments=action_arguments,
        )

    score = 0.8
    value_error = _validate_argument_values(action_name, action_arguments)
    if value_error is not None:
        return _make_result(
            False,
            score,
            "TOOL_SCHEMA_ERROR",
            value_error,
            parsed_data=parsed_data,
            action_name=action_name,
            action_arguments=action_arguments,
        )

    return _make_result(
        True,
        1.0,
        parsed_data=parsed_data,
        action_name=action_name,
        action_arguments=action_arguments,
    )


def score_tool_message(output_text: str) -> float:
    return validate_tool_message(output_text).score


def extract_terminate_answer(output_text: str) -> Optional[str]:
    validation = validate_tool_message(output_text)
    if not validation.is_valid or validation.action_name != "Terminate":
        return None
    return validation.action_arguments.get("ans")
