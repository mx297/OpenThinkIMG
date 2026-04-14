from __future__ import annotations

import copy
from typing import Any, Dict, List

from tool_server.tool_workers.tool_manager.base_manager import ToolManager

from .strict_tool_schema import extract_terminate_answer, validate_tool_message
from .tool_generation import append_conversation_fn, handle_tool_result, pil_to_base64
from .tool_tbgfn_state import SafeToolGFNState


def _make_error_result(message: str, error_type: str = "TOOL_ERROR") -> Dict[str, Any]:
    return {"text": f"{error_type}: {message}", "error_code": 1}


class SafeToolGFNEnv:
    def __init__(self, controller_addr: str, max_rounds: int = 3) -> None:
        self.controller_addr = controller_addr
        self.max_rounds = max_rounds
        self.tool_manager = ToolManager(controller_addr)
        self.tool_manager.available_tools = [
            tool for tool in self.tool_manager.available_tools if tool not in ["crop", "drawline"]
        ]

    def reset(self, prompt: str, image: Any) -> SafeToolGFNState:
        current_image = image
        if current_image and getattr(current_image, "mode", None) in ("RGBA", "LA", "P"):
            current_image = current_image.convert("RGB")
        current_image_base64 = pil_to_base64(current_image)
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": current_image_base64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return SafeToolGFNState(prompt=prompt, conversations=conversations, images=[current_image])

    def step(self, state: SafeToolGFNState, action_text: str) -> SafeToolGFNState:
        state.model_outputs.append(action_text)
        state.conversations = append_conversation_fn(
            conversation=state.conversations,
            text=action_text,
            role="assistant",
        )
        validation = validate_tool_message(action_text)
        state.validation_results.append(validation.__dict__)

        if not validation.is_valid:
            tool_result = _make_error_result(
                validation.error_message,
                error_type=validation.error_type or "TOOL_SCHEMA_ERROR",
            )
            cfg = {"API_name": validation.action_name or "ToolValidator"}
            state.tool_outputs.append(tool_result)
            state.conversations = handle_tool_result(
                cfg=cfg,
                tool_result=tool_result,
                conversations=state.conversations,
                model_mode="general",
                original_prompt=state.prompt,
                input_data_item={"images": state.images},
            )
            state.round_idx += 1
            if state.round_idx >= self.max_rounds:
                state.status = "max_rounds"
            return state

        cfg = validation.as_api_config()
        api_name = validation.action_name
        api_params = validation.action_arguments or {}

        if api_name == "" and api_params == {}:
            state.conversations = append_conversation_fn(
                conversation=state.conversations,
                text=state.prompt,
                role="user",
            )
            state.round_idx += 1
            if state.round_idx >= self.max_rounds:
                state.status = "max_rounds"
            return state

        state.tool_cfgs.append(cfg)

        if api_name == "Terminate":
            state.extracted_answer = extract_terminate_answer(action_text)
            state.status = "finished"
            state.round_idx += 1
            return state

        if api_name not in self.tool_manager.available_tools:
            tool_result = _make_error_result(
                f"Tool {api_name} is valid but not currently available. Available tools: {', '.join(sorted(self.tool_manager.available_tools))}.",
                error_type="TOOL_RUNTIME_ERROR",
            )
        else:
            try:
                tool_result = self.tool_manager.call_tool(api_name, api_params)
            except Exception as exc:
                tool_result = _make_error_result(
                    f"Failed to call tool {api_name}: {exc}",
                    error_type="TOOL_CALL_EXCEPTION",
                )
            if tool_result is None:
                tool_result = _make_error_result(
                    f"Tool {api_name} returned no result.",
                    error_type="TOOL_RUNTIME_ERROR",
                )
            elif not isinstance(tool_result, dict):
                tool_result = _make_error_result(
                    f"Tool {api_name} returned an invalid result object: {tool_result}",
                    error_type="TOOL_RUNTIME_ERROR",
                )
            elif tool_result.get("error_code", 0) != 0:
                existing_text = tool_result.get("text", "")
                if not existing_text.startswith("TOOL_RUNTIME_ERROR:"):
                    tool_result["text"] = f"TOOL_RUNTIME_ERROR: {existing_text}" if existing_text else f"TOOL_RUNTIME_ERROR: Tool {api_name} failed with error_code={tool_result.get('error_code')}"

        state.tool_outputs.append(tool_result)
        state.conversations = handle_tool_result(
            cfg=cfg,
            tool_result=tool_result,
            conversations=state.conversations,
            model_mode="general",
            original_prompt=state.prompt,
            input_data_item={"images": state.images},
        )
        state.round_idx += 1
        if state.round_idx >= self.max_rounds and state.status == "processing":
            state.status = "max_rounds"
        return state
