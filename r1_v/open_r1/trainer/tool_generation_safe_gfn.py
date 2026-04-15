import copy
from typing import Any, Dict, List

from tool_server.tool_workers.tool_manager.base_manager import ToolManager

from .strict_tool_schema import validate_tool_message
from .tool_generation import append_conversation_fn, handle_tool_result, pil_to_base64


def _make_error_result(message: str, error_type: str = "TOOL_ERROR") -> Dict[str, Any]:
    return {"text": f"{error_type}: {message}", "error_code": 1}


def _snapshot_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return copy.deepcopy(messages)


def vllm_generate_with_tool_calls_gfn(
    vllm_model,
    prompts,
    images,
    sampling_params=None,
    max_rounds: int = 3,
    model_mode: str = "general",
    controller_addr: str = "http://SH-IDCA1404-10-140-54-5:20001",
):
    tool_manager = ToolManager(controller_addr)
    tool_manager.available_tools = [tool for tool in tool_manager.available_tools if tool not in ["crop", "drawline"]]

    input_data = []
    for prompt, image in zip(prompts, images):
        current_image = image
        if current_image and current_image.mode in ("RGBA", "LA", "P"):
            current_image = current_image.convert("RGB")

        current_image_base64 = pil_to_base64(current_image)
        if isinstance(prompt, list):
            initial_user_messages = copy.deepcopy(prompt)
            for p in initial_user_messages:
                for c in p["content"]:
                    if c["type"] == "image":
                        c["type"] = "image_url"
                        c["image_url"] = {"url": current_image_base64}
                        c.pop("image", None)
            contents = initial_user_messages[-1]["content"]
            current_prompt = ""
            for content in contents:
                if content["type"] == "text" and content["text"]:
                    current_prompt += content["text"]
        elif isinstance(prompt, str):
            current_prompt = prompt
            initial_user_messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": current_image_base64}},
                    {"type": "text", "text": current_prompt},
                ],
            }]
        else:
            raise ValueError("Prompt should be either a string or a list of messages")

        input_data.append(
            dict(
                conversations=initial_user_messages,
                status="processing",
                model_outputs=[],
                model_output_ids=[],
                tool_cfgs=[],
                tool_outputs=[],
                validation_results=[],
                images=[current_image],
                prompt=current_prompt,
                turn_records=[],
            )
        )

    for _ in range(max_rounds):
        input_conversations = [item["conversations"] for item in input_data if item["status"] == "processing"]
        input_idxs = [idx for idx, item in enumerate(input_data) if item["status"] == "processing"]
        if not input_conversations:
            break

        try:
            outputs = vllm_model.chat(input_conversations, sampling_params=sampling_params, use_tqdm=False)
            output_texts = [output.outputs[0].text for output in outputs]
            output_idss = [output.outputs[0].token_ids for output in outputs]
        except Exception as e:
            print(f"[vllm generation] {e}")
            output_texts = ["Model generation error"] * len(input_conversations)
            output_idss = [(1712, 9471, 1465, 151645)] * len(input_conversations)

        for input_idx, output_text, output_ids in zip(input_idxs, output_texts, output_idss):
            item = input_data[input_idx]
            item["turn_records"].append(
                {
                    "state_messages_before_turn": _snapshot_messages(item["conversations"]),
                    "assistant_action_text": output_text,
                }
            )
            item["model_outputs"].append(output_text)
            item["model_output_ids"].append(output_ids)
            item["conversations"] = append_conversation_fn(conversation=item["conversations"], text=output_text, role="assistant")

            validation = validate_tool_message(output_text)
            item["validation_results"].append(validation.__dict__)

            if not validation.is_valid:
                tool_result = _make_error_result(validation.error_message, error_type=validation.error_type or "TOOL_SCHEMA_ERROR")
                cfg = {"API_name": validation.action_name or "ToolValidator"}
                item["tool_outputs"].append(tool_result)
                item["conversations"] = handle_tool_result(
                    cfg=cfg,
                    tool_result=tool_result,
                    conversations=item["conversations"],
                    model_mode="general",
                    original_prompt=item["prompt"],
                    input_data_item=item,
                )
                continue

            cfg = validation.as_api_config()
            api_name = validation.action_name
            api_params = validation.action_arguments or {}

            if api_name == "" and api_params == {}:
                item["conversations"] = append_conversation_fn(conversation=item["conversations"], text=item["prompt"], role="user")
                continue

            item["tool_cfgs"].append(cfg)

            if api_name == "Terminate":
                item["status"] = "finished"
                continue

            if api_name not in tool_manager.available_tools:
                tool_result = _make_error_result(
                    f"Tool {api_name} is valid but not currently available. Available tools: {', '.join(sorted(tool_manager.available_tools))}.",
                    error_type="TOOL_RUNTIME_ERROR",
                )
            else:
                try:
                    tool_result = tool_manager.call_tool(api_name, api_params)
                except Exception as e:
                    tool_result = _make_error_result(f"Failed to call tool {api_name}: {e}", error_type="TOOL_CALL_EXCEPTION")

                if tool_result is None:
                    tool_result = _make_error_result(f"Tool {api_name} returned no result.", error_type="TOOL_RUNTIME_ERROR")
                elif not isinstance(tool_result, dict):
                    tool_result = _make_error_result(
                        f"Tool {api_name} returned an invalid result object: {tool_result}",
                        error_type="TOOL_RUNTIME_ERROR",
                    )
                elif tool_result.get("error_code", 0) != 0:
                    existing_text = tool_result.get("text", "")
                    if not existing_text.startswith("TOOL_RUNTIME_ERROR:"):
                        tool_result["text"] = (
                            f"TOOL_RUNTIME_ERROR: {existing_text}"
                            if existing_text
                            else f"TOOL_RUNTIME_ERROR: Tool {api_name} failed with error_code={tool_result.get('error_code')}"
                        )

            item["tool_outputs"].append(tool_result)
            item["conversations"] = handle_tool_result(
                cfg=cfg,
                tool_result=tool_result,
                conversations=item["conversations"],
                model_mode="general",
                original_prompt=item["prompt"],
                input_data_item=item,
            )

    return input_data
