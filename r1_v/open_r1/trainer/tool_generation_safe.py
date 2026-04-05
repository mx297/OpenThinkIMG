import copy
from typing import Optional

from tool_server.tool_workers.tool_manager.base_manager import ToolManager

from .tool_generation import (
    append_conversation_fn,
    handle_tool_result,
    parse_tool_config,
    pil_to_base64,
)


def _make_error_result(message: str, error_type: str = "TOOL_ERROR"):
    return {"text": f"{error_type}: {message}", "error_code": 1}


def vllm_generate_with_tool_calls(
    vllm_model,
    prompts,
    images,
    sampling_params=None,
    max_rounds: int = 3,
    model_mode: str = "general",
    controller_addr: str = "http://SH-IDCA1404-10-140-54-5:20001",
):
    tool_manager = ToolManager(controller_addr)
    tool_manager.available_tools = [
        tool for tool in tool_manager.available_tools if tool not in ["crop", "drawline"]
    ]
    print(f"controller_addr: {controller_addr}")
    print(f"Avaliable tools are {tool_manager.available_tools}")

    miss_tool = []
    for tool in [
        "ZoomInSubfigure",
        "DrawHorizontalLineByY",
        "OCR",
        "DrawVerticalLineByX",
        "SegmentRegionAroundPoint",
        "Point",
    ]:
        if tool not in tool_manager.available_tools:
            miss_tool.append(tool)
    if len(miss_tool) == 0:
        print("All tools are called successfully")
    else:
        print(f"Not all tools is called successfully, missing tool {miss_tool}")

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
            initial_user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": current_image_base64}},
                        {"type": "text", "text": current_prompt},
                    ],
                }
            ]
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
                new_round_input=[],
                images=[current_image],
                prompt=current_prompt,
            )
        )

    tool_name_mapping = {
        "drawhorizontallinebyy": "DrawHorizontalLineByY",
        "zoominsubfigure": "ZoomInSubfigure",
        "drawverticallinebyx": "DrawVerticalLineByX",
        "segmentregionaroundpoint": "SegmentRegionAroundPoint",
        "point": "Point",
        "ocr": "OCR",
        "terminate": "Terminate",
    }

    for _ in range(max_rounds):
        input_conversations = [item["conversations"] for item in input_data if item["status"] == "processing"]
        input_idxs = [idx for idx, item in enumerate(input_data) if item["status"] == "processing"]
        if not input_conversations:
            break

        try:
            outputs = vllm_model.chat(
                input_conversations,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            output_texts = [output.outputs[0].text for output in outputs]
            output_idss = [output.outputs[0].token_ids for output in outputs]
        except Exception as e:
            print(f"[vllm generation] {e}")
            output_texts = ["Model generation error"] * len(input_conversations)
            output_idss = [(1712, 9471, 1465, 151645)] * len(input_conversations)

        for input_idx, output_text, output_ids in zip(input_idxs, output_texts, output_idss):
            input_data[input_idx]["model_outputs"].append(output_text)
            input_data[input_idx]["model_output_ids"].append(output_ids)
            input_data[input_idx]["conversations"] = append_conversation_fn(
                conversation=input_data[input_idx]["conversations"],
                text=output_text,
                role="assistant",
            )

            tool_cfg = parse_tool_config(
                output_text,
                model_mode=model_mode,
                image_tool_manager=None,
                newest_image=input_data[input_idx]["images"][-1],
            )

            if not tool_cfg:
                tool_result = _make_error_result(
                    "Invalid tool JSON or missing actions list. No tool was executed.",
                    error_type="PARSER_ERROR",
                )
                input_data[input_idx]["tool_outputs"].append(tool_result)
                input_data[input_idx]["conversations"] = handle_tool_result(
                    cfg={"API_name": "ToolParser"},
                    tool_result=tool_result,
                    conversations=input_data[input_idx]["conversations"],
                    model_mode="general",
                    original_prompt=input_data[input_idx]["prompt"],
                    input_data_item=input_data[input_idx],
                )
                continue

            input_data[input_idx]["tool_cfgs"].append(tool_cfg)
            original_api_name = str(tool_cfg[0].get("API_name", "")).lower()
            api_params = tool_cfg[0].get("API_params", {}) or {}
            api_name = tool_name_mapping.get(original_api_name)

            if api_name == "Terminate":
                input_data[input_idx]["status"] = "finished"
                continue

            if api_name is None:
                tool_result = _make_error_result(
                    f"Invalid tool name '{tool_cfg[0].get('API_name')}'. Available tools: {', '.join(sorted(tool_manager.available_tools))}.",
                    error_type="TOOL_NAME_ERROR",
                )
            else:
                try:
                    if "param" in api_params:
                        print(f"Tool name: {api_name}, params: {api_params['param']}")
                    tool_result = tool_manager.call_tool(api_name, api_params)
                except Exception as e:
                    tool_result = _make_error_result(
                        f"Failed to call tool {api_name}: {e}",
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
                elif tool_result.get("error_code", 0) != 0 and not tool_result.get("text"):
                    tool_result["text"] = f"TOOL_RUNTIME_ERROR: Tool {api_name} failed with error_code={tool_result.get('error_code')}"

            input_data[input_idx]["tool_outputs"].append(tool_result)
            input_data[input_idx]["conversations"] = handle_tool_result(
                cfg=tool_cfg[0],
                tool_result=tool_result,
                conversations=input_data[input_idx]["conversations"],
                model_mode="general",
                original_prompt=input_data[input_idx]["prompt"],
                input_data_item=input_data[input_idx],
            )

    return input_data
