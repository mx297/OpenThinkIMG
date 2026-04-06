from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from r1_v.open_r1.trainer.turn_judge import (
    TrajectoryTurnJudge,
    build_turn_records_from_messages,
    validate_turn_scores,
)


def _extract_question(messages: Iterable[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("from") != "human":
            continue
        value = message.get("value", "")
        if "Question:" in value:
            return value.split("Question:", 1)[1].strip()
        if "OBSERVATION:" not in value:
            return value.strip()
    return ""



def _extract_record(entry: Any, fallback_id: str) -> Dict[str, Any]:
    if isinstance(entry, list):
        return {"id": fallback_id, "messages": entry, "image": None, "ground_truth": ""}
    if isinstance(entry, dict):
        messages = entry.get("messages") or entry.get("trajectory") or entry.get("conversations")
        if messages is None and "from" in entry and "value" in entry:
            messages = [entry]
        if messages is None:
            raise ValueError(f"Could not find trajectory messages in record {fallback_id}.")
        image = entry.get("image")
        if image is None and "image_path" in entry:
            image = {"path": entry["image_path"]}
        if image is None and "image_bytes_base64" in entry:
            image = {"bytes_base64": entry["image_bytes_base64"]}
        ground_truth = entry.get("ground_truth") or entry.get("answer") or entry.get("solution") or ""
        return {
            "id": entry.get("id", fallback_id),
            "messages": messages,
            "image": image,
            "ground_truth": ground_truth,
        }
    raise ValueError(f"Unsupported trajectory entry type: {type(entry)}")



def _select_records(records: List[Any], start: int, end: Optional[int], indices: Optional[List[int]], limit: Optional[int]) -> List[Any]:
    if indices:
        selected = [records[idx] for idx in indices]
    else:
        selected = records[start:end]
    if limit is not None:
        selected = selected[:limit]
    return selected



def main():
    parser = argparse.ArgumentParser(description="Run the OpenThinkIMG step judge on a trajectory dataset.")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge_model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--validate_output_only", action="store_true")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list of trajectories.")

    indices = [int(item) for item in args.indices.split(",")] if args.indices else None
    selected_records = _select_records(records, args.start, args.end, indices, args.limit)

    judge = None
    if not args.validate_output_only:
        judge = TrajectoryTurnJudge(
            model_name=args.judge_model,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

    outputs = []
    for idx, entry in enumerate(selected_records):
        record = _extract_record(entry, fallback_id=f"traj_{idx}")
        turns = build_turn_records_from_messages(record["messages"])
        question = _extract_question(record["messages"])
        ground_truth = record["ground_truth"]
        if args.validate_output_only:
            judge_output = entry.get("judge_output") if isinstance(entry, dict) else None
            if judge_output is None:
                raise ValueError("validate_output_only requires each record to contain judge_output.")
            validated = validate_turn_scores(judge_output, expected_turns=len(turns))
        else:
            validated = judge.judge_trajectory(
                question=question,
                ground_truth_answer=ground_truth,
                turns=turns,
                image=record["image"],
            )
        outputs.append(
            {
                "id": record["id"],
                "judge_output": validated,
                "is_valid": True,
            }
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
