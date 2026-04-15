from __future__ import annotations

from r1_v.open_r1.trainer.tool_tbgfn_args import parse_args
from r1_v.open_r1.trainer.tool_tbgfn_trainer import ToolTBGFNTrainer


def main():
    args = parse_args()
    trainer = ToolTBGFNTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
