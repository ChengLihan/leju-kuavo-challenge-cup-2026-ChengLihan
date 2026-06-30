#!/usr/bin/env python3
"""
Export trained PPO policies to ONNX for deployment inference.

Exports both the ResidualActor and GripperActor networks as separate ONNX files.
The ONNX models can be loaded in challenge_task.py via onnxruntime for
fast inference without PyTorch dependency.

Usage:
    # Export from final checkpoint
    python3 rl/export_onnx.py rl/models/final_policy.pt

    # Export from intermediate checkpoint
    python3 rl/export_onnx.py rl/models/checkpoint_500000.pt --output rl/models/
"""

import os
import sys
import argparse
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from rl.policy_nets import ResidualActor, GripperActor


def export_residual_actor(
    checkpoint_path: str,
    output_path: str,
    residual_obs_dim: int = 38,
    device: str = "cpu",
):
    """Export ResidualActor to ONNX."""
    actor = ResidualActor(obs_dim=residual_obs_dim)
    actor.to(device)
    actor.eval()

    # Load weights from checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    policy_state = ckpt["policy"]

    # Extract residual_actor weights (prefixed with "residual_actor.")
    residual_state = {}
    for key, value in policy_state.items():
        if key.startswith("residual_actor."):
            residual_state[key[len("residual_actor."):]] = value
    actor.load_state_dict(residual_state)

    # Create dummy input
    dummy_input = torch.randn(1, residual_obs_dim, device=device)

    # Export
    torch.onnx.export(
        actor,
        dummy_input,
        output_path,
        input_names=["observation"],
        output_names=["action_mean"],
        opset_version=13,
        dynamic_axes={
            "observation": {0: "batch"},
            "action_mean": {0: "batch"},
        },
    )
    print(f"Exported ResidualActor to {output_path}")


def export_gripper_actor(
    checkpoint_path: str,
    output_path: str,
    gripper_obs_dim: int = 6,
    device: str = "cpu",
):
    """Export GripperActor to ONNX."""
    actor = GripperActor(obs_dim=gripper_obs_dim)
    actor.to(device)
    actor.eval()

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    policy_state = ckpt["policy"]

    # Extract gripper_actor weights
    gripper_state = {}
    for key, value in policy_state.items():
        if key.startswith("gripper_actor."):
            gripper_state[key[len("gripper_actor."):]] = value
    actor.load_state_dict(gripper_state)

    # Create dummy input
    dummy_input = torch.randn(1, gripper_obs_dim, device=device)

    # Export
    torch.onnx.export(
        actor,
        dummy_input,
        output_path,
        input_names=["observation"],
        output_names=["action_mean"],
        opset_version=13,
        dynamic_axes={
            "observation": {0: "batch"},
            "action_mean": {0: "batch"},
        },
    )
    print(f"Exported GripperActor to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export RL policies to ONNX")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: same as checkpoint)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.dirname(args.checkpoint)

    os.makedirs(args.output, exist_ok=True)

    residual_path = os.path.join(args.output, "residual_policy.onnx")
    gripper_path = os.path.join(args.output, "gripper_policy.onnx")

    export_residual_actor(args.checkpoint, residual_path, device=args.device)
    export_gripper_actor(args.checkpoint, gripper_path, device=args.device)

    print(f"\nDone! Models saved to {args.output}/")
    print(f"  residual_policy.onnx — Residual RL arm correction (±5°)")
    print(f"  gripper_policy.onnx  — Adaptive gripper close strategy")
    print(f"\nDeploy: copy .onnx files to rl/models/ in the challenge_task package")


if __name__ == "__main__":
    main()
