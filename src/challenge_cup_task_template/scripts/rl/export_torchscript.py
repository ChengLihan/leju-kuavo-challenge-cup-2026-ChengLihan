#!/usr/bin/env python3
"""Export trained ResidualActor to TorchScript (.pt) for ROS deployment."""
import os, sys, argparse
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from rl.policy_nets import ResidualActor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.dirname(args.checkpoint)
    os.makedirs(args.output, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    actor = ResidualActor(obs_dim=23)
    actor.load_state_dict(ckpt["actor"])
    actor.to(args.device)
    actor.eval()

    dummy = torch.randn(1, 23, device=args.device)
    traced = torch.jit.trace(actor, dummy)
    out_path = os.path.join(args.output, "residual_policy.pt")
    traced.save(out_path)

    # Verify
    loaded = torch.jit.load(out_path)
    with torch.no_grad():
        diff = (actor(dummy)[0] - loaded(dummy)[0]).abs().max().item()
    print(f"Exported: {out_path}  (max diff: {diff:.2e})")
    print("Deploy: copy to rl/models/residual_policy.pt")


if __name__ == "__main__":
    main()
