#!/usr/bin/env python3
"""Scene3 upper-tray grasp.  Flow: sim → nav → safe_home → pregrasp → close → done."""
import argparse, os, sys, traceback, yaml
from .grasp_expert import GraspExpert
from .navigation import ShelfNavigator
from .ros_utils import init_ros, start_sim, stop_sim, wait_roscore, wait_topics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main(argv=None):
    p = argparse.ArgumentParser(description="Scene3 grasp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-existing-sim", action="store_true")
    p.add_argument("--no-navigation", action="store_true")
    args = p.parse_args(argv)

    params_path = os.path.join(SCRIPT_DIR, "configs", "grasp_params.yaml")
    with open(params_path) as f: params = yaml.safe_load(f) or {}
    nav = params.get("navigation", {})
    dist, speed = nav.get("approach_distance", 1.3), nav.get("speed", 0.15)

    sim_proc = None
    try:
        if not args.use_existing_sim:
            sim_proc, _ = start_sim(args.seed)
            print("[1/6] launched sim"); wait_roscore(sim_proc)
        else: print("[1/6] using existing sim")

        init_ros("scene3_grasp")
        wait_topics(["/sensors_data_raw", "/leju_claw_state"], 20.0)
        print("[2/6] ROS ready")

        if not args.no_navigation:
            print(f"[3/6] navigating {dist}m ..."); ShelfNavigator(dist, speed).approach()
        else: print("[3/6] nav skipped")

        expert = GraspExpert(params_path=params_path); expert.setup(); expert.prepare()
        print("[4/6] arm ready")

        expert.run_pregrasp(); print("[5/6] pregrasp complete")
        expert.close_gripper(); print("[6/6] gripper closed — done")

        import rospy; rospy.sleep(2.0)
        print("\n✓ grasp finished"); expert.safe_stop(); expert.shutdown()
    except Exception as e:
        print(f"\n✗ failed: {e}", file=sys.stderr); traceback.print_exc(); return 1
    finally:
        if sim_proc is not None: stop_sim(sim_proc)
    return 0

if __name__ == "__main__": raise SystemExit(main())
