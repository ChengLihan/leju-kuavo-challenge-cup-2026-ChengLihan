#!/usr/bin/env python3
"""
比赛计时器单独运行入口。

正式启动链路由 challenge_sim_launcher 自动拉起 .so 内的计时器；本脚本只用于单独查看或验证计时器。
"""

import argparse
import os
import sys


def _load_secret():
    try:
        import rospkg
        lib_dir = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "lib")
    except Exception:
        lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib"))
    sys.path.insert(0, lib_dir)
    import challenge_secret
    return challenge_secret


def main():
    parser = argparse.ArgumentParser(description="挑战杯计时器单独运行入口")
    parser.add_argument("--time-limit", type=float, default=0.0,
                        help="比赛时长，单位秒；0 表示不限时")
    parser.add_argument("--enforce", action="store_true",
                        help="到时结束目标节点；默认不启用")
    parser.add_argument("--target-node", default="",
                        help="到时结束的 ROS 节点名")
    parser.add_argument("--target-pid", type=int, default=0,
                        help="到时结束的进程 PID")
    parser.add_argument("--no-gui", action="store_true",
                        help="不弹出计时器窗口")
    args = parser.parse_args()

    secret = _load_secret()
    secret.run_match_timer(
        time_limit=args.time_limit,
        enforce=args.enforce,
        gui=not args.no_gui,
        target_node=args.target_node,
        target_pid=args.target_pid,
        eval_mode=False,
    )


if __name__ == "__main__":
    main()
