#!/usr/bin/env python3
"""
挑战杯三场景统一任务入口。

推荐运行方式：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
"""

import argparse
import os
import sys


SCENE_CONFIGS = {
    "scene1": {
        "node_name": "challenge_task_scene1",
        "title": "场景一：包裹称重与摆放",
    },
    "scene2": {
        "node_name": "challenge_task_scene2",
        "title": "场景二：分拣归档",
    },
    "scene3": {
        "node_name": "challenge_task_scene3",
        "title": "场景三：SMT 料盘出库",
    },
}


def _load_launcher():
    try:
        import rospkg
        sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
    except Exception:
        sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "challenge_cup_simulator", "utils")
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher
    return ChallengeSimLauncher


def _get_src_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


def _run_scene3():
    import rospy
    src_dir = _get_src_dir()
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from scene3_controller import Scene3TaskController
    controller = Scene3TaskController()
    success = controller.run()
    if success:
        rospy.loginfo("Scene3 task completed successfully!")
    else:
        rospy.logerr("Scene3 task FAILED.")


def run_scene(scene, seed, node_name=None, timeout=120,
              time_limit=None, timer_gui=True):
    if scene not in SCENE_CONFIGS:
        raise ValueError("unknown scene: {}".format(scene))

    config = SCENE_CONFIGS[scene]
    ChallengeSimLauncher = _load_launcher()

    launcher = ChallengeSimLauncher(
        scene=scene,
        seed=seed,
        match_time_limit=time_limit,
        timer_gui=timer_gui,
    )
    launcher.start(node_name=node_name or config["node_name"], timeout=timeout)

    import rospy

    rospy.loginfo("=== %s任务启动 ===", config["title"])

    rospy.sleep(1.0)
    rospy.loginfo("场景实例已初始化。")

    if scene == "scene3":
        _run_scene3()
    else:
        rospy.loginfo("Scene %s: task logic not yet implemented, waiting...", scene)
        rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="挑战杯三场景统一任务入口")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="scene1",
                        help="要启动的比赛场景")
    parser.add_argument("--seed", type=int, default=0,
                        help="场景种子；正式评测 seed 由组委会指定")
    parser.add_argument("--node-name", default=None,
                        help="ROS 节点名；默认按 scene 自动设置")
    parser.add_argument("--timeout", type=int, default=120,
                        help="等待仿真就绪的超时时间，单位秒")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="比赛时长，单位秒；默认读取 CHALLENGE_TIME_LIMIT，未设置则不限时")
    parser.add_argument("--no-timer-gui", action="store_true",
                        help="不弹出计时器窗口，仅保留后台计时日志")
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
    )


if __name__ == "__main__":
    main()
