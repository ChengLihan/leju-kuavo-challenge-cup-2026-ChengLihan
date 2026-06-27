"""
grasp/ros_utils.py — ROS utilities (copied from working scene3_rosbag_utils logic).
"""
import os
import signal
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", ".."))
CHALLENGE_TASK_SCRIPT = os.path.join(
    REPO_ROOT, "src", "challenge_cup_simulator", "test",
    "collect_scene2_dataset", "challenge_task.py")

_ros_inited = False


def init_ros(name="scene3_grasp"):
    global _ros_inited
    if _ros_inited:
        return
    import rospy
    rospy.init_node(name, disable_signals=True)
    _ros_inited = True


def start_sim(seed, headless=False):
    cmd = [sys.executable, CHALLENGE_TASK_SCRIPT, "--scene", "scene3",
           "--seed", str(seed), "--no-timer-gui"]
    env = os.environ.copy()
    if headless:
        env["CHALLENGE_HEADLESS"] = "1"
    proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid if hasattr(os, "setsid") else None)
    return proc, cmd


def stop_sim(proc, timeout=10.0):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def wait_roscore(proc, timeout=120.0):
    """Wait for the ROS master to be available using XML-RPC (no init_node needed)."""
    import xmlrpc.client

    master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    start = time.time()
    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"sim process exited with code {proc.returncode}")
        try:
            master = xmlrpc.client.ServerProxy(master_uri)
            code, _, _ = master.getSystemState("/grasp_check")
            if code == 1:
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"roscore not ready within {timeout:.1f}s")


def wait_topics(topics, timeout=20.0):
    import rospy
    required = [t for t in topics if t]
    missing = list(required)
    start = time.time()
    while time.time() - start < timeout and not rospy.is_shutdown():
        published = {name for name, _ in rospy.get_published_topics()}
        missing = [t for t in required if t not in published]
        if not missing:
            return
        time.sleep(0.5)
    if missing:
        raise RuntimeError("missing topics: " + ",".join(missing))


def wait_publisher(pub, timeout=10.0):
    import rospy
    start = time.time()
    while pub.get_num_connections() == 0 and time.time() - start < timeout and not rospy.is_shutdown():
        rospy.sleep(0.2)
    if pub.get_num_connections() == 0:
        raise RuntimeError(f"no subscriber for {pub.name}")
