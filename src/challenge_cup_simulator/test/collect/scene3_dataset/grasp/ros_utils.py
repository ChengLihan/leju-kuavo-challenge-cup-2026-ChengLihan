"""
grasp/ros_utils.py
"""
import os, signal, subprocess, sys, time, xmlrpc.client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", ".."))
CHALLENGE_TASK_SCRIPT = os.path.join(REPO_ROOT, "src", "challenge_cup_simulator", "test", "collect_scene2_dataset", "challenge_task.py")
_ros_inited = False

def init_ros(name="scene3_grasp"):
    global _ros_inited
    if _ros_inited: return
    import rospy; rospy.init_node(name, disable_signals=True); _ros_inited = True

def start_sim(seed):
    cmd = [sys.executable, CHALLENGE_TASK_SCRIPT, "--scene", "scene3", "--seed", str(seed), "--no-timer-gui"]
    return subprocess.Popen(cmd, env=os.environ.copy(), preexec_fn=os.setsid if hasattr(os, "setsid") else None), cmd

def stop_sim(proc, timeout=10.0):
    if proc is None or proc.poll() is not None: return
    try: os.killpg(os.getpgid(proc.pid), signal.SIGINT); proc.wait(timeout=timeout)
    except Exception:
        try: proc.kill()
        except Exception: pass

def wait_roscore(proc, timeout=120.0):
    uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    start = time.time()
    while time.time() - start < timeout:
        if proc is not None and proc.poll() is not None: raise RuntimeError(f"sim exited code={proc.returncode}")
        try:
            if xmlrpc.client.ServerProxy(uri).getSystemState("/grasp_check")[0] == 1: return
        except Exception: pass
        time.sleep(0.3)
    raise RuntimeError(f"roscore timeout {timeout}s")

def wait_topics(topics, timeout=20.0):
    import rospy
    required = [t for t in topics if t]; missing = list(required)
    start = time.time()
    while time.time() - start < timeout and not rospy.is_shutdown():
        pub = {n for n,_ in rospy.get_published_topics()}
        missing = [t for t in required if t not in pub]
        if not missing: return
        time.sleep(0.5)
    if missing: raise RuntimeError("missing topics: " + ",".join(missing))

def wait_publisher(pub, timeout=10.0):
    import rospy
    start = time.time()
    while pub.get_num_connections() == 0 and time.time() - start < timeout and not rospy.is_shutdown():
        rospy.sleep(0.2)
    if pub.get_num_connections() == 0: raise RuntimeError(f"no sub for {pub.name}")
