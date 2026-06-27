"""
grasp/mujoco_utils.py — MuJoCo XML/qpos parsing utilities.

Self-contained; extracted from scene3_success_checker so that the whole
grasp pipeline lives under this directory.
"""
import math
import os
import xml.etree.ElementTree as ET


def _joint_qpos_size(elem):
    if elem.tag == "freejoint":
        return 7
    if elem.tag != "joint":
        return 0
    joint_type = elem.get("type", "hinge")
    if joint_type == "free":
        return 7
    if joint_type == "ball":
        return 4
    return 1


def _iter_xml_children_with_includes(path):
    root = ET.parse(path).getroot()
    base_dir = os.path.dirname(path)
    for child in list(root):
        if child.tag == "include":
            include_path = child.get("file")
            if include_path:
                resolved = include_path
                if not os.path.isabs(resolved):
                    resolved = os.path.join(base_dir, resolved)
                yield from _iter_xml_children_with_includes(os.path.abspath(resolved))
            continue
        yield child, base_dir


def _walk_body_for_qpos(body, qpos_addr, qpos_map, body_pos_map, body_freejoint_map):
    body_name = body.get("name")
    if body_name and body.get("pos"):
        body_pos_map[body_name] = [float(v) for v in body.get("pos").split()[:3]]
    for child in list(body):
        size = _joint_qpos_size(child)
        if size:
            name = child.get("name")
            if name:
                qpos_map[name] = qpos_addr
            if child.tag == "freejoint" and body_name:
                body_freejoint_map[body_name] = qpos_addr
            qpos_addr += size
    for child in list(body):
        if child.tag == "body":
            qpos_addr = _walk_body_for_qpos(child, qpos_addr, qpos_map, body_pos_map, body_freejoint_map)
    return qpos_addr


def build_qpos_and_body_maps(xml_path):
    qpos_addr = 0
    qpos_map = {}
    body_pos_map = {}
    body_freejoint_map = {}
    for child, _base_dir in _iter_xml_children_with_includes(xml_path):
        if child.tag != "worldbody":
            continue
        for body in list(child):
            if body.tag == "body":
                qpos_addr = _walk_body_for_qpos(body, qpos_addr, qpos_map, body_pos_map, body_freejoint_map)
    return qpos_map, body_pos_map, body_freejoint_map


def pose_from_qpos(qpos, qpos_map, name):
    if name not in qpos_map:
        raise KeyError(f"freejoint '{name}' not found in XML qpos map")
    addr = qpos_map[name]
    if addr + 2 >= len(qpos):
        raise IndexError(f"qpos for '{name}' address {addr} outside qpos length {len(qpos)}")
    return [float(qpos[addr]), float(qpos[addr + 1]), float(qpos[addr + 2])]


def pose_from_body_freejoint(qpos, body_freejoint_map, name):
    if name not in body_freejoint_map:
        raise KeyError(f"freejoint body '{name}' not found in XML qpos map")
    addr = body_freejoint_map[name]
    if addr + 6 >= len(qpos):
        raise IndexError(f"qpos for body '{name}' address {addr} outside qpos length {len(qpos)}")
    xyz = [float(qpos[addr]), float(qpos[addr + 1]), float(qpos[addr + 2])]
    quat_wxyz = [float(qpos[addr + 3]), float(qpos[addr + 4]), float(qpos[addr + 5]), float(qpos[addr + 6])]
    return xyz, quat_wxyz


def yaw_from_quat_wxyz(quat):
    w, x, y, z = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def world_to_base_xyz(point_world, base_world, base_yaw):
    dx = float(point_world[0]) - float(base_world[0])
    dy = float(point_world[1]) - float(base_world[1])
    c = math.cos(-float(base_yaw))
    s = math.sin(-float(base_yaw))
    return [
        c * dx - s * dy,
        s * dx + c * dy,
        float(point_world[2]) - float(base_world[2]),
    ]
