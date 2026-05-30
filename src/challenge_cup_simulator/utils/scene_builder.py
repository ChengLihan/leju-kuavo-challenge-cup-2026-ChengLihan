#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the Challenge Cup MuJoCo scene from a YAML config.

This mirrors the CRAIC simulator workflow while keeping the Challenge Cup
scene intentionally small: table, markers, sorting box, and parcels.
"""

import argparse
import copy
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("PyYAML is required to generate the scene") from exc


class SceneBuilder:
    def __init__(self, robot_xml="biped_s52.xml", robot_version="52"):
        self.robot_xml = robot_xml
        self.robot_version = str(robot_version)
        self.robot_model_dir = f"biped_s{self.robot_version}"
        self.config = {}
        self.root = None
        self.asset = None
        self.worldbody = None
        self.bodies = {}
        self.config_dir = None

    def load_config(self, config_file):
        config_path = Path(config_file).resolve()
        self.config_dir = config_path.parent
        self.config = self._load_config_file(config_path)
        self.robot_xml = self.config.get("robot_xml", self.robot_xml)
        return self

    def _load_config_file(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        parent = config.pop("extends", None)
        if parent:
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = config_path.parent / parent_path
            base_config = self._load_config_file(parent_path.resolve())
            config = self._deep_merge(base_config, config)

        self._apply_object_overrides(config)
        return config

    def build(self, seed=None):
        config = copy.deepcopy(self.config)
        if seed is not None:
            self._apply_randomization(config, seed)
        self.config = config

        self._init_xml()
        self._add_materials()
        self._add_ground()
        self._add_lights()
        self._add_objects()
        return self

    def save(self, output_file):
        if self.root is None:
            raise RuntimeError("call build() before save()")
        xml_bytes = ET.tostring(self.root, encoding="utf-8")
        pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
        lines = [line for line in pretty.splitlines() if line.strip()]
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Scene saved to: {output_file}")
        return self

    def _init_xml(self):
        self.root = ET.Element("mujoco", model=self.config.get("scene_name", "Challenge Cup Scene"))
        ET.SubElement(self.root, "include", file=self.robot_xml)
        ET.SubElement(self.root, "compiler", angle="radian", autolimits="true")

        camera = self.config.get("camera", {})
        ET.SubElement(
            self.root,
            "statistic",
            center=str(camera.get("center", "0.6 0 1.0")),
            extent=str(camera.get("extent", "2.5")),
        )

        visual_cfg = self.config.get("visual", {})
        visual = ET.SubElement(self.root, "visual")
        headlight = visual_cfg.get("headlight", {})
        ET.SubElement(
            visual,
            "headlight",
            diffuse=str(headlight.get("diffuse", "0.5 0.5 0.5")),
            ambient=str(headlight.get("ambient", "0.25 0.25 0.25")),
            specular=str(headlight.get("specular", "0 0 0")),
        )
        ET.SubElement(visual, "rgba", haze=str(visual_cfg.get("haze", "0.12 0.16 0.20 1")))
        ET.SubElement(
            visual,
            "global",
            azimuth=str(visual_cfg.get("azimuth", 140)),
            elevation=str(visual_cfg.get("elevation", -18)),
        )

        self.asset = ET.SubElement(self.root, "asset")
        ET.SubElement(
            self.asset,
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1="0.35 0.45 0.55",
            rgb2="0.75 0.80 0.85",
            width="128",
            height="128",
        )
        self.worldbody = ET.SubElement(self.root, "worldbody")

    def _add_materials(self):
        for name, props in self.config.get("materials", {}).items():
            props = dict(props)
            texture_cfg = props.pop("texture", None)
            if texture_cfg:
                texture_name = f"{name}_tex"
                texture_attrib = {"name": texture_name}
                texture_attrib.update({k: str(v) for k, v in texture_cfg.items()})
                ET.SubElement(self.asset, "texture", **texture_attrib)
                props["texture"] = texture_name
            attrib = {"name": name}
            attrib.update({k: str(v) for k, v in props.items()})
            ET.SubElement(self.asset, "material", **attrib)

    def _add_ground(self):
        ground = self.config.get("ground", {})
        if not ground.get("enabled", True):
            return
        attrib = {
            "name": str(ground.get("name", "floor")),
            "type": "plane",
            "size": str(ground.get("size", "8 8 0.05")),
        }
        for key in ("material", "rgba", "condim", "friction"):
            if key in ground:
                attrib[key] = str(ground[key])
        ET.SubElement(self.worldbody, "geom", **attrib)

    def _add_lights(self):
        for i, light in enumerate(self.config.get("lights", [])):
            attrib = {
                "name": str(light.get("name", f"light_{i}")),
                "pos": str(light.get("pos", "0 0 5")),
                "dir": str(light.get("dir", "0 0 -1")),
                "directional": str(light.get("directional", "true")),
            }
            for key in ("diffuse", "specular", "ambient", "castshadow"):
                if key in light:
                    attrib[key] = str(light[key])
            ET.SubElement(self.worldbody, "light", **attrib)

    def _add_objects(self):
        for obj in self.config.get("objects", []):
            obj_type = obj.get("type")
            name = obj.get("name", "object")
            if obj_type == "table":
                self._add_table(obj, name)
            elif obj_type == "marker":
                self._add_marker(obj, name)
            elif obj_type == "open_box":
                self._add_open_box(obj, name)
            elif obj_type == "tapered_open_box":
                self._add_tapered_open_box(obj, name)
            elif obj_type == "parcel":
                self._add_parcel(obj, name)
            elif obj_type == "mesh_part":
                self._add_mesh_part(obj, name)
            elif obj_type == "mujoco_model":
                self._add_mujoco_model(obj, name)
            elif obj_type == "box":
                self._add_box(obj, name)
            else:
                print(f"Warning: unsupported object type '{obj_type}' for '{name}'")

    def _parent(self, obj):
        parent_name = obj.get("parent")
        if parent_name:
            return self.bodies.get(parent_name, self.worldbody)
        return self.worldbody

    def _add_table(self, obj, name):
        body = ET.SubElement(self.worldbody, "body", name=name, pos=str(obj.get("pos", "0 0 0")))
        self.bodies[name] = body
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_top",
            type="box",
            size=str(obj.get("top_size", "0.70 0.40 0.015")),
            pos=str(obj.get("top_pos", "0 0 0.815")),
            material=str(obj.get("top_material", "table_top_mat")),
            mass=str(obj.get("top_mass", 10)),
            friction=str(obj.get("friction", "1.2 0.2 0.01")),
        )
        leg_radius = float(obj.get("leg_radius", 0.03))
        leg_height = float(obj.get("leg_height", 0.40))
        leg_z = float(obj.get("leg_z", 0.40))
        for i, xy in enumerate(obj.get("leg_offsets", [])):
            x, y = self._float_list(xy, 2)
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_leg_{i}",
                type="cylinder",
                size=f"{leg_radius:g} {leg_height:g}",
                pos=f"{x:g} {y:g} {leg_z:g}",
                material=str(obj.get("leg_material", "table_leg_mat")),
                mass=str(obj.get("leg_mass", 2)),
            )

    def _add_marker(self, obj, name):
        ET.SubElement(
            self._parent(obj),
            "geom",
            name=name,
            type="box",
            size=str(obj.get("size", "0.10 0.10 0.002")),
            pos=str(obj.get("pos", "0 0 0")),
            material=str(obj.get("material", "")),
            contype="0",
            conaffinity="0",
        )

    def _add_open_box(self, obj, name):
        attrib = {"name": name, "pos": str(obj.get("pos", "0 0 0"))}
        if "euler" in obj:
            attrib["euler"] = str(obj["euler"])
        body = ET.SubElement(self._parent(obj), "body", **attrib)
        self.bodies[name] = body
        length, width, height = self._float_list(obj.get("inner_size", "0.40 0.30 0.30"), 3)
        wall = float(obj.get("wall_thickness", 0.024))
        floor = float(obj.get("floor_thickness", 0.016))
        half_x = length / 2.0
        half_y = width / 2.0
        half_z = height / 2.0
        wall_half = wall / 2.0
        floor_half = floor / 2.0
        floor_mat = str(obj.get("floor_material", "drop_box_mat"))
        wall_mat = str(obj.get("wall_material", "drop_box_edge_mat"))

        ET.SubElement(
            body,
            "geom",
            name=f"{name}_floor",
            type="box",
            size=f"{half_x:g} {half_y:g} {floor_half:g}",
            pos=f"0 0 {floor_half:g}",
            material=floor_mat,
            contype="0",
            conaffinity="0",
        )
        wall_specs = [
            ("front_wall", half_x, wall_half, half_z, 0, -(half_y + wall_half), half_z),
            ("back_wall", half_x, wall_half, half_z, 0, half_y + wall_half, half_z),
            ("left_wall", wall_half, half_y, half_z, -(half_x + wall_half), 0, half_z),
            ("right_wall", wall_half, half_y, half_z, half_x + wall_half, 0, half_z),
        ]
        for suffix, sx, sy, sz, px, py, pz in wall_specs:
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_{suffix}",
                type="box",
                size=f"{sx:g} {sy:g} {sz:g}",
                pos=f"{px:g} {py:g} {pz:g}",
                material=wall_mat,
                contype="1",
                conaffinity="1",
            )

    def _add_tapered_open_box(self, obj, name):
        attrib = {"name": name, "pos": str(obj.get("pos", "0 0 0"))}
        if "euler" in obj:
            attrib["euler"] = str(obj["euler"])
        body = ET.SubElement(self._parent(obj), "body", **attrib)
        self.bodies[name] = body

        length, width, height = self._float_list(obj.get("inner_size", "0.24 0.22 0.10"), 3)
        wall = float(obj.get("wall_thickness", 0.012))
        floor = float(obj.get("floor_thickness", 0.010))
        profile = str(obj.get("profile", "flared_lip"))
        if profile == "front_low_back_high":
            self._add_front_low_back_high_box(body, obj, name, length, width, height, wall, floor)
            return

        base_height = float(obj.get("base_height", height * 0.65))
        lip_height = max(0.0, height - base_height)
        lip_tilt = float(obj.get("lip_tilt", obj.get("wall_tilt", 0.28)))
        half_x = length / 2.0
        half_y = width / 2.0
        wall_half = wall / 2.0
        floor_half = floor / 2.0
        base_half_z = base_height / 2.0
        lip_half_z = lip_height / 2.0
        lip_center_offset = math.tan(lip_tilt) * lip_half_z / 2.0
        floor_mat = str(obj.get("floor_material", obj.get("material", "drop_box_mat")))
        wall_mat = str(obj.get("wall_material", obj.get("material", "drop_box_edge_mat")))

        ET.SubElement(
            body,
            "geom",
            name=f"{name}_floor",
            type="box",
            size=f"{half_x:g} {half_y:g} {floor_half:g}",
            pos=f"0 0 {floor_half:g}",
            material=floor_mat,
            contype="1",
            conaffinity="1",
        )

        wall_specs = [
            (
                "front_wall",
                f"{half_x:g} {wall_half:g} {base_half_z:g}",
                f"0 {-half_y - wall_half:g} {floor + base_half_z:g}",
                None,
            ),
            (
                "back_wall",
                f"{half_x:g} {wall_half:g} {base_half_z:g}",
                f"0 {half_y + wall_half:g} {floor + base_half_z:g}",
                None,
            ),
            (
                "left_wall",
                f"{wall_half:g} {half_y:g} {base_half_z:g}",
                f"{-half_x - wall_half:g} 0 {floor + base_half_z:g}",
                None,
            ),
            (
                "right_wall",
                f"{wall_half:g} {half_y:g} {base_half_z:g}",
                f"{half_x + wall_half:g} 0 {floor + base_half_z:g}",
                None,
            ),
        ]
        if lip_height > 0:
            lip_z = floor + base_height + lip_half_z
            wall_specs.extend(
                [
                    (
                        "front_lip",
                        f"{half_x:g} {wall_half:g} {lip_half_z:g}",
                        f"0 {-half_y - wall_half - lip_center_offset:g} {lip_z:g}",
                        f"{-lip_tilt:g} 0 0",
                    ),
                    (
                        "back_lip",
                        f"{half_x:g} {wall_half:g} {lip_half_z:g}",
                        f"0 {half_y + wall_half + lip_center_offset:g} {lip_z:g}",
                        f"{lip_tilt:g} 0 0",
                    ),
                    (
                        "left_lip",
                        f"{wall_half:g} {half_y:g} {lip_half_z:g}",
                        f"{-half_x - wall_half - lip_center_offset:g} 0 {lip_z:g}",
                        f"0 {lip_tilt:g} 0",
                    ),
                    (
                        "right_lip",
                        f"{wall_half:g} {half_y:g} {lip_half_z:g}",
                        f"{half_x + wall_half + lip_center_offset:g} 0 {lip_z:g}",
                        f"0 {-lip_tilt:g} 0",
                    ),
                ]
            )
        for suffix, size, pos, euler in wall_specs:
            attrib = {
                "name": f"{name}_{suffix}",
                "type": "box",
                "size": size,
                "pos": pos,
                "material": wall_mat,
                "contype": "1",
                "conaffinity": "1",
            }
            if euler:
                attrib["euler"] = euler
            ET.SubElement(body, "geom", **attrib)

    def _add_front_low_back_high_box(self, body, obj, name, length, width, height, wall, floor):
        front_height = float(obj.get("front_height", height * 0.45))
        back_height = float(obj.get("back_height", height))
        rear_length_ratio = float(obj.get("rear_high_length_ratio", 0.45))
        rear_length_ratio = min(1.0, max(0.0, rear_length_ratio))
        half_x = length / 2.0
        half_y = width / 2.0
        wall_half = wall / 2.0
        floor_half = floor / 2.0
        front_half_z = front_height / 2.0
        back_half_z = back_height / 2.0
        rear_len = length * rear_length_ratio
        rear_half_x = rear_len / 2.0
        rear_center_x = half_x - rear_half_x
        front_mat = str(obj.get("front_material", obj.get("wall_material", obj.get("material", "drop_box_edge_mat"))))
        wall_mat = str(obj.get("wall_material", obj.get("material", "drop_box_edge_mat")))
        floor_mat = str(obj.get("floor_material", obj.get("material", "drop_box_mat")))

        ET.SubElement(
            body,
            "geom",
            name=f"{name}_floor",
            type="box",
            size=f"{half_x:g} {half_y:g} {floor_half:g}",
            pos=f"0 0 {floor_half:g}",
            material=floor_mat,
            contype="1",
            conaffinity="1",
        )

        wall_specs = [
            (
                "front_low_wall",
                f"{wall_half:g} {half_y:g} {front_half_z:g}",
                f"{-half_x - wall_half:g} 0 {floor + front_half_z:g}",
                front_mat,
            ),
            (
                "back_high_wall",
                f"{wall_half:g} {half_y:g} {back_half_z:g}",
                f"{half_x + wall_half:g} 0 {floor + back_half_z:g}",
                wall_mat,
            ),
            (
                "left_low_side",
                f"{half_x:g} {wall_half:g} {front_half_z:g}",
                f"0 {half_y + wall_half:g} {floor + front_half_z:g}",
                front_mat,
            ),
            (
                "right_low_side",
                f"{half_x:g} {wall_half:g} {front_half_z:g}",
                f"0 {-half_y - wall_half:g} {floor + front_half_z:g}",
                front_mat,
            ),
            (
                "left_rear_high_side",
                f"{rear_half_x:g} {wall_half:g} {back_half_z:g}",
                f"{rear_center_x:g} {half_y + wall_half:g} {floor + back_half_z:g}",
                wall_mat,
            ),
            (
                "right_rear_high_side",
                f"{rear_half_x:g} {wall_half:g} {back_half_z:g}",
                f"{rear_center_x:g} {-half_y - wall_half:g} {floor + back_half_z:g}",
                wall_mat,
            ),
        ]
        for suffix, size, pos, material in wall_specs:
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_{suffix}",
                type="box",
                size=size,
                pos=pos,
                material=material,
                contype="1",
                conaffinity="1",
            )

    def _add_parcel(self, obj, name):
        attrib = {"name": name, "pos": str(obj.get("pos", "0 0 0"))}
        if "euler" in obj:
            attrib["euler"] = str(obj["euler"])
        if "quat" in obj:
            attrib["quat"] = str(obj["quat"])
        body = ET.SubElement(self.worldbody, "body", **attrib)
        ET.SubElement(body, "freejoint", name=name)
        size = self._float_list(obj.get("size", "0.055 0.040 0.030"), 3)
        geom_type = "ellipsoid" if obj.get("style") == "courier_bag" else "box"
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_geom",
            type=geom_type,
            size=f"{size[0]:g} {size[1]:g} {size[2]:g}",
            material=str(obj.get("material", "parcel_brown_mat")),
            mass=str(obj.get("mass", 0.15)),
            condim=str(obj.get("condim", 6)),
            friction=str(obj.get("friction", "3.0 1.0 0.02")),
            solref=str(obj.get("solref", "0.01 1")),
            solimp=str(obj.get("solimp", "0.95 0.99 0.001")),
        )
        if obj.get("style") == "courier":
            self._add_courier_parcel_visuals(body, name, size, obj)
        elif obj.get("style") == "courier_bag":
            self._add_courier_bag_visuals(body, name, size, obj)

    def _add_visual_box(self, parent, name, size, pos, material):
        ET.SubElement(
            parent,
            "geom",
            name=name,
            type="box",
            size=size,
            pos=pos,
            material=material,
            mass="0.0001",
            contype="0",
            conaffinity="0",
        )

    def _add_courier_parcel_visuals(self, body, name, size, obj):
        hx, hy, hz = size
        z = hz + 0.0015
        tape_mat = str(obj.get("tape_material", "parcel_tape_mat"))
        label_mat = str(obj.get("label_material", "parcel_label_mat"))
        line_mat = str(obj.get("label_line_material", "parcel_label_line_mat"))

        self._add_visual_box(
            body,
            f"{name}_tape_long",
            f"{hx * 0.92:g} 0.006 0.001",
            f"0 0 {z:g}",
            tape_mat,
        )
        self._add_visual_box(
            body,
            f"{name}_tape_cross",
            f"0.006 {hy * 0.92:g} 0.001",
            f"0 0 {z + 0.001:g}",
            tape_mat,
        )
        self._add_visual_box(
            body,
            f"{name}_label",
            f"{hx * 0.34:g} {hy * 0.22:g} 0.001",
            f"{hx * 0.22:g} {hy * 0.28:g} {z + 0.002:g}",
            label_mat,
        )
        self._add_visual_box(
            body,
            f"{name}_label_line_1",
            f"{hx * 0.22:g} 0.002 0.001",
            f"{hx * 0.22:g} {hy * 0.31:g} {z + 0.003:g}",
            line_mat,
        )
        self._add_visual_box(
            body,
            f"{name}_label_line_2",
            f"{hx * 0.22:g} 0.002 0.001",
            f"{hx * 0.22:g} {hy * 0.24:g} {z + 0.003:g}",
            line_mat,
        )

    def _add_courier_bag_visuals(self, body, name, size, obj):
        hx, hy, hz = size
        front_y = -hy - 0.002
        tape_mat = str(obj.get("tape_material", "parcel_tape_mat"))
        label_mat = str(obj.get("label_material", "parcel_label_mat"))
        line_mat = str(obj.get("label_line_material", "parcel_label_line_mat"))

        # A thin base keeps the upright bag visually grounded without changing contacts.
        self._add_visual_box(
            body,
            f"{name}_bottom_fold",
            f"{hx * 0.82:g} {hy * 0.80:g} 0.002",
            f"0 0 {-hz * 0.88:g}",
            tape_mat,
        )
        self._add_visual_box(
            body,
            f"{name}_seal_strip",
            f"{hx * 0.74:g} 0.0015 {hz * 0.055:g}",
            f"0 {front_y:g} {hz * 0.58:g}",
            tape_mat,
        )
        self._add_visual_box(
            body,
            f"{name}_label",
            f"{hx * 0.36:g} 0.0015 {hz * 0.24:g}",
            f"{hx * 0.18:g} {front_y - 0.001:g} {hz * 0.06:g}",
            label_mat,
        )
        self._add_visual_box(
            body,
            f"{name}_label_line_1",
            f"{hx * 0.24:g} 0.0016 {hz * 0.012:g}",
            f"{hx * 0.18:g} {front_y - 0.002:g} {hz * 0.12:g}",
            line_mat,
        )
        self._add_visual_box(
            body,
            f"{name}_label_line_2",
            f"{hx * 0.24:g} 0.0016 {hz * 0.012:g}",
            f"{hx * 0.18:g} {front_y - 0.002:g} {hz * 0.02:g}",
            line_mat,
        )
        for i, x in enumerate((-0.42, -0.18, 0.42)):
            self._add_visual_box(
                body,
                f"{name}_crease_{i}",
                f"{hx * 0.018:g} 0.0014 {hz * 0.70:g}",
                f"{hx * x:g} {front_y - 0.001:g} 0",
                line_mat,
            )

    def _add_mesh_part(self, obj, name):
        visual_meshes = obj.get("visual_meshes")
        if visual_meshes:
            for visual in visual_meshes:
                mesh_name = f"{name}_{visual.get('suffix', 'visual')}_mesh"
                mesh_attrib = {
                    "name": mesh_name,
                    "file": str(visual["mesh_file"]),
                }
                if "mesh_scale" in visual:
                    mesh_attrib["scale"] = str(visual["mesh_scale"])
                elif "mesh_scale" in obj:
                    mesh_attrib["scale"] = str(obj["mesh_scale"])
                ET.SubElement(self.asset, "mesh", **mesh_attrib)
        else:
            mesh_name = f"{name}_mesh"
            mesh_attrib = {
                "name": mesh_name,
                "file": str(obj["mesh_file"]),
            }
            if "mesh_scale" in obj:
                mesh_attrib["scale"] = str(obj["mesh_scale"])
            ET.SubElement(self.asset, "mesh", **mesh_attrib)

        body_attrib = {"name": name, "pos": str(obj.get("pos", "0 0 0"))}
        if "euler" in obj:
            body_attrib["euler"] = str(obj["euler"])
        if "quat" in obj:
            body_attrib["quat"] = str(obj["quat"])
        body = ET.SubElement(self.worldbody, "body", **body_attrib)
        ET.SubElement(body, "freejoint", name=name)

        if visual_meshes:
            for visual in visual_meshes:
                suffix = str(visual.get("suffix", "visual"))
                visual_attrib = {
                    "name": f"{name}_{suffix}_visual",
                    "type": "mesh",
                    "mesh": f"{name}_{suffix}_mesh",
                    "contype": "0",
                    "conaffinity": "0",
                    "group": str(visual.get("visual_group", obj.get("visual_group", 2))),
                }
                if "material" in visual:
                    visual_attrib["material"] = str(visual["material"])
                elif "material" in obj:
                    visual_attrib["material"] = str(obj["material"])
                if "rgba" in visual:
                    visual_attrib["rgba"] = str(visual["rgba"])
                elif "rgba" in obj:
                    visual_attrib["rgba"] = str(obj["rgba"])
                ET.SubElement(body, "geom", **visual_attrib)
        else:
            visual_attrib = {
                "name": f"{name}_visual",
                "type": "mesh",
                "mesh": mesh_name,
                "contype": "0",
                "conaffinity": "0",
                "group": str(obj.get("visual_group", 2)),
            }
            if "material" in obj:
                visual_attrib["material"] = str(obj["material"])
            if "rgba" in obj:
                visual_attrib["rgba"] = str(obj["rgba"])
            ET.SubElement(body, "geom", **visual_attrib)

        collision_geoms = obj.get("collision_geoms")
        if collision_geoms:
            for collision in collision_geoms:
                self._add_mesh_part_collision(body, obj, name, collision, mesh_name)
        else:
            self._add_mesh_part_collision(body, obj, name, None, mesh_name)

    def _add_mujoco_model(self, obj, name):
        model_file = self._resolve_config_path(obj["model_file"])
        model_root = ET.parse(model_file).getroot()
        prefix = str(obj.get("prefix", name))
        asset_path_prefix = obj.get("asset_path_prefix")
        asset_scale = obj.get("asset_scale")
        asset_map = self._clone_mujoco_assets(model_root.find("asset"), prefix, asset_path_prefix, asset_scale)

        wrapper_attrib = {"name": name, "pos": str(obj.get("pos", "0 0 0"))}
        if "euler" in obj:
            wrapper_attrib["euler"] = str(obj["euler"])
        if "quat" in obj:
            wrapper_attrib["quat"] = str(obj["quat"])
        wrapper = ET.SubElement(self.worldbody, "body", **wrapper_attrib)
        self.bodies[name] = wrapper
        if obj.get("movable", False):
            ET.SubElement(wrapper, "freejoint", name=name)

        source_world = model_root.find("worldbody")
        if source_world is None:
            raise ValueError(f"mujoco_model '{name}' has no worldbody: {model_file}")
        for child in list(source_world):
            cloned = copy.deepcopy(child)
            self._prefix_mujoco_tree(cloned, prefix, asset_map)
            wrapper.append(cloned)

    def _clone_mujoco_assets(self, source_asset, prefix, asset_path_prefix=None, asset_scale=None):
        if source_asset is None:
            return {}
        asset_map = {}
        for child in list(source_asset):
            original_name = child.get("name")
            if original_name:
                asset_map[original_name] = f"{prefix}_{original_name}"

        for child in list(source_asset):
            cloned = copy.deepcopy(child)
            original_name = cloned.get("name")
            if original_name:
                cloned.set("name", asset_map[original_name])
            file_name = cloned.get("file")
            if file_name and asset_path_prefix:
                cloned.set("file", self._join_xml_path(asset_path_prefix, file_name))
            if asset_scale and cloned.tag == "mesh":
                cloned.set("scale", str(asset_scale))
            self._rewrite_mujoco_refs(cloned, asset_map)
            self.asset.append(cloned)
        return asset_map

    def _prefix_mujoco_tree(self, root, prefix, asset_map):
        for elem in root.iter():
            original_name = elem.get("name")
            if original_name:
                elem.set("name", f"{prefix}_{original_name}")
            self._rewrite_mujoco_refs(elem, asset_map)

    @staticmethod
    def _rewrite_mujoco_refs(elem, asset_map):
        for attr in ("material", "mesh", "texture"):
            value = elem.get(attr)
            if value in asset_map:
                elem.set(attr, asset_map[value])

    def _add_mesh_part_collision(self, body, obj, name, collision, mesh_name):
        collision = collision or {}
        suffix = collision.get("suffix", "collision")
        collision_type = str(collision.get("type", obj.get("collision_type", "box")))
        collision_attrib = {
            "name": f"{name}_{suffix}",
            "type": collision_type,
            "size": str(collision.get("size", obj.get("collision_size", "0.03 0.03 0.02"))),
            "mass": str(collision.get("mass", obj.get("mass", 0.1))),
            "friction": str(collision.get("friction", obj.get("friction", "4.0 1.0 0.03"))),
            "condim": str(collision.get("condim", obj.get("condim", 6))),
            "solref": str(collision.get("solref", obj.get("solref", "0.008 1.2"))),
            "solimp": str(collision.get("solimp", obj.get("solimp", "0.95 0.99 0.001"))),
            "group": str(collision.get("group", obj.get("collision_group", 3))),
        }
        if collision_type == "mesh":
            collision_attrib.pop("size", None)
            collision_attrib["mesh"] = mesh_name

        field_map = {
            "pos": ("pos", "collision_pos"),
            "euler": ("euler", "collision_euler"),
            "quat": ("quat", "collision_quat"),
            "fromto": ("fromto", None),
            "material": ("material", "collision_material"),
            "rgba": ("rgba", "collision_rgba"),
            "contype": ("contype", "collision_contype"),
            "conaffinity": ("conaffinity", "collision_conaffinity"),
        }
        for xml_key, (collision_key, obj_key) in field_map.items():
            if collision_key in collision:
                collision_attrib[xml_key] = str(collision[collision_key])
            elif obj_key and obj_key in obj:
                collision_attrib[xml_key] = str(obj[obj_key])
        ET.SubElement(body, "geom", **collision_attrib)

    def _add_box(self, obj, name):
        body = ET.SubElement(self.worldbody, "body", name=name, pos=str(obj.get("pos", "0 0 0")))
        if obj.get("movable", False):
            ET.SubElement(body, "freejoint", name=name)
        attrib = {
            "name": f"{name}_geom",
            "type": str(obj.get("geom_type", "box")),
            "size": str(obj.get("size", "0.1 0.1 0.1")),
            "mass": str(obj.get("mass", 0.1)),
            "friction": str(obj.get("friction", "1 0.1 0.01")),
        }
        if "euler" in obj:
            attrib["euler"] = str(obj["euler"])
        if "quat" in obj:
            attrib["quat"] = str(obj["quat"])
        if "material" in obj:
            attrib["material"] = str(obj["material"])
        if "rgba" in obj:
            attrib["rgba"] = str(obj["rgba"])
        for key in ("condim", "solref", "solimp"):
            if key in obj:
                attrib[key] = str(obj[key])
        ET.SubElement(body, "geom", **attrib)

    def _resolve_config_path(self, value):
        path = Path(str(value))
        if path.is_absolute():
            return path
        if self.config_dir is None:
            return path.resolve()
        return (self.config_dir / path).resolve()

    @staticmethod
    def _join_xml_path(prefix, file_name):
        return str(Path(str(prefix)) / str(file_name)).replace(os.sep, "/")

    def _apply_randomization(self, config, seed):
        rand_cfg = config.get("randomization", {})
        shuffle_cfg = rand_cfg.get("shuffleable_parts", {})
        names = shuffle_cfg.get("names", []) if isinstance(shuffle_cfg, dict) else shuffle_cfg
        if not names or len(names) < 2:
            return
        xy_offset = float(shuffle_cfg.get("xy_offset", 0.0)) if isinstance(shuffle_cfg, dict) else 0.0
        objects = config.get("objects", [])
        by_name = {obj.get("name"): obj for obj in objects if obj.get("name")}
        positions = []
        for name in names:
            obj = by_name.get(name)
            if obj:
                x, y, _ = self._float_list(obj.get("pos", "0 0 0"), 3)
                positions.append((x, y, obj.get("euler")))
        rng = random.Random(seed)
        rng.shuffle(positions)
        for name, (x, y, euler) in zip(names, positions):
            obj = by_name.get(name)
            if not obj:
                continue
            _, _, z = self._float_list(obj.get("pos", "0 0 0"), 3)
            x += rng.uniform(-xy_offset, xy_offset)
            y += rng.uniform(-xy_offset, xy_offset)
            obj["pos"] = f"{x:.3f} {y:.3f} {z:.3f}"
            if euler is not None:
                obj["euler"] = euler

    @staticmethod
    def _deep_merge(base, override):
        result = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = SceneBuilder._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def _apply_object_overrides(config):
        overrides = config.pop("object_overrides", {})
        if not overrides:
            return
        objects = config.get("objects", [])
        by_name = {obj.get("name"): obj for obj in objects if obj.get("name")}
        for name, attrs in overrides.items():
            if name in by_name:
                by_name[name].update(attrs or {})
            else:
                new_obj = {"name": name}
                new_obj.update(attrs or {})
                objects.append(new_obj)

    @staticmethod
    def _float_list(value, expected):
        if isinstance(value, str):
            parts = value.split()
        else:
            parts = list(value)
        if len(parts) < expected:
            raise ValueError(f"expected {expected} numbers, got: {value}")
        return [float(parts[i]) for i in range(expected)]


def main():
    parser = argparse.ArgumentParser(description="Generate Challenge Cup MuJoCo scene XML")
    script_dir = Path(__file__).resolve().parent
    pkg_dir = script_dir.parent
    default_config = pkg_dir / "config" / "scenes" / "scene1.yaml"
    default_output_dir = pkg_dir / "models" / "biped_s52" / "xml"
    parser.add_argument("config", nargs="?", default=str(default_config))
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--all", action="store_true", help="generate all config/scenes/scene*.yaml files")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--robot-version", default=os.environ.get("ROBOT_VERSION", "52"))
    args = parser.parse_args()

    config_files = []
    if args.all:
        config_files = sorted((pkg_dir / "config" / "scenes").glob("scene*.yaml"))
    else:
        config_files = [Path(args.config)]

    for config_file in config_files:
        output = Path(args.output) if args.output and not args.all else default_output_dir / f"{config_file.stem}.xml"
        builder = SceneBuilder(robot_version=args.robot_version)
        builder.load_config(config_file).build(seed=args.seed).save(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"scene generation failed: {exc}", file=sys.stderr)
        raise
