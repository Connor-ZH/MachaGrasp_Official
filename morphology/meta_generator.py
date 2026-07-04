import argparse
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh


DUMMY_LINK_TYPE = 0
BOX_TYPE = 1
MESH_BOX_APPROX_TYPE = 1
CYLINDER_TYPE = 2
SPHERE_TYPE = 3

DEFAULT_HAND_URDFS = {
    "shadow": "shadow/shadow.urdf",
    "allegro": "allegro/allegro.urdf",
    "robotiq_3f": "robotiq_3f/robotiq_3f.urdf",
    "barrett": "barrett/barrett.urdf",
}

DEFAULT_OUTPUT_NAMES = {
    "shadow": "shadow_joint_encodings_with_link.pt",
    "allegro": "allegro_joint_encodings_with_link.pt",
    "robotiq_3f": "robotiq_3f_joint_encodings_with_link.pt",
    "barrett": "barrett_joint_encodings_with_link.pt",
}

RELEASED_HANDS = ["shadow", "allegro", "robotiq_3f", "barrett"]


def rotation_matrix_to_translation_rpy(matrix):
    xyz = matrix[:3, 3]
    rpy = rotation_matrix_to_euler_xyz(matrix[:3, :3])
    return xyz, rpy


def rotation_matrix_to_euler_xyz(rotation):
    beta = -np.arcsin(rotation[2, 0])
    alpha = np.arctan2(rotation[2, 1] / np.cos(beta), rotation[2, 2] / np.cos(beta))
    gamma = np.arctan2(rotation[1, 0] / np.cos(beta), rotation[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))


def parse_vector(value, default):
    if value is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(item) for item in value.split()], dtype=float)


def parse_origin(element):
    if element is None:
        return np.zeros(3), np.zeros(3)
    xyz = parse_vector(element.attrib.get("xyz"), [0.0, 0.0, 0.0])
    rpy = parse_vector(element.attrib.get("rpy"), [0.0, 0.0, 0.0])
    return xyz, rpy


def parse_urdf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    links = {}
    joints = []

    for link in root.findall("link"):
        collisions = []
        for collision in link.findall("collision"):
            xyz, rpy = parse_origin(collision.find("origin"))
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            collisions.append({"xyz": xyz, "rpy": rpy, "geometry": geometry})
        links[link.attrib["name"]] = {"name": link.attrib["name"], "collisions": collisions}

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        xyz, rpy = parse_origin(joint.find("origin"))
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib.get("type", "fixed"),
                "parent": parent.attrib["link"],
                "child": child.attrib["link"],
                "xyz": xyz,
                "rpy": rpy,
                "axis": parse_vector(axis.attrib.get("xyz") if axis is not None else None, [1.0, 0.0, 0.0]),
                "lower": float(limit.attrib.get("lower", 0.0)) if limit is not None else 0.0,
                "upper": float(limit.attrib.get("upper", 0.0)) if limit is not None else 0.0,
            }
        )

    return {"links": links, "joints": joints}


def get_joint_encoding(joint):
    xyz = joint["xyz"]
    rpy = joint["rpy"]
    is_fixed = joint["type"] == "fixed"

    if is_fixed:
        lower, upper = 0.0, 0.0
        axis = np.zeros(3)
    else:
        lower = joint["lower"]
        upper = joint["upper"]
        axis = joint["axis"]

    joint_struct = {
        "name": joint["name"],
        "joint_value_range": {"lower": lower, "upper": upper},
        "origin": {"xyz": xyz, "rpy": rpy},
        "axis": axis,
        "parent_link": joint["parent"],
        "child_link": joint["child"],
    }
    encoding = [
        lower,
        upper,
        rpy[0],
        rpy[1],
        rpy[2],
        xyz[0],
        xyz[1],
        xyz[2],
        axis[0],
        axis[1],
        axis[2],
    ]
    return joint_struct, encoding, is_fixed


def resolve_mesh_path(urdf_path, mesh_filename):
    mesh_path = Path(mesh_filename)
    if mesh_path.is_absolute() and mesh_path.exists():
        return mesh_path

    candidates = [
        urdf_path.parent / mesh_filename,
        urdf_path.parent / mesh_filename.replace("package://", ""),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    lower_name = mesh_path.name.lower()
    for candidate in urdf_path.parent.rglob("*"):
        if candidate.is_file() and candidate.name.lower() == lower_name:
            return candidate

    raise FileNotFoundError(f"Cannot resolve mesh path '{mesh_filename}' from {urdf_path}")


def mesh_bounding_box_size(mesh_path, scale):
    mesh = trimesh.load(mesh_path, force="mesh")
    if scale is not None:
        mesh.vertices *= np.asarray(scale)
    bounds = mesh.bounds
    return (bounds[1] - bounds[0]).tolist()


def get_link_collision_geometry_encoding(links, link_name, urdf_path):
    link = links.get(link_name)
    if link is None or len(link["collisions"]) == 0:
        return [DUMMY_LINK_TYPE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    collision = link["collisions"][0]
    xyz = collision["xyz"]
    rpy = collision["rpy"]
    geometry = collision["geometry"]

    mesh = geometry.find("mesh")
    if mesh is not None:
        mesh_path = resolve_mesh_path(urdf_path, mesh.attrib["filename"])
        scale = parse_vector(mesh.attrib.get("scale"), [1.0, 1.0, 1.0])
        size = mesh_bounding_box_size(mesh_path, scale)
        return [MESH_BOX_APPROX_TYPE, *rpy.tolist(), *xyz.tolist(), *size]

    box = geometry.find("box")
    if box is not None:
        size = parse_vector(box.attrib["size"], [0.0, 0.0, 0.0])
        return [BOX_TYPE, *rpy.tolist(), *xyz.tolist(), *size.tolist()]

    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return [
            CYLINDER_TYPE,
            *rpy.tolist(),
            *xyz.tolist(),
            float(cylinder.attrib["length"]),
            float(cylinder.attrib["radius"]),
            0.0,
        ]

    sphere = geometry.find("sphere")
    if sphere is not None:
        return [SPHERE_TYPE, *rpy.tolist(), *xyz.tolist(), float(sphere.attrib["radius"]), 0.0, 0.0]

    raise ValueError(f"Unsupported collision geometry for link '{link_name}'")


def tokenize_urdf(urdf_path, include_link_geometry=True):
    urdf_path = Path(urdf_path)
    robot = parse_urdf(urdf_path)
    joint_structs = []
    joint_encodings = []
    joint_revolute_property = []

    for joint in robot["joints"]:
        joint_struct, encoding, is_fixed = get_joint_encoding(joint)
        if include_link_geometry:
            parent_encoding = get_link_collision_geometry_encoding(
                robot["links"], joint_struct["parent_link"], urdf_path
            )
            child_encoding = get_link_collision_geometry_encoding(
                robot["links"], joint_struct["child_link"], urdf_path
            )
            encoding.extend(parent_encoding)
            encoding.extend(child_encoding)

        joint_structs.append(joint_struct)
        joint_encodings.append(encoding)
        joint_revolute_property.append(0 if is_fixed else 1)

    return joint_structs, joint_encodings, joint_revolute_property


def get_adjacency_matrix(joints):
    joint_names = [joint["name"] for joint in joints]
    adjacency_matrix = np.zeros((len(joint_names), len(joint_names)), dtype=int)

    for joint in joints:
        joint_index = joint_names.index(joint["name"])
        for other_joint in joints:
            other_joint_index = joint_names.index(other_joint["name"])
            if other_joint["child_link"] == joint["parent_link"]:
                adjacency_matrix[joint_index][other_joint_index] = 1
                adjacency_matrix[other_joint_index][joint_index] = 1
            if other_joint["parent_link"] == joint["child_link"]:
                adjacency_matrix[joint_index][other_joint_index] = 1
                adjacency_matrix[other_joint_index][joint_index] = 1

    return adjacency_matrix


def compute_spatial_distance_matrix(adjacency_matrix):
    num_joints = adjacency_matrix.shape[0]
    distance_matrix = np.full((num_joints, num_joints), np.inf)
    np.fill_diagonal(distance_matrix, 0)

    for start in range(num_joints):
        queue = deque([start])
        visited = {start}
        while queue:
            joint_index = queue.popleft()
            for neighbor_index in np.where(adjacency_matrix[joint_index] == 1)[0]:
                if neighbor_index in visited:
                    continue
                distance_matrix[start][neighbor_index] = distance_matrix[start][joint_index] + 1
                visited.add(neighbor_index)
                queue.append(neighbor_index)

    return distance_matrix


def compute_kinematic_distance_matrices(joints):
    num_joints = len(joints)
    parent_distance_matrix = np.full((num_joints, num_joints), np.inf)
    child_distance_matrix = np.full((num_joints, num_joints), np.inf)
    joint_index_map = {joint["name"]: idx for idx, joint in enumerate(joints)}
    np.fill_diagonal(parent_distance_matrix, 0)
    np.fill_diagonal(child_distance_matrix, 0)

    for joint in joints:
        current_joint_idx = joint_index_map[joint["name"]]
        for other_joint in joints:
            other_joint_idx = joint_index_map[other_joint["name"]]
            if joint["child_link"] == other_joint["parent_link"]:
                parent_distance_matrix[current_joint_idx][other_joint_idx] = 1
            if joint["parent_link"] == other_joint["child_link"]:
                child_distance_matrix[current_joint_idx][other_joint_idx] = 1

    for k in range(num_joints):
        for i in range(num_joints):
            for j in range(num_joints):
                parent_distance_matrix[i][j] = min(
                    parent_distance_matrix[i][j],
                    parent_distance_matrix[i][k] + parent_distance_matrix[k][j],
                )
                child_distance_matrix[i][j] = min(
                    child_distance_matrix[i][j],
                    child_distance_matrix[i][k] + child_distance_matrix[k][j],
                )

    return parent_distance_matrix, child_distance_matrix


def build_hand_meta(urdf_path, include_link_geometry=True):
    joint_structs, joint_encodings, joint_revolute_property = tokenize_urdf(
        urdf_path, include_link_geometry=include_link_geometry
    )
    joint_names = [joint["name"] for joint in joint_structs]
    adjacency_matrix = get_adjacency_matrix(joint_structs)
    spatial_distance_matrix = compute_spatial_distance_matrix(adjacency_matrix)
    parent_distance_matrix, child_distance_matrix = compute_kinematic_distance_matrices(
        joint_structs
    )

    return {
        "joint_structs": joint_structs,
        "joint_encodings": joint_encodings,
        "joint_revolute_property": joint_revolute_property,
        "adjacency_matrix": adjacency_matrix,
        "spatial_distance_matrix": spatial_distance_matrix,
        "parent_distance_matrix": parent_distance_matrix,
        "child_distance_matrix": child_distance_matrix,
        "joint_names": joint_names,
    }


def generate_meta(name, assets_root, output_dir, include_link_geometry=True):
    if name not in DEFAULT_HAND_URDFS:
        raise ValueError(f"Unknown hand '{name}'. Choices: {sorted(DEFAULT_HAND_URDFS)}")

    assets_root = Path(assets_root)
    output_dir = Path(output_dir)
    urdf_path = assets_root / DEFAULT_HAND_URDFS[name]
    output_path = output_dir / DEFAULT_OUTPUT_NAMES[name]
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = build_hand_meta(urdf_path, include_link_geometry=include_link_geometry)
    torch.save(meta, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate hand morphology metadata from URDFs.")
    parser.add_argument(
        "--assets_root",
        default=str(Path(__file__).resolve().parents[1] / "assets"),
        help="Directory containing hand asset folders such as shadow/, allegro/, and barrett/.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent / "meta"),
        help="Directory where generated .pt metadata files are written.",
    )
    parser.add_argument(
        "--hands",
        nargs="+",
        default=RELEASED_HANDS,
        choices=sorted(DEFAULT_HAND_URDFS.keys()),
        help="Hand metadata entries to generate.",
    )
    parser.add_argument(
        "--no_link_geometry",
        action="store_true",
        help="Generate joint-only tokens without parent/child link geometry.",
    )
    args = parser.parse_args()

    for hand_name in args.hands:
        output_path = generate_meta(
            hand_name,
            args.assets_root,
            args.output_dir,
            include_link_geometry=not args.no_link_geometry,
        )
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
