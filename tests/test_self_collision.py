"""Mesh clearance, signed analytic derivatives, and failed-solve history contracts."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from wuji_retargeting import Retargeter
from wuji_retargeting.robot import RobotWrapper
from wuji_retargeting.opt import LPFilter
from wuji_retargeting.self_collision import SelfCollision

_RIGHT_FEASIBLE = np.array([
    -0.0207093425, -0.1633110760, -0.0861826230, -0.0208680513,
    -0.0461527753, -0.2364904527, 0.1506765441, 0.7702559112,
    -0.0647914667, 0.3675256914, -0.0932094473, 0.0714426468,
    0.2252271604, -0.2897688222, 0.1595319194, 0.1791393839,
    0.5460659683, -0.2151768176, 0.1827556419, 0.5468318024,
])
_LEFT_FEASIBLE = np.array([
    -0.2299051100, -0.5421521853, 0.5590627251, -0.7860267655,
    -0.3874062936, -0.6449293414, 0.3619610526, 1.3005896040,
    -0.3450213630, 0.2455636133, 0.2745402908, -0.1277431315,
    0.7259031149, 0.2838707474, 1.4220043041, 0.4024256015,
    0.6860262470, 0.0923153620, 0.6137657150, 1.0833185464,
])


@pytest.fixture
def sliding_meshes(tmp_path):
    # Two complete 20 mm cubes on independent prismatic joints. Their bodies
    # are not mechanical neighbors; a shared parent is not an exclusion.
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ]) * 0.01
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    facets = []
    for face in faces:
        points = "\n".join("vertex " + " ".join(map(str, vertices[i])) for i in face)
        facets.append(f"facet normal 0 0 0\nouter loop\n{points}\nendloop\nendfacet")
    mesh = tmp_path / "cube.stl"
    mesh.write_text("solid cube\n" + "\n".join(facets) + "\nendsolid cube\n")
    links = []
    for name in ("a", "b"):
        links.append(f'''<link name="{name}"><collision><geometry>
          <mesh filename="{mesh}"/></geometry></collision></link>
          <joint name="{name}_slide" type="prismatic"><parent link="root"/>
          <child link="{name}"/><axis xyz="1 0 0"/>
          <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>''')
    urdf = tmp_path / "slides.urdf"
    urdf.write_text('<robot name="slides"><link name="root"/>' + "".join(links) + '</robot>')
    robot = RobotWrapper(str(urdf))
    return SelfCollision(robot, str(urdf), {"margin_m": 0.0005})


def test_mesh_signed_distance_gradient_and_margin(sliding_meshes):
    collision = sliding_meshes
    for separation, expected_distance_cm in ((0.03, 1.0), (0.015, -0.5)):
        qpos = np.array([0.0, separation])
        distance, gradient = collision.evaluate(qpos, gradients=True)
        np.testing.assert_allclose(distance, [expected_distance_cm], atol=1e-5)
        np.testing.assert_allclose(gradient, [[-100.0, 100.0]], atol=1e-4)
        residual, derivative = np.empty(1), np.empty((1, 2))
        collision.constraint(residual, qpos, derivative)
        np.testing.assert_allclose(residual, [0.05 - expected_distance_cm], atol=1e-5)
        np.testing.assert_allclose(derivative, [[100.0, -100.0]], atol=1e-4)
    collision.validate([0.0, 0.0206])
    # Positive distance alone is insufficient: enforce the configured margin.
    for separation in (0.0204, 0.015):
        with pytest.raises(RuntimeError, match="self-collision infeasible"):
            collision.validate([0.0, separation])


def _retargeter(side):
    root = Path(__file__).resolve().parents[1]
    path = root / "config" / "retargeting" / "official" / f"{side}.yaml"
    config = yaml.safe_load(path.read_text())
    config["__yaml_dir"] = str(path.parent)
    config["optimizer"]["self_collision"]["enabled"] = True
    return Retargeter.from_config(config, side)


def test_filtered_interpolation_cannot_commit_a_collision(sliding_meshes):
    first, second = np.array([0.0, 0.03]), np.array([0.03, 0.0])
    sliding_meshes.validate(first)
    sliding_meshes.validate(second)
    retargeter = Retargeter.__new__(Retargeter)
    retargeter.lp_filter = LPFilter(0.5)
    retargeter.lp_filter.next(first)
    retargeter.optimizer = SimpleNamespace(
        num_joints=2, last_qpos=first.copy(), self_collision=sliding_meshes,
    )

    def solve(_):
        retargeter.optimizer.last_qpos = second.copy()
        return second.copy()

    retargeter.optimizer.solve = solve
    with pytest.raises(RuntimeError, match="self-collision infeasible"):
        retargeter._solve_and_filter(np.zeros((21, 3)), True)
    np.testing.assert_array_equal(retargeter.optimizer.last_qpos, first)
    np.testing.assert_array_equal(retargeter.lp_filter.y, first)
    assert retargeter.lp_filter.is_init


@pytest.mark.parametrize("side,qpos", [("right", _RIGHT_FEASIBLE), ("left", _LEFT_FEASIBLE)])
def test_real_hand_known_feasible_pose(side, qpos):
    collision = _retargeter(side).optimizer.self_collision
    collision.validate(qpos)
    distances, _ = collision.evaluate(qpos)
    assert np.all(distances >= (collision.margins_m - collision.tolerance_m) * 100.0)


@pytest.mark.parametrize("side,qpos", [("right", _RIGHT_FEASIBLE), ("left", _LEFT_FEASIBLE)])
def test_real_hand_collision_gradients_match_joint_motion(side, qpos):
    collision = _retargeter(side).optimizer.self_collision
    _, analytical = collision.evaluate(qpos, gradients=True)
    analytical = analytical.copy()
    # Exercise rotational witness offsets, shared ancestors and the fixed wrist;
    # the sliding-cube case alone cannot detect incorrect batched Jacobians.
    step = 1e-6
    for joint in range(qpos.size):
        offset = np.zeros_like(qpos)
        offset[joint] = step
        plus, _ = collision.evaluate(qpos + offset)
        plus = plus.copy()
        minus, _ = collision.evaluate(qpos - offset)
        numerical = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(analytical[:, joint], numerical, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("side", ["left", "right"])
def test_real_hand_rejects_positive_status_infeasible_solution_without_history_commit(side):
    optimizer = _retargeter(side).optimizer
    collision = optimizer.self_collision
    penetrating = np.zeros(optimizer.num_joints)
    prefix = "l_" if side == "left" else "r_"
    names = optimizer.robot.dof_joint_names
    penetrating[names.index(prefix + "index_finger_mcp_abd")] = 0.6
    penetrating[names.index(prefix + "middle_finger_mcp_abd")] = -0.6
    distances, _ = collision.evaluate(penetrating)
    crossing_pair = {
        prefix + "index_finger_proximal_abd", prefix + "middle_finger_proximal_abd",
    }
    pair_index = next(
        index for index, (a, b) in enumerate(collision.pairs)
        if {collision.geometry_names[a], collision.geometry_names[b]} == crossing_pair
    )
    assert distances[pair_index] < -1.0  # Measured inter-finger penetration exceeds 10 mm.
    with pytest.raises(RuntimeError, match="self-collision infeasible"):
        collision.validate(penetrating)
    previous = _LEFT_FEASIBLE if side == "left" else _RIGHT_FEASIBLE
    optimizer.last_qpos = previous.copy()
    original = optimizer.opt

    class FalseSuccess:
        def set_min_objective(self, objective):
            pass

        def optimize(self, initial):
            return penetrating.copy()

        def last_optimize_result(self):
            return 3

        def last_optimum_value(self):
            return 0.0

    optimizer.opt = FalseSuccess()
    try:
        with pytest.raises(RuntimeError, match="self-collision infeasible"):
            optimizer._run_optimization(lambda q, grad: 0.0, previous)
        np.testing.assert_array_equal(optimizer.last_qpos, previous)
    finally:
        optimizer.opt = original
