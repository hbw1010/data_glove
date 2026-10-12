"""Conservative Hand2 mesh self-collision constraints (meters at the geometry boundary).

Each URDF collision mesh has a complete convex hull, not a sphere or sampled
surface. Hull separation is conservative, but fills real assembly recesses.
The wrist / root-abduction two-axis assemblies therefore retain their exact STL
surfaces for distance checking; those pairs are not removed.
Fixed joints form one rigid body; pairs within that body and across one moving
joint are assembly contacts and are excluded. All other pairs remain active,
including non-adjacent segments of the same finger. No neutral-pose culling.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import coal
import numpy as np
import pinocchio as pin


class SelfCollision:
    """Cached signed hull distances and analytic contact-point Jacobians.

    Public distances/Jacobians are cm and cm/rad, respectively, matching the
    retarget objective. NLopt receives margin_cm - distance_cm <= 0. Convex
    signed distance is piecewise differentiable; at feature changes Coal picks
    one supporting normal (a local subgradient).
    """

    def __init__(self, robot, urdf_path: str, config: dict):
        self.model = robot.model
        if self.model.nq != self.model.nv:
            raise ValueError("Self-collision requires scalar Hand2 joints")
        self.margin_m = float(config.get("margin_m", 0.0005))
        self.tolerance_m = float(config.get("tolerance_m", 1e-7))
        if not np.isfinite(self.margin_m) or self.margin_m <= 0:
            raise ValueError("self_collision.margin_m must be positive and finite")
        if not np.isfinite(self.tolerance_m) or not 0 <= self.tolerance_m < self.margin_m:
            raise ValueError("self_collision.tolerance_m must be finite and smaller than margin_m")
        self.data = self.model.createData()
        self.geometry_model = pin.buildGeomFromUrdf(
            self.model, str(urdf_path), pin.GeometryType.COLLISION,
            package_dirs=[str(Path(urdf_path).parent)],
        )
        self.geometry_names = []
        self._geometry_description = []
        meshes = []
        for geometry in self.geometry_model.geometryObjects:
            mesh = geometry.geometry
            if not isinstance(mesh, coal.BVHModelBase):
                raise ValueError(f"Expected a collision STL mesh: {geometry.name}")
            meshes.append(mesh)
            mesh.buildConvexHull(True, "Qt")
            geometry.geometry = mesh.convex
            geometry.geometry.computeLocalAABB()
            link = self.model.frames[geometry.parentFrame].name
            self.geometry_names.append(link)
            self._geometry_description.append({
                "name": geometry.name, "link": link,
                "mesh": str(Path(geometry.meshPath).resolve()),
                "scale": np.asarray(geometry.meshScale).tolist(),
                "parent_joint": int(geometry.parentJoint),
                "placement_in_parent_joint": np.asarray(geometry.placement.homogeneous).tolist(),
            })
        if not self.geometry_names:
            raise ValueError("URDF has no collision meshes")
        self.geometry_data = self.geometry_model.createData()
        root = ET.parse(urdf_path).getroot()
        bodies = {link.attrib["name"]: link.attrib["name"] for link in root.findall("link")}

        def body(link):
            while bodies[link] != link:
                link = bodies[link]
            return link

        joints = []
        for joint in root.findall("joint"):
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            joints.append((parent, child, joint.attrib["type"]))
            if joint.attrib["type"] == "fixed":
                bodies[body(child)] = body(parent)
        adjacent = {
            frozenset((body(parent), body(child)))
            for parent, child, kind in joints if kind != "fixed"
        }
        # Hand2's nested flex/abduction root sits in a concave palm socket.
        # A single palm hull fills that socket (ring/pinky hulls cannot reach
        # 0.5 mm anywhere in their joint limits). Preserve the CAD surfaces
        # for exactly these named, topology-verified two-axis assemblies.
        parents = {child: (parent, kind) for parent, child, kind in joints}
        exact_assemblies = {}
        for link in self.geometry_names:
            if not link.endswith("_proximal_abd"):
                continue
            proximal, kind = parents[link]
            if proximal != link.removesuffix("_abd") or kind != "revolute":
                continue
            wrist, kind = parents[proximal]
            if wrist not in ("l_wrist", "r_wrist") or kind != "revolute":
                continue
            exact_assemblies[frozenset((wrist, link))] = [wrist, proximal, link]
        self.exact_assembly_pairs = []
        self._pair_geometries = []
        self.pairs = []
        self.excluded_pairs = []
        for first, link1 in enumerate(self.geometry_names):
            for second in range(first + 1, len(self.geometry_names)):
                link2 = self.geometry_names[second]
                body1, body2 = body(link1), body(link2)
                reason = None
                if body1 == body2:
                    reason = "same rigid body (URDF fixed-joint component)"
                elif frozenset((body1, body2)) in adjacent:
                    reason = "mechanical neighbors (one moving URDF joint between rigid bodies)"
                if reason:
                    self.excluded_pairs.append({"links": [link1, link2], "reason": reason})
                else:
                    self.pairs.append((first, second))
                    chain = exact_assemblies.get(frozenset((link1, link2)))
                    if chain is not None:
                        self._pair_geometries.append((meshes[first], meshes[second]))
                        self.exact_assembly_pairs.append({
                            "links": [link1, link2], "chain": chain,
                            "reason": "Two-axis root assembly; exact STL preserves the concave palm socket.",
                        })
                    else:
                        self._pair_geometries.append((
                            self.geometry_model.geometryObjects[first].geometry,
                            self.geometry_model.geometryObjects[second].geometry,
                        ))
        if not self.pairs:
            raise ValueError("URDF has no non-assembly self-collision pairs")
        self.margins_m = np.full(len(self.pairs), self.margin_m)
        pair_indices = {
            frozenset((self.geometry_names[a], self.geometry_names[b])): index
            for index, (a, b) in enumerate(self.pairs)
        }
        self.pair_margin_overrides = []
        overridden = set()
        for override in config.get("pair_margins", []):
            links = override["links"]
            key = frozenset(links)
            margin = float(override["margin_m"])
            reason = str(override["reason"]).strip()
            if len(links) != 2 or key not in pair_indices or key in overridden:
                raise ValueError(f"Unknown, excluded, or duplicate collision-margin pair: {links}")
            if not np.isfinite(margin) or margin <= self.tolerance_m or not reason:
                raise ValueError("Pair margin must exceed numerical tolerance and include a reason")
            self.margins_m[pair_indices[key]] = margin
            overridden.add(key)
            self.pair_margin_overrides.append({"links": list(links), "margin_m": margin, "reason": reason})
        self._margins_cm = self.margins_m * 100.0
        self.request = coal.DistanceRequest()
        self.request.enable_signed_distance = True
        # Keep geometric error below the independently enforced 0.1 um tolerance.
        self.request.gjk_tolerance = 1e-9
        self.request.epa_tolerance = 1e-9
        self._surface_pairs = np.array([
            index for index, (first, _) in enumerate(self._pair_geometries)
            if isinstance(first, coal.BVHModelBase)
        ], dtype=np.intp)
        self._results = [coal.DistanceResult() for _ in self.pairs]
        self._distances = np.empty(len(self.pairs))
        self._normals = np.empty((len(self.pairs), 3))
        self._points = np.empty((len(self.pairs), 2, 3))
        self._gradients = np.empty((len(self.pairs), self.model.nv))
        self._joint_ids = sorted({int(g.parentJoint) for g in self.geometry_model.geometryObjects})
        self._pair_joints = np.array([
            [self.geometry_model.geometryObjects[index].parentJoint for index in pair]
            for pair in self.pairs
        ], dtype=np.intp)
        # Joint zero is the fixed world: its twist and origin remain zero.
        self._joint_jacobians = np.zeros((self.model.njoints, 6, self.model.nv))
        self._joint_origins = np.zeros((self.model.njoints, 3))
        self._witness_twists = np.empty((len(self.pairs), 2, 6))
        self._witness_gradients = np.empty((len(self.pairs), 2, self.model.nv))
        self._qpos = None
        self._has_gradients = False

    def describe(self) -> dict:
        """Serializable policy and actual geometry/pair inventory, not a safety claim for old data."""
        return {
            "enabled": True,
            "representation": "complete collision-STL convex hulls; exact STL for listed two-axis root assemblies",
            "conservatism": "Hull contains mesh and fills concavities; listed root assemblies use original CAD surfaces instead.",
            "limitation": "Excluded assembly pairs are not collision checked; no environment or swept-motion checking.",
            "distance_units": "cm", "gradient_units": "cm/rad",
            "margin_m": self.margin_m, "tolerance_m": self.tolerance_m,
            "gjk_tolerance_m": self.request.gjk_tolerance,
            "epa_tolerance_m": self.request.epa_tolerance,
            "constraint": "NLopt SLSQP: 100 * (pair_margin_m - signed_distance_m) <= 0",
            "final_validation": "fresh signed-distance evaluation before warm-start commit and after filtering",
            "pair_policy": "All pairs except fixed-joint rigid bodies and single-moving-joint assembly neighbors; no pose-based exclusions.",
            "geometry_count": len(self.geometry_names), "pair_count": len(self.pairs),
            "geometry": self._geometry_description,
            "pairs": [[self.geometry_names[a], self.geometry_names[b]] for a, b in self.pairs],
            "excluded_pairs": self.excluded_pairs,
            "exact_assembly_pairs": self.exact_assembly_pairs,
            "pair_margin_overrides": self.pair_margin_overrides,
        }

    def evaluate(self, qpos, *, gradients=False, fresh=False):
        """Return cached (signed distances_cm, gradients_cm_per_rad or None).

        Returned arrays belong to this object; copy if retaining across calls.
        Separate Pinocchio data prevents objective FK from invalidating this cache.
        """
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError("Self-collision received invalid joint positions")
        if fresh or self._qpos is None or not np.array_equal(qpos, self._qpos):
            self._qpos = None
            pin.computeJointJacobians(self.model, self.data, qpos)
            pin.updateGeometryPlacements(self.model, self.data, self.geometry_model, self.geometry_data)
            transforms = [coal.Transform3s(p.rotation, p.translation) for p in self.geometry_data.oMg]
            for index, (first, second) in enumerate(self.pairs):
                result = self._results[index]
                result.clear()
                geometry1, geometry2 = self._pair_geometries[index]
                distance = coal.distance(
                    geometry1, transforms[first],
                    geometry2, transforms[second], self.request, result,
                )
                self._distances[index] = float(distance) * 100.0
                self._normals[index] = result.normal
                self._points[index, 0] = result.getNearestPoint1()
                self._points[index, 1] = result.getNearestPoint2()
            # Coal's BVH triangle-distance traversal does not populate normal.
            # Separated surfaces have the witness-difference normal; intersecting
            # surfaces report zero distance (a flat infeasible region), not a
            # signed penetration depth. Never use uninitialized result.normal.
            self._normals[self._surface_pairs] = 0.0
            separated = self._surface_pairs[self._distances[self._surface_pairs] > 0.0]
            self._normals[separated] = (
                self._points[separated, 1] - self._points[separated, 0]
            ) / (self._distances[separated, None] * 0.01)
            if not (np.isfinite(self._distances).all() and np.isfinite(self._normals).all()
                    and np.isfinite(self._points).all()):
                raise RuntimeError("Coal returned non-finite self-collision distance or witnesses")
            self._qpos = qpos.copy()
            self._has_gradients = False
        if gradients and not self._has_gradients:
            for joint in self._joint_ids:
                if joint != 0:
                    self._joint_jacobians[joint] = pin.getJointJacobian(
                        self.model, self.data, joint, pin.LOCAL_WORLD_ALIGNED,
                    )
                    self._joint_origins[joint] = self.data.oMi[joint].translation
            offsets = self._points - self._joint_origins[self._pair_joints]
            # n dot (v + omega cross offset), batched over every pair/witness.
            # Scalar np.cross dispatch for each witness dominated the solve.
            self._witness_twists[:, :, :3] = self._normals[:, None, :]
            self._witness_twists[:, :, 3:] = np.cross(offsets, self._normals[:, None, :])
            np.einsum(
                "pwc,pwcj->pwj", self._witness_twists,
                self._joint_jacobians[self._pair_joints], out=self._witness_gradients,
            )
            np.subtract(
                self._witness_gradients[:, 1], self._witness_gradients[:, 0],
                out=self._gradients,
            )
            self._gradients *= 100.0
            if not np.isfinite(self._gradients).all():
                raise RuntimeError("Non-finite self-collision constraint gradient")
            self._has_gradients = True
        return self._distances, self._gradients if gradients else None

    def constraint(self, result, qpos, gradient):
        """NLopt vector inequality callback; cm residual and cm/rad derivative."""
        distances, derivatives = self.evaluate(qpos, gradients=bool(gradient.size))
        result[:] = self._margins_cm - distances
        if gradient.size:
            gradient[:] = -derivatives

    def validate(self, qpos) -> None:
        """Independently reject infeasible output; never changes optimizer history."""
        distances, _ = self.evaluate(qpos, fresh=True)
        index = int(np.argmax(self._margins_cm - distances))
        if distances[index] < self._margins_cm[index] - self.tolerance_m * 100.0:
            first, second = self.pairs[index]
            raise RuntimeError(
                f"Hand2 self-collision infeasible: {self.geometry_names[first]} / "
                f"{self.geometry_names[second]} signed distance {distances[index] * 10:.6f} mm "
                f"< margin {self.margins_m[index] * 1000:.6f} mm"
            )
