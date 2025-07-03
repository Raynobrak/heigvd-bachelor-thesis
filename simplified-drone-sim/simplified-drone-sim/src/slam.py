# slam.py — Open3D ICP + pose‑graph 2‑D (unités : pixel)
# -------------------------------------------------------------------
# Paramètres testés :
#   voxel_size  = 4 px   | max_corr = 12 px
#   fitness_min = 0.25   | z_noise  = 0.2 px
#   key_distance = 25 px | window_kf = 30 | kf_fitness_min = 0.45 (KF gardés dans la carte)
# -------------------------------------------------------------------
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from typing import Optional, List, Tuple


class Slam:
    """ICP incrémental + pose‑graph pour carte 2‑D planaire."""

    def __init__(self,
                 voxel_size: float = 4.0,
                 max_corr: float = 12.0,
                 z_noise_sigma: float = 0.2,
                 key_distance: float = 25.0,
                 window_kf: int = 30,
                 kf_fitness_min: float = 0.45,
                 rng: Optional[np.random.Generator] = None):
        self.voxel = voxel_size
        self.max_corr = max_corr
        self.z_sigma = z_noise_sigma
        self.key_dist = key_distance
        self.window_kf = window_kf
        self.kf_fitness_min = kf_fitness_min
        self.rng = rng or np.random.default_rng()

        self.prev_down: Optional[o3d.geometry.PointCloud] = None  # nuage ref (local)
        self.pose = np.eye(4)                                     # monde ← capteur courant
        self.map = o3d.geometry.PointCloud()                      # carte (fenêtre KF)

        # --- pose‑graph ---
        self.pg = o3d.pipelines.registration.PoseGraph()
        self.pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
        self.kf_poses: List[np.ndarray] = [np.eye(4)]

    # ---------------- pré‑traitement ----------------
    def _prep(self, xyz: np.ndarray) -> o3d.geometry.PointCloud:
        xyz = np.asarray(xyz, np.float64)
        xyz[:, 2] += self.rng.normal(0.0, self.z_sigma, size=len(xyz))
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        p = p.voxel_down_sample(self.voxel)
        p, _ = p.remove_statistical_outlier(20, 2.0)
        return p

    # ------------- helpers --------------------------
    @staticmethod
    def _sanitize(T: np.ndarray) -> np.ndarray:
        """Verrouille le plan + orthogonalise la rotation."""
        T = T.copy()
        T[2, 3] = 0; T[0:2, 2] = 0; T[2, 0:2] = 0; T[2, 2] = 1
        R3 = T[:3, :3]
        if np.linalg.det(R3) < 0:
            R3[:, 1] *= -1
        U, _, Vt = np.linalg.svd(R3); T[:3, :3] = U @ Vt
        return T

    def _add_to_map(self, cloud_local: o3d.geometry.PointCloud):
        """Ajoute un KF transformé à la carte et gère la fenêtre glissante."""
        c = o3d.geometry.PointCloud(cloud_local)  # clone
        c.transform(self.pose)
        self.map += c
        # Fenêtre glissante
        if len(self.kf_poses) > self.window_kf:
            self.map.clear()
            self.kf_poses = self.kf_poses[-self.window_kf:]
            for pose in self.kf_poses:
                dummy = o3d.geometry.PointCloud(cloud_local)
                dummy.transform(pose)
                self.map += dummy

    # ---------------- boucle principale -------------
    def add_scan(self, scan_xyz: np.ndarray):
        down = self._prep(scan_xyz)

        if self.prev_down is None:  # premier scan
            self.prev_down = o3d.geometry.PointCloud(down)
            self._add_to_map(down)
            return

        reg = o3d.pipelines.registration.registration_icp(
            source=down,
            target=self.prev_down,
            max_correspondence_distance=self.max_corr,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

        if reg.fitness < 0.25:
            return  # recouvrement trop faible

        delta = self._sanitize(reg.transformation)
        self.pose = self.pose @ delta

        # --- key‑frame : distance ET bon overlap ---
        move_px = np.linalg.norm(self.pose[:2, 3] - self.kf_poses[-1][:2, 3])
        if move_px > self.key_dist and reg.fitness >= self.kf_fitness_min:
            self.pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(self.pose.copy()))
            self.pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                len(self.pg.nodes) - 2, len(self.pg.nodes) - 1, delta, uncertain=False))
            self.kf_poses.append(self.pose.copy())
            self._add_to_map(down)
            if len(self.pg.nodes) % 15 == 0:  # optimisation périodique
                option = o3d.pipelines.registration.GlobalOptimizationOption(
                    max_correspondence_distance=self.max_corr)
                o3d.pipelines.registration.global_optimization(
                    self.pg,
                    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                    option)
                self.pose = self.pg.nodes[-1].pose

        self.prev_down = o3d.geometry.PointCloud(down)

    # ---------------- getter -------------------------
    def get_estimated_state(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.pose[:3, 3].astype(float)
        quat = R.from_matrix(self.pose[:3, :3]).as_quat()
        return pos, np.roll(quat, 1)  # wxyz


# -------------------------------------------------------------------
# UTILITY : 2‑D points + couleurs magma (sans matplotlib)
# -------------------------------------------------------------------

def map_points_and_colors(slam: Slam, max_pts: int = 50_000):
    """Extrait (x,y) + couleur (violet→jaune) sans matplotlib."""
    xyz = np.asarray(slam.map.points, dtype=np.float32)
    if xyz.size == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 3), np.float32)

    if len(xyz) > max_pts:
        xyz = xyz[np.random.choice(len(xyz), max_pts, replace=False)]

    z = xyz[:, 2]
    z_norm = (z - z.min()) / (np.ptp(z) + 1e-8)

    # magma‑like: dark purple → yellow
    r = np.sqrt(z_norm)
    g = np.power(z_norm, 0.25)
    b = np.power(1.0 - z_norm, 2.0)
    colors = np.stack((r, g, b), axis=1).astype(np.float32)

    pts_xy = xyz[:, :2].astype(np.float32)
    return pts_xy, colors
