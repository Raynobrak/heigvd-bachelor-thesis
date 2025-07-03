# slam.py — Open3D ICP pour LiDAR 360° 2‑D (unités : pixel)
# -------------------------------------------------------------------
#  Paramètres par défaut (carte 1000 × 450 px) :
#    voxel_size  = 4  px   → sous‑échantillonnage de la « local map »
#    max_corr    = 15 px   → rayon de recherche ICP (≈ 4 × voxel)
#    fitness_min = 0.20    → rejet si recouvrement trop faible
#    z_noise σ   = 0.2 px  → casse la coplanarité XY
# -------------------------------------------------------------------
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from typing import Optional


class Slam:
    """ICP point‑to‑point robuste pour un LiDAR plan simulé (unités : pixel)."""

    # -----------------------------------------------------------------
    def __init__(self,
                 voxel_size: float = 4.0,
                 max_corr: float = 15.0,
                 z_noise_sigma: float = 0.05,
                 rng: Optional[np.random.Generator] = None):
        self.voxel = float(voxel_size)
        self.max_corr = float(max_corr)
        self.z_sigma = float(z_noise_sigma)
        self.rng = rng or np.random.default_rng()

        # nuage de référence (repère capteur) pour l'ICP incrémental
        self.prev_down: Optional[o3d.geometry.PointCloud] = None
        # pose monde ← capteur courant
        self.pose = np.eye(4, dtype=np.float64)
        # carte cumulée (repère monde)
        self.map = o3d.geometry.PointCloud()

    # ------------------------- pré‑traitement ------------------------
    def _prep(self, xyz: np.ndarray) -> o3d.geometry.PointCloud:
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("scan (N,3) attendu")
        # bruit Z léger pour éviter la dégénérescence planaire
        xyz[:, 2] += self.rng.normal(0.0, self.z_sigma, size=xyz.shape[0])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        if self.voxel > 0:
            pcd = pcd.voxel_down_sample(self.voxel)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd

    # ------------------ aide : ajout à la carte monde ---------------
    def _integrate_into_map(self, pcd_local: o3d.geometry.PointCloud) -> None:
        """Copie `pcd_local`, le transforme vers le monde et l’ajoute à la carte."""
        pcd_world = o3d.geometry.PointCloud(pcd_local)  # clone local → world
        pcd_world.transform(self.pose)                  # in‑place
        self.map += pcd_world

    # ------------------ aide : nettoyage de transformation ----------
    @staticmethod
    def _sanitize_transform(T: np.ndarray) -> np.ndarray:
        """Verrouille Z + roll/pitch et ré‑orthogonalise R (det = +1)."""
        # translation Z = 0
        T[2, 3] = 0
        # annule les composantes hors plan
        T[0:2, 2] = 0
        T[2, 0:2] = 0
        T[2, 2] = 1
        # force rotation droitière
        R3 = T[:3, :3]
        if np.linalg.det(R3) < 0:
            R3[:, 1] *= -1
        U, _, Vt = np.linalg.svd(R3)
        T[:3, :3] = U @ Vt
        return T

    # --------------------------- boucle ICP -------------------------
    def add_scan(self, scan_xyz: np.ndarray) -> None:
        """Ajoute un scan (N,3) repère capteur et met à jour pose + carte."""
        down_local = self._prep(scan_xyz)

        # premier scan : initialise pose + carte
        if self.prev_down is None:
            self.prev_down = o3d.geometry.PointCloud(down_local)  # stocké local
            self._integrate_into_map(down_local)
            return

        # ICP incrémental entre deux scans *locaux*
        reg = o3d.pipelines.registration.registration_icp(
            source=down_local,
            target=self.prev_down,
            max_correspondence_distance=self.max_corr,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.
                TransformationEstimationPointToPoint())

        if reg.fitness < 0.20:
            # mauvais recouvrement : on ignore pour la stabilité
            print("[SLAM] scan ignoré – fitness trop bas")
            return

        delta_T = self._sanitize_transform(reg.transformation.copy())
        # accumulation de la pose monde ← capteur_k
        self.pose @= delta_T
        # mise à jour de la carte (copie monde)
        self._integrate_into_map(down_local)
        # prépare le prochain tour
        self.prev_down = o3d.geometry.PointCloud(down_local)

    # ----------------------------- getter ---------------------------
    def get_estimated_state(self):
        """Retourne (position xyz, quaternion wxyz) dans le repère monde."""
        pos = self.pose[:3, 3].astype(float)
        quat_xyzw = R.from_matrix(self.pose[:3, :3]).as_quat()
        quat_wxyz = np.roll(quat_xyzw, 1)
        return pos, quat_wxyz
