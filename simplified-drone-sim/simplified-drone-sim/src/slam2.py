# slam2.py — révision finale pour LiDAR 2‑D (unité : pixel)
# -------------------------------------------------------------
# Paramètres testés et stables (KISS‑ICP ≥ 1.2) :
#   • voxel_size   = 4 px  (sous‑échantillonnage de la carte locale)
#   • initial_threshold = 8 px  (≈ 2 × voxel, rayon de recherche ICP)
#   • fixed_threshold   = None   (on laisse l'adaptation activée)
#   • min_motion_th     = 0.5 px (ignore les scans Δ<0.5 px)
#   • deskew = False             (scan planaire instantané)
#   • z_noise_sigma = 0.2 px     (bruit Z anti‑coplanarité)
# ------------------------------------------------------------------
import numpy as np
from kiss_icp.kiss_icp import KissICP
from kiss_icp.config import KISSConfig
from scipy.spatial.transform import Rotation
from typing import Optional, Union


class KissICPSLAM:
    """Wrapper minimal KISS‑ICP pour un environnement 2‑D (pixels)."""

    def __init__(self,
                 voxel_size: float = 4.0,
                 z_noise_sigma: float = 0.2,
                 rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.z_sigma = float(z_noise_sigma)

        cfg = KISSConfig()
        cfg.mapping.voxel_size = float(voxel_size)
        cfg.adaptive_threshold.initial_threshold = 2.0 * voxel_size  # 8 px
        cfg.adaptive_threshold.fixed_threshold = None  # adaptation ON
        cfg.adaptive_threshold.min_motion_th = 0.5  # 0.5 px ⇒ pas de gel pose
        cfg.data.deskew = False  # scans instantanés
        cfg.registration.max_num_iterations = 40

        self.odometry = KissICP(config=cfg)
        self._pose = np.eye(4, dtype=np.float64)

    # ------------------------------------------------------------------
    def add_scan(self,
                 scan_xyz: np.ndarray,
                 stamp: Union[float, np.ndarray, None] = None) -> None:
        """Intègre un scan (N,3) en repère capteur.

        *Ajoute un bruit Z gaussien pour casser la dégénérescence plane.*
        """
        pts = np.asarray(scan_xyz, dtype=np.float32, order="C")
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("scan_xyz doit être de forme (N,3)")

        # Bruit Z (coplanarity breaker)
        pts[:, 2] += self.rng.normal(0.0, self.z_sigma, size=len(pts)).astype(np.float32)

        # Timestamps → vecteur zeros (deskew désactivé)
        if stamp is None:
            timestamps = np.zeros(len(pts), dtype=np.float64)
        elif np.isscalar(stamp):
            timestamps = np.full(len(pts), float(stamp), dtype=np.float64)
        else:
            timestamps = np.asarray(stamp, dtype=np.float64)
            if timestamps.shape[0] != len(pts):
                raise ValueError("len(timestamps) != len(points)")

        # Protection : on ignore proprement une frame si Sophus lève une erreur
        try:
            self.odometry.register_frame(pts, timestamps)
            self._pose = self.odometry.last_pose.copy()
        except RuntimeError:
            # on garde la pose précédente et log optionnel
            return

    # ------------------------------------------------------------------
    def get_estimated_state(self):
        """Retourne (pos xyz, quaternion wxyz) repère monde."""
        pos = self._pose[:3, 3].astype(float)
        quat_xyzw = Rotation.from_matrix(self._pose[:3, :3]).as_quat()
        return pos, np.roll(quat_xyzw, 1)
