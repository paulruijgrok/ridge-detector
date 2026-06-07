import numpy as np
from ridge_detector import RidgeDetector
from ridge_detector.utils import convolve_gauss, LinesUtil, fix_locations, bresenham
from scipy.ndimage import convolve
from ridge_detector.constants import *


def eigh_2x2(a, b, d):
    """
    Analytical eigendecomposition of batched symmetric 2x2 matrices.
    M = [[a, b], [b, d]]

    Replaces np.linalg.eigh for the special case of 2x2 matrices —
    exact (no approximation), fully vectorized, no LAPACK overhead.

    Returns:
        eigvals: (..., 2) sorted by absolute value descending
        eigvecs: (..., 2, 2) where eigvecs[..., :, i] is the i-th eigenvector
    """
    # Eigenvalues via quadratic formula
    disc = np.sqrt((a - d) ** 2 + 4 * b ** 2)  # always >= 0
    trace = a + d
    lam1 = (trace + disc) / 2
    lam2 = (trace - disc) / 2

    # Sort by absolute value descending
    eigvals = np.stack([lam1, lam2], axis=-1)
    idx = np.abs(eigvals).argsort()[..., ::-1]
    eigvals = np.take_along_axis(eigvals, idx, axis=-1)

    # Eigenvector for lam1: v = normalize([b, lam1 - a])
    # When b=0 and lam1=a (degenerate), fall back to [1, 0]
    v_r = b
    v_c = lam1 - a
    norm = np.sqrt(v_r ** 2 + v_c ** 2)
    degenerate = norm < 1e-10
    safe_norm = np.where(degenerate, 1.0, norm)
    ev1_r = np.where(degenerate, 1.0, v_r / safe_norm)
    ev1_c = np.where(degenerate, 0.0, v_c / safe_norm)

    # Second eigenvector is perpendicular to first
    ev2_r = -ev1_c
    ev2_c = ev1_r

    # Build eigvecs: shape (..., 2, 2), column i = i-th eigenvector
    eigvecs = np.stack([
        np.stack([ev1_r, ev1_c], axis=-1),
        np.stack([ev2_r, ev2_c], axis=-1),
    ], axis=-1)

    # Reorder columns to match eigenvalue sort
    eigvecs = np.take_along_axis(eigvecs, idx[..., None, :], axis=-1)

    return eigvals, eigvecs


class OptimizedRidgeDetector(RidgeDetector):
    """
    RidgeDetector with np.linalg.eigh replaced by an analytical 2x2 solver.
    Results are numerically equivalent; speed improvement comes from avoiding
    LAPACK overhead on trivially small matrices.
    """

    def apply_filtering(self):
        height, width = self.gray.shape[:2]
        num_scales = len(self.sigmas)
        saliency = np.zeros((height, width, num_scales), dtype=float)
        orientation = np.zeros((height, width, 2, num_scales), dtype=float)
        rys   = np.zeros((height, width, num_scales), dtype=float)
        rxs   = np.zeros((height, width, num_scales), dtype=float)
        ryys  = np.zeros((height, width, num_scales), dtype=float)
        rxys  = np.zeros((height, width, num_scales), dtype=float)
        rxxs  = np.zeros((height, width, num_scales), dtype=float)
        low_threshs  = np.zeros((height, width, num_scales), dtype=float)
        high_threshs = np.zeros((height, width, num_scales), dtype=float)
        sigma_maps   = np.zeros((height, width, num_scales), dtype=float)

        gray = self.gray.astype(float)
        for scale_idx, sigma in enumerate(self.sigmas):
            ry  = convolve_gauss(gray, sigma, LinesUtil.DERIV_R)
            rx  = convolve_gauss(gray, sigma, LinesUtil.DERIV_C)
            ryy = convolve_gauss(gray, sigma, LinesUtil.DERIV_RR)
            rxy = convolve_gauss(gray, sigma, LinesUtil.DERIV_RC)
            rxx = convolve_gauss(gray, sigma, LinesUtil.DERIV_CC)

            # --- Replaced: np.linalg.eigh + take_along_axis ---
            eigvals_s, eigvecs_s = eigh_2x2(ryy, rxy, rxx)

            saliency[:, :, scale_idx]       = sigma ** 2.0 * eigvals_s[:, :, 0]
            orientation[:, :, :, scale_idx] = eigvecs_s[:, :, :, 0]

            rys[..., scale_idx]  = ry
            rxs[..., scale_idx]  = rx
            ryys[..., scale_idx] = ryy
            rxys[..., scale_idx] = rxy
            rxxs[..., scale_idx] = rxx

            line_width = 2 * np.sqrt(3) * (sigma - 0.5)
            low_thresh  = (0.17 * sigma ** 2.0 *
                           np.floor(self.clow * line_width / (np.sqrt(2 * np.pi) * sigma ** 3) *
                                    np.exp(-line_width ** 2 / (8 * sigma ** 2))))
            high_thresh = (0.17 * sigma ** 2.0 *
                           np.floor(self.chigh * line_width / (np.sqrt(2 * np.pi) * sigma ** 3) *
                                    np.exp(-line_width ** 2 / (8 * sigma ** 2))))
            low_threshs[..., scale_idx]  = low_thresh
            high_threshs[..., scale_idx] = high_thresh
            sigma_maps[..., scale_idx]   = sigma

        global_max_idx = saliency.argsort()[..., -1]
        self.lower_thresh = np.squeeze(np.take_along_axis(low_threshs,  global_max_idx[:, :, None], axis=-1))
        self.upper_thresh = np.squeeze(np.take_along_axis(high_threshs, global_max_idx[:, :, None], axis=-1))
        self.sigma_map    = np.squeeze(np.take_along_axis(sigma_maps,   global_max_idx[:, :, None], axis=-1))

        self.derivatives = np.zeros((5, height, width), dtype=float)
        self.derivatives[0] = np.squeeze(np.take_along_axis(rys,  global_max_idx[:, :, None], axis=-1))
        self.derivatives[1] = np.squeeze(np.take_along_axis(rxs,  global_max_idx[:, :, None], axis=-1))
        self.derivatives[2] = np.squeeze(np.take_along_axis(ryys, global_max_idx[:, :, None], axis=-1))
        self.derivatives[3] = np.squeeze(np.take_along_axis(rxys, global_max_idx[:, :, None], axis=-1))
        self.derivatives[4] = np.squeeze(np.take_along_axis(rxxs, global_max_idx[:, :, None], axis=-1))

        self.grady   = self.derivatives[0]
        self.gradx   = self.derivatives[1]
        self.eigvals = np.take_along_axis(saliency,     global_max_idx[:, :, None],       axis=-1)
        self.eigvecs = np.take_along_axis(orientation,  global_max_idx[:, :, None, None],  axis=-1)

    def compute_line_width(self):
        height, width = self.grady.shape[:2]
        length     = 2.5 * self.sigma_map
        max_length = np.ceil(length * 1.2).astype(int)
        grad = np.sqrt(self.grady ** 2 + self.gradx ** 2)

        grad_dr  = convolve(grad, kernel_r,  mode='mirror')
        grad_dc  = convolve(grad, kernel_c,  mode='mirror')
        grad_dd  = convolve(grad, kernel_d,  mode='mirror')
        grad_drr = convolve(grad, kernel_rr, mode='mirror')
        grad_drc = convolve(grad, kernel_rc, mode='mirror')
        grad_dcc = convolve(grad, kernel_cc, mode='mirror')

        # --- Replaced: np.linalg.eigh + take_along_axis ---
        eigvals, eigvecs = eigh_2x2(2 * grad_drr, grad_drc, 2 * grad_dcc)

        bb  = grad_dr * eigvecs[:, :, 0, 0] + grad_dc * eigvecs[:, :, 1, 0]
        aa  = 2.0 * (grad_drr * eigvecs[:, :, 0, 0] ** 2 +
                     grad_drc * eigvecs[:, :, 0, 0] * eigvecs[:, :, 1, 0] +
                     grad_dcc * eigvecs[:, :, 1, 0] ** 2)
        tt  = bb / (aa + np.finfo(float).eps)
        pp1 = tt * eigvecs[:, :, 0, 0]
        pp2 = tt * eigvecs[:, :, 1, 0]
        grad_rl = (grad_dd + pp1 * grad_dr + pp2 * grad_dc +
                   pp1 * pp1 * grad_drr + pp1 * pp2 * grad_drc + pp2 * pp2 * grad_dcc)

        for i, cont in enumerate(self.contours):
            num_points = cont.num
            width_l = np.zeros(num_points, dtype=float)
            width_r = np.zeros(num_points, dtype=float)
            grad_l  = np.zeros(num_points, dtype=float)
            grad_r  = np.zeros(num_points, dtype=float)
            pos_x   = np.zeros(num_points, dtype=float)
            pos_y   = np.zeros(num_points, dtype=float)

            for j in range(num_points):
                py, px = cont.row[j], cont.col[j]
                pos_y[j], pos_x[j] = py, px
                r, c = LinesUtil.BR(round(py), height), LinesUtil.BC(round(px), width)
                ny, nx = np.sin(cont.angle[j]), np.cos(cont.angle[j])

                line     = bresenham(ny, nx, max_length[r, c])
                num_line = line.shape[0]
                width_r[j] = width_l[j] = 0

                for direct in [-1, 1]:
                    for k in range(num_line):
                        y = LinesUtil.BR(r + direct * line[k, 0], height)
                        x = LinesUtil.BC(c + direct * line[k, 1], width)
                        val = -eigvals[y, x, 0]
                        if val > 0.0:
                            p1, p2 = pp1[y, x], pp2[y, x]
                            if abs(p1) <= 0.5 and abs(p2) <= 0.5:
                                t = (ny * (py - (r + direct * line[k, 0] + p1)) +
                                     nx * (px - (c + direct * line[k, 1] + p2)))
                                if direct == 1:
                                    grad_r[j] = grad_rl[y, x]
                                    width_r[j] = abs(t)
                                else:
                                    grad_l[j] = grad_rl[y, x]
                                    width_l[j] = abs(t)
                                break

            fix_locations(cont, width_l, width_r, grad_l, grad_r, pos_y, pos_x,
                          self.sigma_map, self.correct_pos, self.mode)
