import math
import cv2
import numpy as np
import numba
from ridge_detector import RidgeDetector
from ridge_detector.utils import (convolve_gauss, LinesUtil, fix_locations, bresenham,
                                   Line, Junction,
                                   compute_gauss_mask_0, compute_gauss_mask_1, compute_gauss_mask_2)
from scipy.ndimage import convolve
from ridge_detector.constants import *

# Use float32 throughout — halves memory bandwidth, same results within image noise
DTYPE = np.float32

# Max ridge points per contour direction (2× image diagonal is a safe upper bound)
MAX_TRACE_POINTS = 12000


# ── Numba JIT kernel ──────────────────────────────────────────────────────────

@numba.njit(cache=True)
def _trace_direction(
    ismax, normy, normx, posy, posx,
    label, indx, done,
    resp_dd, resp_dr, resp_dc, resp_drr, resp_drc, resp_dcc,
    dirtab, cleartab,
    maxy, maxx,           # seed pixel (response interpolation origin)
    last_octant,          # octant_seed for dir1, octant_seed+4 for dir2
    last_beta,            # beta at seed
    num_cont,             # current contour index (0-based)
    height, width,
    out_row, out_col, out_angle, out_resp   # pre-allocated output buffers
):
    """
    Trace one ridge direction from seed (maxy, maxx) using pre-compiled Numba kernel.

    Modifies label / indx / done in-place.
    Returns (n_pts, hit_junction, junc_y, junc_x).
    """
    PI           = math.pi
    TWO_PI       = 2.0 * PI
    MAX_ANG_DIFF = PI / 6.0     # MAX_ANGLE_DIFFERENCE from constants

    n_pts   = 0
    max_pts = len(out_row)
    y, x    = maxy, maxx

    while True:
        # ── Direction at current pixel ───────────────────────────────────────
        ny_v = normy[y, x]
        nx_v = -normx[y, x]
        py   = posy[y, x]
        px   = posx[y, x]

        # inline normalize_to_half_circle(arctan2(ny, nx))
        alpha = math.atan2(ny_v, nx_v)
        if alpha < 0.0: alpha += PI
        if alpha >= PI: alpha -= PI

        # octant with continuity unwrapping
        octant = int(math.floor(4.0 / PI * alpha + 0.5)) % 4
        if   octant == 0 and 3 <= last_octant <= 5: octant = 4
        elif octant == 1 and 4 <= last_octant <= 6: octant = 5
        elif octant == 2 and 4 <= last_octant <= 7: octant = 6
        elif octant == 3 and (last_octant == 0 or last_octant >= 6): octant = 7
        last_octant = octant

        # ── Find best neighbor in dirtab ──────────────────────────────────────
        nextismax = False
        nexti     = 1
        mindiff   = 1e18
        for ti in range(3):
            ny_ = y + dirtab[octant, ti, 0]
            nx_ = x + dirtab[octant, ti, 1]
            if ny_ < 0 or ny_ >= height or nx_ < 0 or nx_ >= width: continue
            if ismax[ny_, nx_] == 0: continue
            dy   = posy[ny_, nx_] - py
            dx   = posx[ny_, nx_] - px
            dist = math.sqrt(dx*dx + dy*dy)
            na   = math.atan2(normy[ny_, nx_], -normx[ny_, nx_])
            if na < 0.0: na += PI
            if na >= PI: na -= PI
            diff = abs(alpha - na)
            if diff >= PI * 0.5: diff = PI - diff
            diff = dist + diff
            if diff < mindiff:
                mindiff = diff
                nexti   = ti
            nextismax = True

        # ── Mark double responses (cleartab) ──────────────────────────────────
        for ni in range(2):
            cy = y + cleartab[octant, ni, 0]
            cx = x + cleartab[octant, ni, 1]
            if cy < 0 or cy >= height or cx < 0 or cx >= width: continue
            if ismax[cy, cx] > 0:
                na = math.atan2(normy[cy, cx], -normx[cy, cx])
                if na < 0.0: na += PI
                if na >= PI: na -= PI
                diff = abs(alpha - na)
                if diff >= PI * 0.5: diff = PI - diff
                if diff < MAX_ANG_DIFF:
                    label[cy, cx] = num_cont + 1
                    ci = indx[cy, cx]
                    if ci != 0:
                        done[ci - 1] = True

        if not nextismax:
            break

        # ── Move to next pixel ────────────────────────────────────────────────
        y += dirtab[octant, nexti, 0]
        x += dirtab[octant, nexti, 1]

        if n_pts >= max_pts:
            break

        # ── Record point ──────────────────────────────────────────────────────
        out_row[n_pts] = posy[y, x]
        out_col[n_pts] = posx[y, x]

        # angle: normalize_to_half_circle(arctan2(normy, normx))  ← normx NOT negated
        beta = math.atan2(normy[y, x], normx[y, x])
        if beta < 0.0: beta += PI
        if beta >= PI: beta -= PI
        diff1     = min(abs(beta - last_beta),     TWO_PI - abs(beta - last_beta))
        alt_beta  = (beta + PI) % TWO_PI
        diff2     = min(abs(alt_beta - last_beta), TWO_PI - abs(alt_beta - last_beta))
        chosen_beta   = beta if diff1 < diff2 else alt_beta
        out_angle[n_pts] = chosen_beta
        last_beta = chosen_beta

        # response interpolation (origin = seed point)
        yy = posy[y, x] - float(maxy)
        xx = posx[y, x] - float(maxx)
        out_resp[n_pts] = (resp_dd[y, x] + yy*resp_dr[y, x]     + xx*resp_dc[y, x]  +
                           yy*yy*resp_drr[y, x] + xx*yy*resp_drc[y, x] + xx*xx*resp_dcc[y, x])
        n_pts += 1

        # ── Junction / end-of-line check ──────────────────────────────────────
        if label[y, x] > 0:
            return n_pts, True, y, x     # junction: return, Python handles resolution

        # mark as part of current contour
        label[y, x] = num_cont + 1
        ci = indx[y, x]
        if ci != 0:
            done[ci - 1] = True

    return n_pts, False, -1, -1


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
    disc = np.sqrt((a - d) ** 2 + 4 * b ** 2)
    trace = a + d
    lam1 = (trace + disc) / 2
    lam2 = (trace - disc) / 2

    eigvals = np.stack([lam1, lam2], axis=-1)
    idx = np.abs(eigvals).argsort()[..., ::-1]
    eigvals = np.take_along_axis(eigvals, idx, axis=-1)

    # Eigenvector for lam1: v = normalize([b, lam1 - a])
    # Degenerate case (b=0, lam1=a): fall back to [1, 0]
    v_r = b
    v_c = lam1 - a
    norm = np.sqrt(v_r ** 2 + v_c ** 2)
    # float32 epsilon ~1.2e-7, so use 1e-6 as degenerate threshold
    degenerate = norm < np.finfo(DTYPE).eps * 1e1
    safe_norm = np.where(degenerate, DTYPE(1.0), norm)
    ev1_r = np.where(degenerate, DTYPE(1.0), v_r / safe_norm)
    ev1_c = np.where(degenerate, DTYPE(0.0), v_c / safe_norm)

    # Second eigenvector is perpendicular
    ev2_r = -ev1_c
    ev2_c = ev1_r

    eigvecs = np.stack([
        np.stack([ev1_r, ev1_c], axis=-1),
        np.stack([ev2_r, ev2_c], axis=-1),
    ], axis=-1)
    eigvecs = np.take_along_axis(eigvecs, idx[..., None, :], axis=-1)

    return eigvals, eigvecs


class OptimizedRidgeDetector(RidgeDetector):
    """
    RidgeDetector with two optimizations over the original:
      1. np.linalg.eigh replaced by an analytical 2x2 solver (lossless)
      2. All internal arrays use float32 instead of float64 (lossless in practice)
    """

    def apply_filtering(self):
        height, width = self.gray.shape[:2]
        num_scales = len(self.sigmas)
        saliency     = np.zeros((height, width, num_scales), dtype=DTYPE)
        orientation  = np.zeros((height, width, 2, num_scales), dtype=DTYPE)
        rys          = np.zeros((height, width, num_scales), dtype=DTYPE)
        rxs          = np.zeros((height, width, num_scales), dtype=DTYPE)
        ryys         = np.zeros((height, width, num_scales), dtype=DTYPE)
        rxys         = np.zeros((height, width, num_scales), dtype=DTYPE)
        rxxs         = np.zeros((height, width, num_scales), dtype=DTYPE)
        low_threshs  = np.zeros((height, width, num_scales), dtype=DTYPE)
        high_threshs = np.zeros((height, width, num_scales), dtype=DTYPE)
        sigma_maps   = np.zeros((height, width, num_scales), dtype=DTYPE)

        gray = self.gray.astype(DTYPE)
        for scale_idx, sigma in enumerate(self.sigmas):
            # Pre-compute the 3 distinct 1D Gaussian masks (float32) once per sigma.
            # cv2.sepFilter2D(src, ddepth, kernelX, kernelY):
            #   kernelX filters each row  (X / column direction)
            #   kernelY filters each col  (Y / row    direction)
            # scipy convention: convolve(img, hr.reshape(-1,1)) = Y direction,
            #                   convolve(img, hc.reshape(1,-1)) = X direction
            # So:  kernelX = hc,  kernelY = hr
            h0 = compute_gauss_mask_0(sigma)[0].astype(DTYPE)   # Gaussian
            h1 = compute_gauss_mask_1(sigma)[0].astype(DTYPE)   # 1st derivative
            h2 = compute_gauss_mask_2(sigma)[0].astype(DTYPE)   # 2nd derivative

            # cv2.sepFilter2D uses correlation (no kernel flip); scipy.ndimage.convolve
            # uses true convolution (flips kernel before correlating).
            # h0 and h2 are symmetric  → flip has no effect → no correction needed.
            # h1 is antisymmetric      → flip negates it    → pass -h1 wherever h1
            #                            appears an ODD number of times in a given call.
            # ry  (kernelY=h1 once)  → negate kernelY
            # rx  (kernelX=h1 once)  → negate kernelX
            # rxy (h1 in both dims)  → two negations cancel, no correction needed
            h1n   = -h1
            border = cv2.BORDER_REPLICATE
            # DERIV_R:  hr=h1 (Y), hc=h0 (X)
            ry  = cv2.sepFilter2D(gray, cv2.CV_32F, h0,  h1n, borderType=border)
            # DERIV_C:  hr=h0 (Y), hc=h1 (X)
            rx  = cv2.sepFilter2D(gray, cv2.CV_32F, h1n, h0,  borderType=border)
            # DERIV_RR: hr=h2 (Y), hc=h0 (X)  — both symmetric, no correction
            ryy = cv2.sepFilter2D(gray, cv2.CV_32F, h0,  h2,  borderType=border)
            # DERIV_RC: hr=h1 (Y), hc=h1 (X)  — h1 twice, signs cancel
            rxy = cv2.sepFilter2D(gray, cv2.CV_32F, h1,  h1,  borderType=border)
            # DERIV_CC: hr=h0 (Y), hc=h2 (X)  — both symmetric, no correction
            rxx = cv2.sepFilter2D(gray, cv2.CV_32F, h2,  h0,  borderType=border)

            eigvals_s, eigvecs_s = eigh_2x2(ryy, rxy, rxx)

            saliency[:, :, scale_idx]       = DTYPE(sigma ** 2.0) * eigvals_s[:, :, 0]
            orientation[:, :, :, scale_idx] = eigvecs_s[:, :, :, 0]

            rys[..., scale_idx]  = ry
            rxs[..., scale_idx]  = rx
            ryys[..., scale_idx] = ryy
            rxys[..., scale_idx] = rxy
            rxxs[..., scale_idx] = rxx

            line_width  = 2 * np.sqrt(3) * (sigma - 0.5)
            low_thresh  = (0.17 * sigma ** 2.0 *
                           np.floor(self.clow * line_width / (np.sqrt(2 * np.pi) * sigma ** 3) *
                                    np.exp(-line_width ** 2 / (8 * sigma ** 2))))
            high_thresh = (0.17 * sigma ** 2.0 *
                           np.floor(self.chigh * line_width / (np.sqrt(2 * np.pi) * sigma ** 3) *
                                    np.exp(-line_width ** 2 / (8 * sigma ** 2))))
            low_threshs[..., scale_idx]  = DTYPE(low_thresh)
            high_threshs[..., scale_idx] = DTYPE(high_thresh)
            sigma_maps[..., scale_idx]   = DTYPE(sigma)

        global_max_idx = saliency.argsort()[..., -1]
        self.lower_thresh = np.squeeze(np.take_along_axis(low_threshs,  global_max_idx[:, :, None], axis=-1))
        self.upper_thresh = np.squeeze(np.take_along_axis(high_threshs, global_max_idx[:, :, None], axis=-1))
        self.sigma_map    = np.squeeze(np.take_along_axis(sigma_maps,   global_max_idx[:, :, None], axis=-1))

        self.derivatives = np.zeros((5, height, width), dtype=DTYPE)
        self.derivatives[0] = np.squeeze(np.take_along_axis(rys,  global_max_idx[:, :, None], axis=-1))
        self.derivatives[1] = np.squeeze(np.take_along_axis(rxs,  global_max_idx[:, :, None], axis=-1))
        self.derivatives[2] = np.squeeze(np.take_along_axis(ryys, global_max_idx[:, :, None], axis=-1))
        self.derivatives[3] = np.squeeze(np.take_along_axis(rxys, global_max_idx[:, :, None], axis=-1))
        self.derivatives[4] = np.squeeze(np.take_along_axis(rxxs, global_max_idx[:, :, None], axis=-1))

        self.grady   = self.derivatives[0]
        self.gradx   = self.derivatives[1]
        self.eigvals = np.take_along_axis(saliency,    global_max_idx[:, :, None],      axis=-1)
        self.eigvecs = np.take_along_axis(orientation, global_max_idx[:, :, None, None], axis=-1)

    def compute_line_width(self):
        height, width = self.grady.shape[:2]
        length     = 2.5 * self.sigma_map
        max_length = np.ceil(length * 1.2).astype(int)
        grad = np.sqrt(self.grady ** 2 + self.gradx ** 2).astype(DTYPE)

        grad_dr  = convolve(grad, kernel_r,  mode='mirror').astype(DTYPE)
        grad_dc  = convolve(grad, kernel_c,  mode='mirror').astype(DTYPE)
        grad_dd  = convolve(grad, kernel_d,  mode='mirror').astype(DTYPE)
        grad_drr = convolve(grad, kernel_rr, mode='mirror').astype(DTYPE)
        grad_drc = convolve(grad, kernel_rc, mode='mirror').astype(DTYPE)
        grad_dcc = convolve(grad, kernel_cc, mode='mirror').astype(DTYPE)

        eigvals, eigvecs = eigh_2x2(
            (DTYPE(2.0) * grad_drr).astype(DTYPE),
            grad_drc,
            (DTYPE(2.0) * grad_dcc).astype(DTYPE)
        )

        bb  = grad_dr * eigvecs[:, :, 0, 0] + grad_dc * eigvecs[:, :, 1, 0]
        aa  = DTYPE(2.0) * (grad_drr * eigvecs[:, :, 0, 0] ** 2 +
                             grad_drc * eigvecs[:, :, 0, 0] * eigvecs[:, :, 1, 0] +
                             grad_dcc * eigvecs[:, :, 1, 0] ** 2)
        tt  = bb / (aa + np.finfo(DTYPE).eps)
        pp1 = tt * eigvecs[:, :, 0, 0]
        pp2 = tt * eigvecs[:, :, 1, 0]
        grad_rl = (grad_dd + pp1 * grad_dr + pp2 * grad_dc +
                   pp1 * pp1 * grad_drr + pp1 * pp2 * grad_drc + pp2 * pp2 * grad_dcc)

        for i, cont in enumerate(self.contours):
            num_points = cont.num
            width_l = np.zeros(num_points, dtype=DTYPE)
            width_r = np.zeros(num_points, dtype=DTYPE)
            grad_l  = np.zeros(num_points, dtype=DTYPE)
            grad_r  = np.zeros(num_points, dtype=DTYPE)
            pos_x   = np.zeros(num_points, dtype=DTYPE)
            pos_y   = np.zeros(num_points, dtype=DTYPE)

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

    def compute_contours(self):
        """
        Optimized compute_contours: vectorized seed scan + Numba JIT inner loop.

        Drop-in replacement for the original. Matches original logic:
          - Crossref sorted descending → np.argsort descending
          - Seed row/col stored as integer coordinates (matches original)
          - Junction detection and registration in Python
        """
        height, width = self.eigval.shape[:2]

        # ── Response surface derivatives for sub-pixel interpolation ─────────
        eigval = self.eigval.reshape(height, width).astype(DTYPE)
        resp_dd  = convolve(eigval, kernel_d,  mode='mirror').astype(DTYPE)
        resp_dr  = convolve(eigval, kernel_r,  mode='mirror').astype(DTYPE)
        resp_dc  = convolve(eigval, kernel_c,  mode='mirror').astype(DTYPE)
        resp_drr = convolve(eigval, kernel_rr, mode='mirror').astype(DTYPE)
        resp_drc = convolve(eigval, kernel_rc, mode='mirror').astype(DTYPE)
        resp_dcc = convolve(eigval, kernel_cc, mode='mirror').astype(DTYPE)

        # ── Seed list sorted descending by eigenvalue (= original cross.sort()) ──
        sy, sx = np.where(self.ismax >= 2)
        seed_vals = self.eigval[sy, sx]
        sort_order = np.argsort(seed_vals)[::-1]   # descending
        sy = sy[sort_order].astype(np.int32)
        sx = sx[sort_order].astype(np.int32)
        seed_vals = seed_vals[sort_order]
        n_seeds = len(sy)

        # ── indx: pixel → 1-based position in seed array ─────────────────────
        indx = np.zeros((height, width), dtype=np.int32)
        done = np.zeros(n_seeds, dtype=np.bool_)
        indx[sy, sx] = np.arange(1, n_seeds + 1, dtype=np.int32)
        label = np.zeros((height, width), dtype=np.int32)

        self.contours, self.junctions = [], []

        # ── Contiguous float32 arrays for Numba ───────────────────────────────
        normy    = np.ascontiguousarray(self.normy.astype(DTYPE))
        normx    = np.ascontiguousarray(self.normx.astype(DTYPE))
        posy     = np.ascontiguousarray(self.posy.astype(DTYPE))
        posx     = np.ascontiguousarray(self.posx.astype(DTYPE))
        ismax_a  = np.ascontiguousarray(self.ismax.astype(np.int32))
        dirtab_a = np.ascontiguousarray(dirtab.astype(np.int32))
        cltab_a  = np.ascontiguousarray(cleartab.astype(np.int32))

        # Pre-allocate trace buffers (reused every call)
        out_row   = np.empty(MAX_TRACE_POINTS, dtype=DTYPE)
        out_col   = np.empty(MAX_TRACE_POINTS, dtype=DTYPE)
        out_angle = np.empty(MAX_TRACE_POINTS, dtype=DTYPE)
        out_resp  = np.empty(MAX_TRACE_POINTS, dtype=DTYPE)

        num_cont = 0

        for seed_idx in range(n_seeds):
            if done[seed_idx]:
                continue

            maxy = int(sy[seed_idx])
            maxx = int(sx[seed_idx])

            if seed_vals[seed_idx] == 0.0:
                break           # all remaining seeds are zero (same as original)

            if label[maxy, maxx] > 0:
                continue        # already claimed by another contour

            # ── Seed initialization ───────────────────────────────────────────
            ny_v = float(self.normy[maxy, maxx])
            nx_v = -float(self.normx[maxy, maxx])
            alpha = math.atan2(ny_v, nx_v)
            if alpha < 0.0:   alpha += math.pi
            if alpha >= math.pi: alpha -= math.pi
            octant_seed    = int(math.floor(4.0 / math.pi * alpha + 0.5)) % 4
            last_beta_seed = alpha + math.pi / 2.0
            if last_beta_seed >= 2.0 * math.pi:
                last_beta_seed -= 2.0 * math.pi

            # Seed row/col: integer (matches original row.append(maxy))
            seed_r = float(maxy)
            seed_c = float(maxx)
            # Seed angle = alpha + pi/2 (matches original angle.append(beta))
            seed_angle = last_beta_seed
            # Seed response: offset from integer pixel
            yy_seed = float(self.posy[maxy, maxx]) - maxy
            xx_seed = float(self.posx[maxy, maxx]) - maxx
            seed_resp_val = (float(resp_dd[maxy, maxx])
                             + yy_seed  * float(resp_dr[maxy, maxx])
                             + xx_seed  * float(resp_dc[maxy, maxx])
                             + yy_seed**2 * float(resp_drr[maxy, maxx])
                             + xx_seed * yy_seed * float(resp_drc[maxy, maxx])
                             + xx_seed**2 * float(resp_dcc[maxy, maxx]))

            label[maxy, maxx] = num_cont + 1
            ci = indx[maxy, maxx]
            if ci != 0:
                done[ci - 1] = True

            # ── Trace both directions ─────────────────────────────────────────
            paths = []
            for it in range(2):
                last_octant = octant_seed if it == 0 else octant_seed + 4
                n_pts, hit_junc, jy, jx = _trace_direction(
                    ismax_a, normy, normx, posy, posx,
                    label, indx, done,
                    resp_dd, resp_dr, resp_dc, resp_drr, resp_drc, resp_dcc,
                    dirtab_a, cltab_a,
                    maxy, maxx,
                    last_octant, last_beta_seed,
                    num_cont,
                    height, width,
                    out_row, out_col, out_angle, out_resp
                )
                paths.append((out_row[:n_pts].copy(), out_col[:n_pts].copy(),
                              out_angle[:n_pts].copy(), out_resp[:n_pts].copy(),
                              hit_junc, int(jy), int(jx)))

            # ── Assemble: [dir1 reversed] + [seed] + [dir2] ──────────────────
            r1, c1, a1, s1 = paths[0][0][::-1], paths[0][1][::-1], paths[0][2][::-1], paths[0][3][::-1]
            r2, c2, a2, s2 = paths[1][0],        paths[1][1],        paths[1][2],        paths[1][3]

            all_rows  = np.concatenate([r1, [seed_r],         r2]).astype(float)
            all_cols  = np.concatenate([c1, [seed_c],         c2]).astype(float)
            all_angs  = np.concatenate([a1, [seed_angle],     a2]).astype(float)
            all_resps = np.concatenate([s1, [seed_resp_val],  s2]).astype(float)
            num_pnt   = len(all_rows)

            if num_pnt <= 1:
                # Single point: clear its label (matches original cleanup)
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        r_ = LinesUtil.BR(maxy + di, height)
                        c_ = LinesUtil.BC(maxx + dj, width)
                        if label[r_, c_] == num_cont + 1:
                            label[r_, c_] = 0
                continue

            # ── Contour class ─────────────────────────────────────────────────
            hit1, hit2 = paths[0][4], paths[1][4]
            if hit1 and hit2:
                cls = LinesUtil.ContourClass.cont_both_junc
            elif hit1:
                cls = LinesUtil.ContourClass.cont_start_junc
            elif hit2:
                cls = LinesUtil.ContourClass.cont_end_junc
            else:
                # Check for closed contour
                if (num_pnt > 2 and abs(all_rows[0] - all_rows[-1]) < 1.5
                        and abs(all_cols[0] - all_cols[-1]) < 1.5):
                    cls = LinesUtil.ContourClass.cont_closed
                else:
                    cls = LinesUtil.ContourClass.cont_no_junc

            # ── Build Line object ─────────────────────────────────────────────
            new_line = Line()
            new_line.row       = all_rows
            new_line.col       = all_cols
            new_line.angle     = all_angs
            new_line.response  = all_resps
            new_line.width_r   = None
            new_line.width_l   = None
            new_line.asymmetry = None
            new_line.intensity = None
            new_line.num       = num_pnt
            new_line.set_contour_class(cls)
            self.contours.append(new_line)

            # ── Junction registration ─────────────────────────────────────────
            for it_j, path in enumerate(paths):
                if not path[4]:   # hit_junc
                    continue
                jy_p, jx_p = path[5], path[6]
                k = label[jy_p, jx_p] - 1
                if k < 0 or k == num_cont or k >= len(self.contours):
                    continue
                cont_k = self.contours[k]
                jpy = float(posy[jy_p, jx_p])
                jpx = float(posx[jy_p, jx_p])
                # Find exact position on cont_k
                j = -1
                for jj in range(cont_k.num):
                    if cont_k.row[jj] == jpy and cont_k.col[jj] == jpx:
                        j = jj
                        break
                if j < 0:
                    dists = (cont_k.row - jpy)**2 + (cont_k.col - jpx)**2
                    j = int(np.argmin(dists))
                if 0 < j < cont_k.num - 1:
                    self.junctions.append(Junction(k, num_cont, j, jpy, jpx))

            num_cont += 1

        # ── Adjust angles to point right of line (same as original) ──────────
        for i in range(num_cont):
            tmp_cont = self.contours[i]
            n        = tmp_cont.num
            if n > 1:
                k  = (n - 1) // 2
                dy = tmp_cont.row[k + 1] - tmp_cont.row[k]
                dx = tmp_cont.col[k + 1] - tmp_cont.col[k]
                ny_v = math.sin(tmp_cont.angle[k])
                nx_v = math.cos(tmp_cont.angle[k])
                if ny_v * dx - nx_v * dy < 0:
                    tmp_cont.angle = np.array(
                        [(ang + math.pi) % (2 * math.pi) for ang in tmp_cont.angle]
                    )
