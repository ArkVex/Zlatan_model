"""Heading estimation: gyro + magnetometer complementary filter.

Problem (from Part C): gyro-only heading drifts ~18% over a 1 km blackout. The
magnetometer gives an absolute (drift-free) heading reference but is noisy and, inside a
vehicle, offset by a constant (phone-mount yaw + body distortion). We:

  1. Compute a per-sample magnetic heading in the DEVICE horizontal plane (tilt-compensated
     with the gravity vector). Its absolute zero is arbitrary but CONSTANT.
  2. Calibrate that constant offset against GPS course during GPS-good driving, so the mag
     heading becomes an absolute vehicle heading.
  3. Complementary-filter it with the gyro: gyro carries fast changes, the magnetometer
     slowly pulls the estimate back to absolute north so it never drifts.

All blending is done on unit vectors (cos/sin) to avoid angle-wrap bugs.
"""
import numpy as np

DT = 0.1


def magnetic_heading(grav, mag):
    """Per-sample heading (rad) in a device-fixed horizontal frame.

    grav, mag: [T,3]. Uses gravity to define 'up', projects mag to horizontal, and measures
    its angle against a device axis projected to horizontal. Absolute zero is arbitrary but
    constant across the drive (removed later by offset calibration).
    """
    up = grav / (np.linalg.norm(grav, axis=1, keepdims=True) + 1e-6)
    # horizontal component of the magnetic field
    m_h = mag - np.sum(mag * up, axis=1, keepdims=True) * up
    # a device reference axis projected onto the horizontal plane; fall back if degenerate
    refX = np.tile([1.0, 0.0, 0.0], (len(grav), 1))
    e1 = refX - np.sum(refX * up, axis=1, keepdims=True) * up
    bad = np.linalg.norm(e1, axis=1) < 1e-3
    if bad.any():
        refY = np.tile([0.0, 1.0, 0.0], (len(grav), 1))
        e1[bad] = (refY - np.sum(refY * up, axis=1, keepdims=True) * up)[bad]
    e1 /= (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-6)
    e2 = np.cross(up, e1)
    return np.arctan2(np.sum(m_h * e2, axis=1), np.sum(m_h * e1, axis=1))


def circular_offset(true_ang, meas_ang, mask):
    """Constant offset o minimizing wrap(true - (meas + o)), via circular mean of the diff."""
    d = true_ang[mask] - meas_ang[mask]
    return float(np.arctan2(np.nanmean(np.sin(d)), np.nanmean(np.cos(d))))


def complementary_heading(h0, gyro_yaw, k, mag_head_abs, alpha=0.98):
    """Fuse gyro (rate) with an absolute magnetic heading over one segment.

    h0: seed heading (rad). gyro_yaw: [n] rad/s. k: gyro->heading gain (calibrated).
    mag_head_abs: [n] absolute magnetic heading (rad), already offset-corrected.
    alpha near 1 trusts the gyro short-term; (1-alpha) slowly corrects toward magnetometer.
    Returns array of fused headings [n].
    """
    h = h0
    out = np.empty(len(gyro_yaw))
    for i in range(len(gyro_yaw)):
        h_gyro = h + k * gyro_yaw[i] * DT
        if np.isfinite(mag_head_abs[i]):
            # blend on the unit circle to avoid wrap issues
            s = alpha * np.sin(h_gyro) + (1 - alpha) * np.sin(mag_head_abs[i])
            c = alpha * np.cos(h_gyro) + (1 - alpha) * np.cos(mag_head_abs[i])
            h = np.arctan2(s, c)
        else:
            h = h_gyro
        out[i] = h
    return out
