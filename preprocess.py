"""Standalone preprocessing reference — the spec the Kotlin app must match bit-for-bit.

No training dependencies. Turns raw phone IMU (accelerometer, gravity, gyroscope) into
the model-ready [WIN, 7] feature window described in the integration contract (A3/A4/A5).

Channel order (must match norm.json and the .tflite input):
  0 a_horiz   1 a_vert   2 a_lin_mag   3 gyr_y   4 gyr_p   5 gyr_r   6 gyro_mag
"""
import numpy as np

WIN = 50            # samples (5.0 s @ 10 Hz)
SAMPLE_RATE_HZ = 10
CHANNELS = ["a_horiz", "a_vert", "a_lin_mag", "gyr_y", "gyr_p", "gyr_r", "gyro_mag"]


def compute_features(accel, gravity, gyro):
    """Per-timestep 7 features from raw sensors.

    accel, gravity, gyro: arrays shape [T, 3] in Android units
      accel   m/s^2 (TYPE_ACCELEROMETER, gravity still present)
      gravity m/s^2 (TYPE_GRAVITY)
      gyro    rad/s (TYPE_GYROSCOPE), order [yaw, pitch, roll]
    Returns [T, 7] float32.
    """
    accel = np.asarray(accel, np.float64)
    gravity = np.asarray(gravity, np.float64)
    gyro = np.asarray(gyro, np.float64)

    a_lin = accel - gravity
    g_hat = gravity / (np.linalg.norm(gravity, axis=1, keepdims=True) + 1e-6)
    a_vert = np.sum(a_lin * g_hat, axis=1)
    a_horiz = np.linalg.norm(a_lin - a_vert[:, None] * g_hat, axis=1)
    a_lin_mag = np.linalg.norm(a_lin, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    return np.stack([a_horiz, a_vert, a_lin_mag,
                     gyro[:, 0], gyro[:, 1], gyro[:, 2], gyro_mag], axis=1).astype(np.float32)


def normalize(features, mean, std):
    """Per-channel standardization. mean/std are length-7 arrays from norm.json."""
    return ((features - np.asarray(mean)) / np.asarray(std)).astype(np.float32)


def preprocess(accel, gravity, gyro, mean=None, std=None):
    """Take the most recent WIN samples and return the model input.

    Returns [WIN, 7] (normalized if mean/std given, else raw features).
    Assumes inputs are already resampled to SAMPLE_RATE_HZ (the app downsamples first).
    """
    feats = compute_features(accel, gravity, gyro)
    if len(feats) < WIN:
        raise ValueError(f"need >= {WIN} samples, got {len(feats)}")
    window = feats[-WIN:]
    if mean is not None and std is not None:
        window = normalize(window, mean, std)
    return window


if __name__ == "__main__":
    # Smoke test with fake data.
    T = 60
    accel = np.random.randn(T, 3) + np.array([0, 0, 9.8])
    gravity = np.tile([0, 0, 9.8], (T, 1))
    gyro = np.random.randn(T, 3) * 0.1
    out = preprocess(accel, gravity, gyro)
    print("preprocess ->", out.shape, "channels:", CHANNELS)
