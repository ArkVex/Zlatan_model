"""Shared, framework-agnostic metrics so both trainers are judged identically."""
import numpy as np

# Shared hyperparameters — identical for both frameworks (no bias).
HP = dict(
    conv1=32, conv2=64, gru=64, kernel=5,
    lr=1e-3, batch=256, epochs=12, seed=42,
)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, np.float64).ravel()
    y_pred = np.asarray(y_pred, np.float64).ravel()
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mean_speed = float(np.mean(np.abs(y_true)))
    return dict(
        mae_ms=round(mae, 4),
        rmse_ms=round(rmse, 4),
        r2=round(r2, 4),
        mean_speed_ms=round(mean_speed, 4),
        pct_error=round(100 * mae / mean_speed, 2) if mean_speed > 0 else None,
    )
