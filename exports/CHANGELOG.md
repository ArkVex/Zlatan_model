# Model export changelog

## tcn_v1_2026-08-29_419ee90
- First real TFLite export for app integration (was stub-only before).
- Architecture: TCN (dilated causal 1-D convs), ~115k params, builtins-only (no SELECT_TF_OPS).
- Input `imu_window` [1,50,7] f32 (7 rotation-invariant IMU features @10Hz); output `speed` [1,1] f32 (m/s).
- Speed accuracy (held-out): MAE ~4.1 m/s, R^2 ~0.66. Along-track distance drift median ~12%.
- Ships as a matched triple: this .tflite + norm.json + manifest.json (same model_version).
- G1 export parity (keras vs tflite, Python): max abs diff 5.7e-6 -> PASS.
