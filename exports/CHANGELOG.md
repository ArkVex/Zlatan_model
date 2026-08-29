# Model export changelog

## tcn_v1_2026-08-29_419ee90
- First real TFLite export for app integration (was stub-only before).
- Architecture: TCN (dilated causal 1-D convs), ~115k params, builtins-only (no SELECT_TF_OPS).
- Input `imu_window` [1,50,7] f32 (7 rotation-invariant IMU features @10Hz); output `speed` [1,1] f32 (m/s).
- Speed accuracy (held-out): MAE ~4.1 m/s, R^2 ~0.66. Along-track distance drift median ~12%.
- Ships as a matched triple: this .tflite + norm.json + manifest.json (same model_version).
- G1 export parity (keras vs tflite, Python): max abs diff 5.7e-6 -> PASS.

## tcn_v2_2026-08-30_ft1
- Retrained with "not-driving" negatives (synthetic stationary + recorded hand-held trips), labelled 0 m/s.
- Fixes the out-of-domain rocket: hand-held motion now predicts ~0 (was ~18 km/h; replay on recorded trip: 28m phantom -> 0m).
- Driving accuracy held: MAE 4.41 m/s, R2 0.611 (base v1: 4.06 / 0.66).
- Same interface [1,50,7]->[1,1] builtins-only. New norm.json + manifest.json (matched triple).
