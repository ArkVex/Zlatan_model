"""Create engine_stub.tflite with the FINAL interface but trivial internals.

Lets the Android team build and parity-test the whole pipeline (sensor -> preprocess ->
model call -> output) against the real [1,50,7]->[1,1] interface before the real model
exists. The stub ignores its input and returns a constant speed.
"""
import os
import numpy as np
import tensorflow as tf

WIN, C = 50, 7
CONST_SPEED = 8.33   # m/s == 30 km/h


from paths import EXPORTS


def main():
    inp = tf.keras.Input(shape=(WIN, C), name="imu_window")
    # Multiply by ~0 then add the constant, so the graph genuinely uses the input shape.
    x = tf.keras.layers.GlobalAveragePooling1D()(inp)
    x = tf.keras.layers.Dense(1, name="speed",
                              kernel_initializer="zeros",
                              bias_initializer=tf.keras.initializers.Constant(CONST_SPEED))(x)
    model = tf.keras.Model(inp, x)

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tfl = conv.convert()
    with open(os.path.join(EXPORTS, "engine_stub.tflite"), "wb") as f:
        f.write(tfl)

    # Verify interface.
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(di["index"], np.random.randn(1, WIN, C).astype(np.float32))
    it.invoke()
    out = it.get_tensor(do["index"])
    print(f"stub in {di['shape']} {di['dtype']} -> out {do['shape']} = {out.ravel()[0]:.2f} m/s")
    print("wrote", os.path.join(EXPORTS, "engine_stub.tflite"))


if __name__ == "__main__":
    main()
