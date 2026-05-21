# ============================================================
# 1. MOUNT DRIVE
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# 2. IMPORTS
# ============================================================

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Eager mode:", tf.executing_eagerly())

# ============================================================
# 3. DATA GENERATORS
# ============================================================

IMG_SIZE = (224,224)
BATCH_SIZE = 32

ct_train_path = "/content/drive/MyDrive/Dataset/CT/train"
ct_val_path   = "/content/drive/MyDrive/Dataset/CT/test"

datagen = ImageDataGenerator(rescale=1./255)

ct_train = datagen.flow_from_directory(
    ct_train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True
)

ct_val = datagen.flow_from_directory(
    ct_val_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

num_classes = ct_train.num_classes
print("Classes:", ct_train.class_indices)

# ============================================================
# 4. DEFINE GABOR LAYER
# ============================================================

class GaborLayer(layers.Layer):

    def build(self, input_shape):

        ksize = 11
        sigma = 3.0
        lambd = 6.0
        gamma = 0.5
        thetas = [0, np.pi/4, np.pi/2]

        kernels = []
        half = ksize // 2

        for theta in thetas:

            y, x = np.meshgrid(
                np.arange(-half, half+1),
                np.arange(-half, half+1)
            )

            x_theta = x*np.cos(theta) + y*np.sin(theta)
            y_theta = -x*np.sin(theta) + y*np.cos(theta)

            gabor = np.exp(
                -(x_theta**2 + gamma**2*y_theta**2) /
                (2*sigma**2)
            ) * np.cos(2*np.pi*x_theta/lambd)

            gabor = gabor[..., np.newaxis, np.newaxis]
            kernels.append(gabor)

        kernels = np.concatenate(kernels, axis=-1)
        kernels = np.repeat(kernels, input_shape[-1], axis=2)

        self.kernels = tf.constant(kernels, dtype=tf.float32)

    def call(self, inputs):
        return tf.nn.conv2d(inputs, self.kernels, strides=1, padding="SAME")

# ============================================================
# 5. BUILD TEACHER (TRAINING MODEL - SINGLE OUTPUT)
# ============================================================

base_ct = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

inputs = layers.Input(shape=(224,224,3))

# Texture preprocessing
x = GaborLayer(name="gabor_ct")(inputs)

# Channel projection for ResNet compatibility
x = layers.Conv2D(3, kernel_size=1, padding="same")(x)

# CNN encoder
x = base_ct(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

# Latent embedding
Z_CT = layers.Dense(256, activation="relu", name="Z_CT")(x)
Z_CT = layers.Dropout(0.5)(Z_CT)

# Classification output
p_CT = layers.Dense(num_classes, activation="softmax", name="p_CT")(Z_CT)

# TRAINING MODEL (only classification output)
ct_teacher_train = models.Model(inputs, p_CT)

ct_teacher_train.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-4, weight_decay=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

ct_teacher_train.summary()

# ============================================================
# 6. TRAIN TEACHER
# ============================================================

history = ct_teacher_train.fit(
    ct_train,
    validation_data=ct_val,
    epochs=5
)

# ============================================================
# 7. BUILD DISTILLATION MODEL (MULTI-OUTPUT)
# ============================================================

ct_teacher = models.Model(
    ct_teacher_train.input,
    [
        ct_teacher_train.get_layer("p_CT").output,
        ct_teacher_train.get_layer("Z_CT").output
    ]
)

ct_teacher.trainable = False

# ============================================================
# 8. SAVE TEACHER FOR DISTILLATION
# ============================================================

save_dir = "/content/drive/MyDrive/saved_models_MACAI"
os.makedirs(save_dir, exist_ok=True)

ct_teacher.save(
    os.path.join(save_dir, "ct_teacher.keras"),
    include_optimizer=False
)

print("CT Teacher saved successfully for distillation.")

