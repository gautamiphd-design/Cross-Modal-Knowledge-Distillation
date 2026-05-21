# ============================================================
# 1. IMPORTS
# ============================================================

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

print("TensorFlow Version:", tf.__version__)

# ============================================================
# 2. DATA GENERATORS
# ============================================================

IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 50

train_path = "/content/drive/MyDrive/Dataset/CXR/train"
val_path   = "/content/drive/MyDrive/Dataset/CXR/val"

datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    val_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

num_classes = train_data.num_classes

class_names = list(train_data.class_indices.keys())

print("Classes:", class_names)

# ============================================================
# 3. BUILD SIMPLE CXR MODEL
# ============================================================

base_model = MobileNetV3Small(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3)
)

base_model.trainable = True

inputs = layers.Input(shape=(224,224,3))

x = base_model(inputs)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(
    256,
    activation='relu'
)(x)

x = layers.Dropout(0.3)(x)

# Feature embedding
embedding = layers.Activation(
    "linear",
    name="embedding"
)(x)

outputs = layers.Dense(
    num_classes,
    activation='softmax'
)(embedding)

model = models.Model(
    inputs,
    outputs
)

model.summary()

# ============================================================
# 4. COMPILE MODEL
# ============================================================

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-4,
    weight_decay=1e-5
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# 5. TRAIN MODEL
# ============================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ============================================================
# 6. SAVE MODEL
# ============================================================

save_dir = "/content/drive/MyDrive/saved_models_MACAI"

os.makedirs(save_dir, exist_ok=True)

model.save(
    os.path.join(save_dir, "simple_cxr_model.keras")
)

print("Simple CXR model saved successfully")

# ============================================================
# 7. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8,6))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# 8. LOSS GRAPH
# ============================================================

plt.figure(figsize=(8,6))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# 9. PREDICTIONS
# ============================================================

y_true = []
y_pred = []
y_prob = []

for images, labels in val_data:

    probs = model.predict(images, verbose=0)

    preds = np.argmax(probs, axis=1)

    y_true.extend(labels)
    y_pred.extend(preds)
    y_prob.extend(probs)

    if len(y_true) >= val_data.samples:
        break

y_true = np.array(y_true[:val_data.samples])
y_pred = np.array(y_pred[:val_data.samples])
y_prob = np.array(y_prob[:val_data.samples])

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()

# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# ============================================================
# 12. ROC CURVE
# ============================================================

y_true_bin = label_binarize(
    y_true,
    classes=np.arange(num_classes)
)

plt.figure(figsize=(8,6))

for i in range(num_classes):

    fpr, tpr, _ = roc_curve(
        y_true_bin[:, i],
        y_prob[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f'{class_names[i]} (AUC = {roc_auc:.4f})'
    )

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# 13. t-SNE VISUALIZATION
# ============================================================

# Feature extractor model
feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("embedding").output
)

embeddings = []
labels_all = []

for images, labels in val_data:

    feat = feature_model.predict(images, verbose=0)

    embeddings.append(feat)

    labels_all.extend(labels)

    if len(labels_all) >= val_data.samples:
        break

embeddings = np.concatenate(embeddings, axis=0)
embeddings = embeddings[:val_data.samples]

labels_all = np.array(labels_all[:val_data.samples])

# ============================================================
# t-SNE
# ============================================================

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    random_state=42
)

tsne_features = tsne.fit_transform(embeddings)

# ============================================================
# PLOT t-SNE
# ============================================================

plt.figure(figsize=(8,6))

for i, class_name in enumerate(class_names):

    idx = labels_all == i

    plt.scatter(
        tsne_features[idx, 0],
        tsne_features[idx, 1],
        label=class_name,
        alpha=0.7
    )

plt.title("t-SNE Feature Visualization")

plt.legend()

plt.grid(True)

plt.show()
