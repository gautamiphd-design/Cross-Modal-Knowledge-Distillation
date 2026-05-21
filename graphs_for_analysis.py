import matplotlib.pyplot as plt

# Assuming 'history' object is available from the previous training run in cell uOVqRwA6GTem
if 'history' in locals():
    plt.figure(figsize=(12, 5))

    # Plot Training & Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Error: 'history' object not found. Please ensure the model training cell was run successfully.")
  import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize

# Ensure GaborLayer is defined or imported, assuming it's already defined with
# @tf.keras.utils.register_keras_serializable() in a previous cell like 3a688eed.
# If not, you would need to include its definition here or ensure the cell defining it is run.
# For this fix, we assume the GaborLayer class from cell 3a688eed is available globally.

# Re-define ct_val for CXR validation data to ensure it's available
# This assumes the necessary path and IMG_SIZE/BATCH_SIZE constants are available
# from other cells (e.g., ECFJuIK85Sfo or uOVqRwA6GTem)
# For a self-contained fix, you might need to explicitly define them here.

# If IMG_SIZE, BATCH_SIZE, val_path are not defined globally, add them here
IMG_SIZE = (224,224)
BATCH_SIZE = 32
val_path   = "/content/drive/MyDrive/Dataset/CXR/val"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

ct_val = datagen.flow_from_directory(
    val_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)
num_classes = ct_val.num_classes
class_names = list(ct_val.class_indices.keys())


# -------------------------------------------------
# LOAD SAVED MODEL
# -------------------------------------------------
model_path = "/content/drive/MyDrive/saved_models_MACAI/ct_teacher_resnet50.keras"
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"GaborLayer": GaborLayer},
    compile=False # Set compile=False if you only need inference and not retraining
)

print("Model loaded successfully.")

# -------------------------------------------------
# GENERATE PREDICTIONS ON TEST SET (using ct_val as the validation dataset)
# -------------------------------------------------
ct_val.reset() # Reset generator to ensure fresh pass
y_pred_probs = model.predict(ct_val)
y_true = ct_val.classes

# num_classes and class_names are already defined from ct_val above

# -------------------------------------------------
# ONE-HOT ENCODING
# -------------------------------------------------
y_true_one_hot = label_binarize(y_true, classes=range(num_classes))

# -------------------------------------------------
# ROC COMPUTATION
# -------------------------------------------------
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute Micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_one_hot.ravel(), y_pred_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute Macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# -------------------------------------------------
# PLOT ROC CURVES
# -------------------------------------------------
plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green'])

for i, color in zip(range(num_classes), colors):
    plt.plot(
        fpr[i], tpr[i],
        color=color, lw=2,
        label=f'ROC of {class_names[i]} (AUC = {roc_auc[i]:0.4f})'
    )

plt.plot(
    fpr["micro"], tpr["micro"],
    linestyle='--', color='black', lw=2,
    label=f'Micro-average ROC (AUC = {roc_auc["micro"]:0.4f})'
)

plt.plot(
    fpr["macro"], tpr["macro"],
    linestyle=':', color='orange', lw=2,
    label=f'Macro-average ROC (AUC = {roc_auc["macro"]:0.4f})'
)

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – CXR Student Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# -------------------------------------------------
# PRINT AUC VALUES
# -------------------------------------------------
print("\nPer-Class AUC Scores:")
for i in range(num_classes):
    print(f"{class_names[i]}: {roc_auc[i]:.4f}")

print(f"\nMicro-average AUC: {roc_auc['micro']:.4f}")
print(f"Macro-average AUC: {roc_auc['macro']:.4f}")

# -------------------------------------------------
# t-SNE graph
# -------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf

# Ensure GaborLayer is defined for model loading
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
            y, x = np.meshgrid(np.arange(-half, half+1), np.arange(-half, half+1))
            x_theta = x*np.cos(theta) + y*np.sin(theta)
            y_theta = -x*np.sin(theta) + y*np.cos(theta)
            gabor = np.exp(-(x_theta**2 + gamma**2*y_theta**2) / (2*sigma**2)) * np.cos(2*np.pi*x_theta/lambd)
            gabor = gabor[..., np.newaxis, np.newaxis]
            kernels.append(gabor)
        kernels = np.concatenate(kernels, axis=-1)
        kernels = np.repeat(kernels, input_shape[-1], axis=2)
        self.kernels = tf.constant(kernels, dtype=tf.float32)
    def call(self, inputs):
        return tf.nn.conv2d(inputs, self.kernels, strides=1, padding="SAME")
    def get_config(self):
        return super().get_config()


# Assuming ct_teacher model and ct_val generator are already available
# and that ct_teacher has an output layer named 'Z_CT'

# Create a new model that outputs only the Z_CT embedding
embedding_model = Model(inputs=ct_teacher.input, outputs=ct_teacher.get_layer('Z_CT').output)

print("Extracting embeddings...")

# Extract embeddings for the validation set
ct_val.reset() # Reset generator to ensure fresh pass
Z_CT_embeddings = embedding_model.predict(ct_val)

print(f"Extracted embeddings shape: {Z_CT_embeddings.shape}")

# Get true labels from ct_val generator
y_true_val = ct_val.classes

# Apply t-SNE
print("Applying t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
Z_CT_tsne = tsne.fit_transform(Z_CT_embeddings)

print("t-SNE completed. Plotting results...")

# Get class names from the generator
class_names = list(ct_val.class_indices.keys())

# Create a DataFrame for plotting
import pandas as pd
tsne_df = pd.DataFrame({'TSNE-1': Z_CT_tsne[:, 0], 'TSNE-2': Z_CT_tsne[:, 1], 'Label': y_true_val})

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='TSNE-1', y='TSNE-2',
    hue='Label',
    palette=sns.color_palette("hls", len(class_names)),
    data=tsne_df,
    legend='full',
    alpha=0.7
)
plt.title('t-SNE Visualization of CXR Teacher Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Class', labels=class_names)
plt.grid(True)
plt.show()
