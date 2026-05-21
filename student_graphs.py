# ============================================================
# 12. IMPORTS FOR EVALUATION
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

# ============================================================
# 13. STORE TRAINING HISTORY
# ============================================================

train_losses = []
train_accuracies = []
val_accuracies = []

# ============================================================
# MODIFY YOUR TRAINING LOOP
# ============================================================

EPOCHS = 50

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for images, labels in cxr_train:

        loss = train_step(images, labels)

    train_acc = train_acc_metric.result()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    for images, labels in cxr_val:

        val_step(images, labels)

    val_acc = val_acc_metric.result()

    # --------------------------------------------------------
    # STORE HISTORY
    # --------------------------------------------------------

    train_losses.append(float(loss))
    train_accuracies.append(float(train_acc))
    val_accuracies.append(float(val_acc))

    print(
        f"Train Loss: {loss:.4f} | "
        f"Train Accuracy: {train_acc:.4f} | "
        f"Validation Accuracy: {val_acc:.4f}"
    )

    # Reset metrics
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()

# ============================================================
# 14. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8,6))

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# 15. LOSS GRAPH
# ============================================================

plt.figure(figsize=(8,6))

plt.plot(train_losses, label='Training Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# 16. PREDICTIONS
# ============================================================

y_true = []
y_pred = []
y_prob = []

for images, labels in cxr_val:

    _, probs, embeddings = student.predict(images, verbose=0)

    preds = np.argmax(probs, axis=1)

    y_true.extend(labels)
    y_pred.extend(preds)
    y_prob.extend(probs)

    # stop duplication
    if len(y_true) >= cxr_val.samples:
        break

y_true = np.array(y_true[:cxr_val.samples])
y_pred = np.array(y_pred[:cxr_val.samples])
y_prob = np.array(y_prob[:cxr_val.samples])

# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

class_names = list(cxr_val.class_indices.keys())

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
# 18. CLASSIFICATION REPORT
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
# 19. ROC CURVE
# ============================================================

# Convert labels to one-hot
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
# 20. t-SNE VISUALIZATION
# ============================================================

# Extract embeddings
all_embeddings = []
all_labels = []

for images, labels in cxr_val:

    _, _, embeddings = student.predict(images, verbose=0)

    all_embeddings.append(embeddings)
    all_labels.extend(labels)

    if len(all_labels) >= cxr_val.samples:
        break

all_embeddings = np.concatenate(all_embeddings, axis=0)
all_embeddings = all_embeddings[:cxr_val.samples]

all_labels = np.array(all_labels[:cxr_val.samples])

# ============================================================
# t-SNE REDUCTION
# ============================================================

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    random_state=42
)

tsne_features = tsne.fit_transform(all_embeddings)

# ============================================================
# PLOT t-SNE
# ============================================================

plt.figure(figsize=(8,6))

for i, class_name in enumerate(class_names):

    idx = all_labels == i

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
