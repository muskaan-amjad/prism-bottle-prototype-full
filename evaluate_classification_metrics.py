import numpy as np
import pandas as pd
from PIL import Image
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Paths (update if needed)
MODEL_PATH = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/models/best_model.h5"
VAL_LABELS_CSV = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data/val_labels.csv"
VAL_IMAGES_DIR = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data/val"

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load validation labels
val_labels = pd.read_csv(VAL_LABELS_CSV)
val_labels['image_filename'] = val_labels['image_filename'].str.strip().str.lower()
val_labels.set_index('image_filename', inplace=True)

def preprocess_image(path):
    image = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(image).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)

def binary_classify(scores, threshold=5.0):
    return (scores >= threshold).astype(int)

# Store true and predicted scores
y_true = []
y_pred = []

for img_name in val_labels.index:
    img_path = os.path.join(VAL_IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"Skipping missing image: {img_name}")
        continue
    img_arr = preprocess_image(img_path)
    pred = model.predict(img_arr)[0]
    true_vals = val_labels.loc[img_name, ['durability_score', 'chemical_safety_score', 'ergonomics_score']].values.astype(float)
    
    y_pred.append(pred)
    y_true.append(true_vals)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

metrics_names = ['Durability', 'Chemical Safety', 'Ergonomics']
threshold = 5.0  # You can adjust this threshold

for i, name in enumerate(metrics_names):
    true_binary = binary_classify(y_true[:, i], threshold)
    pred_binary = binary_classify(y_pred[:, i], threshold)

    tn, fp, fn, tp = confusion_matrix(true_binary, pred_binary).ravel()
    accuracy = accuracy_score(true_binary, pred_binary)
    precision = precision_score(true_binary, pred_binary, zero_division=0)
    recall = recall_score(true_binary, pred_binary, zero_division=0)

    print(f"--- {name} ---")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")
