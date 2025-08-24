import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Function to preprocess images
def preprocess_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)

# Prepare lists to store predictions and true values
y_true = []
y_pred = []

# Predict and collect for each validation image
for img_name in val_labels.index:
    img_path = os.path.join(VAL_IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"Image {img_name} not found, skipping.")
        continue
    img_arr = preprocess_image(img_path)
    pred = model.predict(img_arr)[0]
    y_pred.append(pred)
    true_values = val_labels.loc[img_name, ['durability_score', 'chemical_safety_score', 'ergonomics_score']].values.astype(float)
    y_true.append(true_values)

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate metrics for each metric column
mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred, multioutput='raw_values')

metrics_names = ['Durability', 'Chemical Safety', 'Ergonomics']

# Display results
for i, name in enumerate(metrics_names):
    print(f"{name} - MAE: {mae[i]:.4f}, RMSE: {rmse[i]:.4f}, R²: {r2[i]:.4f}")
