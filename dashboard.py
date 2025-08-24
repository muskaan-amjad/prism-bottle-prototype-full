import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all,1=filter INFO,2=filter WARNING,3=filter ERROR
import streamlit as st
import tensorflow as tf
keras = tf.keras  # pyright: ignore[reportAttributeAccessIssue]
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path

# === Config paths (relative to this file) ===
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.h5"
VAL_LABELS_CSV = BASE_DIR / "data" / "val_labels.csv"
VAL_IMAGES_DIR = BASE_DIR / "data" / "val"

# === LOAD MODEL & LABELS ===
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(str(MODEL_PATH))
    model = keras.models.load_model(str(MODEL_PATH), compile=False)
    # NOTE: Original model task likely regression for 3 scores; keep simple compile for predict usage.
    try:
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    except Exception:
        # If model already compiled or incompatible just ignore.
        pass
    return model

@st.cache_data
def load_val_labels():
    if not VAL_LABELS_CSV.exists():
        st.error(f"Validation labels CSV not found: {VAL_LABELS_CSV}")
        return pd.DataFrame(columns=['image_filename','durability_score','chemical_safety_score','ergonomics_score'])
    df = pd.read_csv(VAL_LABELS_CSV)
    # Normalize filenames in CSV index: strip whitespace and lowercase
    df['image_filename'] = df['image_filename'].str.strip().str.lower()
    df.set_index('image_filename', inplace=True)
    return df

model = load_model()
val_labels = load_val_labels()

def feedback(score):
    if score >= 8:
        return "Excellent"
    elif score >= 6:
        return "Good"
    elif score >= 4:
        return "Average"
    else:
        return "Needs Improvement"

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((224, 224))
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return image, arr

def predict_scores(arr):
    pred = model.predict(arr, verbose=0)
    return pred[0]

def display_comparison(image_name, pred_scores):
    normalized_name = image_name.strip().lower()
    if normalized_name in val_labels.index:
        true_scores = val_labels.loc[normalized_name]
        # Explicit float conversion to avoid pandas series/potential dtype issues
        actual = [
            float(true_scores['durability_score']),
            float(true_scores['chemical_safety_score']),
            float(true_scores['ergonomics_score'])
        ]
    else:
        actual = ["N/A"] * 3
        st.warning(f"No ground truth available for {image_name}")

    data = {
        "Metric": ["Durability", "Chemical Safety", "Ergonomics"],
        "Predicted Score": [round(float(x), 2) for x in pred_scores],
        "AI Feedback": [feedback(float(x)) for x in pred_scores],
        "Actual Score": actual
    }
    df = pd.DataFrame(data)
    st.table(df)

# === SIDEBAR NAVIGATION ===
st.sidebar.markdown("### Navigation")
menu = st.sidebar.radio(
    "Go to",
    [
        "Upload & Predict",
        "Compare Bottles",
    "Evaluation Metrics",
        "ChatGPT Assistant",
        "Feedback Form"
    ]
)

# === MAIN HEADER & DESCRIPTION ===
if menu == "Upload & Predict":
    st.markdown(
        "<h1 style='text-align: center; color: #e60073;'>AI Powered Product Analysis Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style='text-align: center; font-size:1.1em'>
    <b>Upload a <span style='color:#1976D2'>bottle image</span></b> and receive predictions for:
    <ul style='list-style:disc; margin-left:2em; text-align:left;'>
      <li>Master Category</li>
      <li>Subtype</li>
      <li>Morphological Features</li>
      <li>Functional Factors</li>
      <li>Real World Usage Traits</li>
    </ul>
    <i>Use the integrated <b>AI assistant</b> for expert insights, and download a PDF report.</i>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload bottle image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        filename_norm = uploaded_file.name.strip().lower()
        image, arr = preprocess_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        pred_scores = predict_scores(arr)
        display_comparison(filename_norm, pred_scores)

elif menu == "Compare Bottles":
    st.markdown("<h2>Compare Bottles</h2>", unsafe_allow_html=True)
    if not VAL_IMAGES_DIR.exists():
        st.error(f"Validation images directory not found: {VAL_IMAGES_DIR}")
        st.stop()
    image_files = [f for f in os.listdir(VAL_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        st.warning("No images found in validation directory.")
        st.stop()

    img1 = st.selectbox("Select First Bottle Image", image_files)
    img2 = st.selectbox("Select Second Bottle Image", image_files, index=1 if len(image_files) > 1 else 0)

    col1, col2 = st.columns(2)

    def load_and_predict(image_name):
        # Map normalized name back to actual filename (case insensitive)
        matched_file = next((f for f in image_files if f.strip().lower() == image_name.strip().lower()), None)
        if matched_file is None:
            st.error(f"Cannot find image file {image_name}")
            return None, [0, 0, 0]

        image_path = os.path.join(VAL_IMAGES_DIR, matched_file)
        image = Image.open(image_path).convert("RGB")
        img_resized = image.resize((224, 224))
        arr = np.array(img_resized).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = model.predict(arr)
        return image, pred[0]

    with col1:
        st.subheader("First Bottle")
        img1_obj, pred1 = load_and_predict(img1)
        if img1_obj:
            st.image(img1_obj, caption=img1, use_container_width=True)
            display_comparison(img1.lower(), pred1)

    with col2:
        st.subheader("Second Bottle")
        img2_obj, pred2 = load_and_predict(img2)
        if img2_obj:
            st.image(img2_obj, caption=img2, use_container_width=True)
            display_comparison(img2.lower(), pred2)

elif menu == "Evaluation Metrics":
    st.markdown("<h2>Evaluation Metrics</h2>", unsafe_allow_html=True)
    st.write("Binary classification style evaluation of each score using a configurable threshold (>= threshold => Positive).")

    # Threshold selector
    threshold = st.slider("Threshold for Positive Class", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
    # Debug toggle
    debug = st.checkbox("Show debug logs", value=False, help="Print progress details to server console.")

    def log(msg: str):
        if debug:
            print(f"[Eval] {msg}")

    # Precompute predictions once and store in session_state for speed & interactive thresholding
    if 'eval_pred_cache' not in st.session_state:
        progress = st.progress(0, text="Scanning image files...")
        start_time = __import__('time').time()
        file_map = {}
        for idx, (root, _, files) in enumerate(os.walk(VAL_IMAGES_DIR)):
            for f in files:
                fl = f.strip().lower()
                if fl.endswith((".jpg", ".jpeg", ".png")):
                    file_map.setdefault(fl, os.path.join(root, f))
        log(f"Mapped {len(file_map)} image files.")
        progress.progress(10, text="Loading & preprocessing images...")

        y_true_list = []
        image_arrays = []
        missing = 0
        load_errors = 0
        total = len(val_labels.index)
        for i, img_name in enumerate(val_labels.index):
            if total:
                pct = 10 + int( fifty := ( (i+1) / total ) * 60 )  # load phase up to 70%
                progress.progress(min(pct, 70), text=f"Loading images... {i+1}/{total}")
            key = img_name.strip().lower()
            path = file_map.get(key)
            if path is None:
                missing += 1
                continue
            try:
                img = Image.open(path).convert("RGB").resize((224, 224))
            except Exception:
                load_errors += 1
                continue
            arr = np.array(img).astype("float32") / 255.0
            image_arrays.append(arr)
            y_true_list.append(val_labels.loc[img_name, ['durability_score','chemical_safety_score','ergonomics_score']].values.astype(float))

        # Log detailed statistics
        log(f"Total labels: {total}, Successfully loaded: {len(image_arrays)}, Missing files: {missing}, Load errors: {load_errors}")
        if debug and (missing > 0 or load_errors > 0):
            st.info(f"📊 Image loading stats: {len(image_arrays)} loaded, {missing} files not found, {load_errors} load errors out of {total} total labels")

        if not image_arrays:
            st.session_state['eval_pred_cache'] = {
                'y_true': np.empty((0,3)),
                'y_pred': np.empty((0,3)),
                'missing': missing + load_errors,
                'pred_time': 0.0
            }
        else:
            progress.progress(75, text="Running batch predictions...")
            batch_size = 32
            preds = []
            for i in range(0, len(image_arrays), batch_size):
                batch = np.stack(image_arrays[i:i+batch_size], axis=0)
                batch_pred = model.predict(batch, verbose=0)
                preds.append(batch_pred)
                # Update progress within 75-95%
                progress.progress(75 + int(20 * ( (i+batch_size) / len(image_arrays) )), text=f"Predicting... {min(i+batch_size, len(image_arrays))}/{len(image_arrays)}")
            y_pred_full = np.concatenate(preds, axis=0)
            pred_time = __import__('time').time() - start_time
            progress.progress(100, text="Prediction complete")
            log(f"Processed {len(image_arrays)} images; missing {missing}; time {pred_time:.2f}s")
            st.session_state['eval_pred_cache'] = {
                'y_true': np.vstack(y_true_list),
                'y_pred': y_pred_full,
                'missing': missing + load_errors,
                'pred_time': pred_time
            }
        # Remove progress bar after completion
        progress.empty()
    else:
        if debug:
            st.info("Using cached predictions (toggle off debug to hide this).")
        log("Cache hit; skipping recomputation.")
        # Optionally allow manual refresh
        if st.button("Recompute Predictions"):
            del st.session_state['eval_pred_cache']
            st.rerun()

    # Debug information section (separated from metrics)
    if debug:
        st.subheader("🔍 Debug Information")
        
        cache = st.session_state['eval_pred_cache']
        y_true_debug = cache['y_true']
        y_pred_debug = cache['y_pred']
        
        if y_true_debug.shape[0] != y_pred_debug.shape[0]:
            min_samples = min(y_true_debug.shape[0], y_pred_debug.shape[0])
            st.info(f"ℹ️ Array size adjustment: Using {min_samples} samples (y_true: {y_true_debug.shape[0]}, y_pred: {y_pred_debug.shape[0]})")
            y_true_debug = y_true_debug[:min_samples]
            y_pred_debug = y_pred_debug[:min_samples]
        
        st.write("**Data Distribution Analysis:**")
        for i, name in enumerate(['Durability', 'Chemical Safety', 'Ergonomics']):
            true_col = y_true_debug[:, i]
            pred_col = y_pred_debug[:, i]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{name} - True Scores:**")
                st.write(f"  - Min: {true_col.min():.2f}")
                st.write(f"  - Max: {true_col.max():.2f}")
                st.write(f"  - Mean: {true_col.mean():.2f}")
                st.write(f"  - Std: {true_col.std():.2f}")
                st.write(f"  - ≥ {threshold}: {(true_col >= threshold).sum()}/{len(true_col)} ({100*(true_col >= threshold).mean():.1f}%)")
            
            with col2:
                st.write(f"**{name} - Predicted Scores:**")
                st.write(f"  - Min: {pred_col.min():.2f}")
                st.write(f"  - Max: {pred_col.max():.2f}")
                st.write(f"  - Mean: {pred_col.mean():.2f}")
                st.write(f"  - Std: {pred_col.std():.2f}")
                st.write(f"  - ≥ {threshold}: {(pred_col >= threshold).sum()}/{len(pred_col)} ({100*(pred_col >= threshold).mean():.1f}%)")
        
        st.markdown("---")

    def compute_metrics_from_arrays(y_true, y_pred, threshold: float):
        if y_true.size == 0:
            return pd.DataFrame(), {}
        
        # Ensure arrays have the same number of samples
        min_samples = min(y_true.shape[0], y_pred.shape[0])
        if y_true.shape[0] != y_pred.shape[0]:
            y_true = y_true[:min_samples]
            y_pred = y_pred[:min_samples]
        
        feature_names = ['Durability', 'Chemical Safety', 'Ergonomics']
        rows = []
        macro_acc = []
        macro_prec = []
        macro_rec = []
        for i, name in enumerate(feature_names):
            true_binary = (y_true[:, i] >= threshold).astype(int)
            pred_binary = (y_pred[:, i] >= threshold).astype(int)
            
            # Calculate confusion matrix elements
            tp = int(((true_binary == 1) & (pred_binary == 1)).sum())
            tn = int(((true_binary == 0) & (pred_binary == 0)).sum())
            fp = int(((true_binary == 0) & (pred_binary == 1)).sum())
            fn = int(((true_binary == 1) & (pred_binary == 0)).sum())
            
            total = tp + tn + fp + fn
            
            # Validate total matches expected
            if total != len(true_binary):
                st.error(f"Calculation error for {name}: TP+TN+FP+FN={total} != total_samples={len(true_binary)}")
                continue
            
            # Check for degenerate cases
            all_positive_true = (true_binary == 1).all()
            all_negative_true = (true_binary == 0).all()
            all_positive_pred = (pred_binary == 1).all()
            all_negative_pred = (pred_binary == 0).all()
            
            # Calculate metrics with proper handling of edge cases
            accuracy = (tp + tn) / total if total > 0 else 0.0
            
            # Precision: TP / (TP + FP) - undefined if no positive predictions
            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0 if not all_negative_pred else float('nan')  # No positive predictions
            
            # Recall: TP / (TP + FN) - undefined if no positive ground truth
            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0 if not all_negative_true else float('nan')  # No positive ground truth
            
            # F1 score
            if precision + recall > 0 and not (np.isnan(precision) or np.isnan(recall)):
                f1 = (2 * precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            # Handle NaN values for display
            precision_display = "N/A" if np.isnan(precision) else round(precision, 4)
            recall_display = "N/A" if np.isnan(recall) else round(recall, 4)
            f1_display = "N/A" if np.isnan(f1) else round(f1, 4)
            
            macro_acc.append(accuracy)
            # Only include in macro averages if not NaN
            if not np.isnan(precision):
                macro_prec.append(precision)
            if not np.isnan(recall):
                macro_rec.append(recall)
            
            rows.append({
                'Feature': name,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'Accuracy': round(accuracy, 4),
                'Precision': precision_display,
                'Recall': recall_display,
                'F1': f1_display
            })
        df_metrics = pd.DataFrame(rows)
        
        # Calculate macro averages with proper handling of empty lists
        summary_part = {
            'Macro Accuracy': round(float(np.mean(macro_acc)), 4) if macro_acc else 0.0,
            'Macro Precision': round(float(np.mean(macro_prec)), 4) if macro_prec else "N/A",
            'Macro Recall': round(float(np.mean(macro_rec)), 4) if macro_rec else "N/A"
        }
        return df_metrics, summary_part

    if val_labels.empty:
        st.warning("Validation labels are not loaded; cannot compute metrics.")
        st.stop()

    cache = st.session_state['eval_pred_cache']
    y_true_arr = cache['y_true']
    y_pred_arr = cache['y_pred']
    missing_ct = cache['missing']
    pred_time = cache['pred_time']

    metrics_df, summary_core = compute_metrics_from_arrays(y_true_arr, y_pred_arr, threshold)
    summary = {
        **summary_core,
        'Images Evaluated': int(y_true_arr.shape[0]),
        'Images Missing': int(missing_ct),
        'Pred Time (s)': round(float(pred_time), 2)
    }

    if metrics_df.empty:
        st.warning("No validation predictions could be generated (missing images?).")
    else:
        st.subheader("Per-Feature Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Summary (Macro Averages)")
        cols = st.columns(len(summary))
        for (k, v), col in zip(summary.items(), cols):
            col.metric(k, v)

        # Add threshold recommendations
        st.subheader("💡 Threshold Analysis")
        
        # Check for problematic thresholds
        has_perfect_scores = any(row['Accuracy'] == 1.0 for _, row in metrics_df.iterrows())
        has_na_values = any('N/A' in str(row['Precision']) or 'N/A' in str(row['Recall']) for _, row in metrics_df.iterrows())
        
        if has_perfect_scores and not has_na_values:
            st.info("⚠️ **All metrics show perfect scores (1.0)** - This suggests your threshold is too low. "
                   "Try increasing the threshold to get more meaningful classification results.")
            
        elif has_na_values:
            st.info("⚠️ **Some metrics show 'N/A'** - This indicates all samples are classified the same way. "
                   "Try adjusting the threshold to create a more balanced classification.")
        
        if debug:
            cache = st.session_state['eval_pred_cache']
            y_check = cache['y_true'][:min(cache['y_true'].shape[0], cache['y_pred'].shape[0])]
            overall_range = f"{y_check.min():.1f} to {y_check.max():.1f}"
            st.info(f"💡 **Tip**: Score range is approximately {overall_range}. "
                   f"Try setting threshold around the median (~{np.median(y_check):.1f}) for balanced classification.")

        st.caption("TP/TN/FP/FN computed after thresholding continuous scores. Adjust threshold to explore trade-offs.")

elif menu == "ChatGPT Assistant":
    st.markdown("### ChatGPT Assistant")
    st.info("Coming soon: Integrated AI assistant for insights and automated PDF reports.")

elif menu == "Feedback Form":
    st.markdown("<h2>Give Us Feedback</h2>", unsafe_allow_html=True)
    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        comments = st.text_area("Comments or Suggestions")
        submitted = st.form_submit_button("Submit")

        if submitted:
            st.success("Thank you for your feedback!")
