import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ─── EDIT THESE 3 PATHS IF NEEDED ──────────────────────────────
DATASET_ROOT = r"C:\Users\User\Downloads\archive\Bottle Images\Bottle Images"
LABELS_CSV   = r"C:\Users\User\OneDrive\Documents\prism\bottle_labels_essential_trio.csv"
OUTPUT_ROOT  = r"C:\Users\User\OneDrive\Documents\prism\data"
# ───────────────────────────────────────────────────────────────

CATEGORIES = [
    "Wine Bottle",
    "Water Bottle",
    "Soda Bottle",
    "Plastic Bottles",
    "Beer Bottles",
]

def ensure_dirs():
    for split in ["train", "val", "test"]:
        for cat in CATEGORIES:
            dst = os.path.join(
                OUTPUT_ROOT, split, cat.replace(" ", "_").lower()
            )
            os.makedirs(dst, exist_ok=True)

def split_and_save_labels():
    df = pd.read_csv(LABELS_CSV)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["category"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["category"], random_state=42
    )

    train_df.to_csv(os.path.join(OUTPUT_ROOT, "train_labels.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_ROOT, "val_labels.csv"),   index=False)
    test_df.to_csv(os.path.join(OUTPUT_ROOT, "test_labels.csv"), index=False)

    print("✅ Label CSVs saved:")
    for name, part in zip(
        ["Train", "Validation", "Test"], [train_df, val_df, test_df]
    ):
        pct = len(part) / len(df) * 100
        print(f"   {name}: {len(part):5d} ({pct:4.1f}%)")

def copy_images(split_csv, split_name):
    split_df = pd.read_csv(split_csv)
    for _, row in split_df.iterrows():
        src = os.path.join(DATASET_ROOT, row["category"], row["image_filename"])
        dst_dir = os.path.join(
            OUTPUT_ROOT, split_name, row["category"].replace(" ", "_").lower()
        )
        dst = os.path.join(dst_dir, row["image_filename"])
        if not os.path.exists(dst):           # skip if already copied
            shutil.copyfile(src, dst)

def main():
    ensure_dirs()
    split_and_save_labels()
    print("⏳ Copying images (this can take several minutes)…")
    copy_images(os.path.join(OUTPUT_ROOT, "train_labels.csv"), "train")
    copy_images(os.path.join(OUTPUT_ROOT, "val_labels.csv"),   "val")
    copy_images(os.path.join(OUTPUT_ROOT, "test_labels.csv"),  "test")
    print("🎉 Done!  Images organised under:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
