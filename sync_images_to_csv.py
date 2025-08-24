# save as: sync_images_to_csv.py  (run:  python sync_images_to_csv.py)

import os, shutil, pandas as pd

# === EDIT THESE ===
SOURCE_DIR = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data/Bottle Images/Bottle Images"


DEST_BASE  = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data"
TRAIN_CSV  = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data/train_labels.csv"
VAL_CSV    = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data/val_labels.csv"

MOVE_FILES = False   # True = move, False = copy (safer)

# ------------------
def norm_cat_for_lookup(s):   # to match source folder names robustly
    return s.strip().lower().replace("_", " ")

def dest_cat_name(s):         # what your training code expects
    return s.strip().lower().replace(" ", "_")

# map "normalized name" -> "actual folder name" by scanning SOURCE_DIR
cat_map = {}
for d in os.listdir(SOURCE_DIR):
    p = os.path.join(SOURCE_DIR, d)
    if os.path.isdir(p):
        cat_map[norm_cat_for_lookup(d)] = d

def find_file_case_insensitive(folder, filename):
    want = filename.lower()
    # exact name first
    p = os.path.join(folder, filename)
    if os.path.exists(p): 
        return p
    # try case-insensitive / ext-variation search
    stem = os.path.splitext(want)[0]
    for f in os.listdir(folder):
        if os.path.splitext(f.lower())[0] == stem:
            return os.path.join(folder, f)
    return None

def sync_one_split(split_name, csv_path):
    df = pd.read_csv(csv_path)
    missing = []

    for _, row in df.iterrows():
        cat_csv = str(row["category"])
        fname   = str(row["image_filename"])
        src_cat = cat_map.get(norm_cat_for_lookup(cat_csv))

        if not src_cat:
            missing.append(f"[no src folder for category] {cat_csv} / {fname}")
            continue

        src_folder = os.path.join(SOURCE_DIR, src_cat)
        src_file   = find_file_case_insensitive(src_folder, fname)

        if not src_file:
            missing.append(f"[file not found] {src_folder} / {fname}")
            continue

        dest_cat  = dest_cat_name(cat_csv)
        dest_dir  = os.path.join(DEST_BASE, split_name, dest_cat)
        os.makedirs(dest_dir, exist_ok=True)
        dest_file = os.path.join(dest_dir, os.path.basename(src_file))

        if MOVE_FILES:
            shutil.move(src_file, dest_file)
        else:
            shutil.copy2(src_file, dest_file)

    print(f"✅ {split_name}: done. Missing: {len(missing)}")
    if missing:
        print("Examples:")
        for m in missing[:10]:
            print(" -", m)

def main():
    os.makedirs(os.path.join(DEST_BASE, "train"), exist_ok=True)
    os.makedirs(os.path.join(DEST_BASE, "val"),   exist_ok=True)
    sync_one_split("train", TRAIN_CSV)
    sync_one_split("val",   VAL_CSV)

if __name__ == "__main__":
    main()
