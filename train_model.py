import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── CONFIG ──────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 15

# Paths – update to *full* location of the split folders
DATA_DIR = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data/train"
VAL_DIR = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data/val"


TRAIN_CSV = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data/train_labels.csv"
VAL_CSV   = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data/val_labels.csv"



# ──────────────────────────────────────────────────────────────

# 1 ▸ Load label files
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

# 2 ▸ Custom data generator (reads image, applies aug, returns 3-score target)
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, root_dir, batch_size=BATCH_SIZE,
                 img_size=IMAGE_SIZE, augment=False):
        self.df        = df.reset_index(drop=True)
        self.root_dir  = root_dir
        self.batch     = batch_size
        self.img_size  = img_size
        self.augment   = augment
        self.datagen   = ImageDataGenerator(
            rescale=1/255.,
            horizontal_flip=augment,
            rotation_range=15 if augment else 0,
            brightness_range=[0.8, 1.2] if augment else None,
        )
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):                            # batches/epoch
        return len(self.df) // self.batch

    def __getitem__(self, idx):
        idxs = self.indexes[idx*self.batch:(idx+1)*self.batch]
        batch  = self.df.iloc[idxs]
        imgs   = []
        labels = []
        for _, row in batch.iterrows():
            sub = row["category"].replace(" ", "_").lower()
            path = os.path.join(self.root_dir, sub, row["image_filename"])
            img  = tf.keras.preprocessing.image.load_img(path,
                                                         target_size=self.img_size)
            img  = tf.keras.preprocessing.image.img_to_array(img)
            img  = self.datagen.random_transform(img)
            img  = self.datagen.standardize(img)
            imgs.append(img)
            labels.append([
                row["durability_score"],
                row["chemical_safety_score"],
                row["ergonomics_score"],
            ])
        return np.array(imgs), np.array(labels)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

train_gen = DataGenerator(train_df, DATA_DIR, augment=True)
val_gen   = DataGenerator(val_df,   VAL_DIR,  augment=False)

# 3 ▸ Build the model
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
x = tf.keras.layers.Dense(128, activation="relu")(base.output)
out = tf.keras.layers.Dense(3, activation="linear")(x)     # 3 regression heads
model = tf.keras.Model(base.input, out)

model.compile(optimizer="adam",
              loss="mse",
              metrics=["mae"])

# 4 ▸ Callbacks
cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                     patience=3,
                                     restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("models/best_model.h5",
                                       monitor="val_loss",
                                       save_best_only=True),
]

# 5 ▸ Train
model.fit(train_gen,
          epochs=EPOCHS,
          validation_data=val_gen,
          callbacks=cbs)

print("✅ Training complete! Best weights → models/best_model.h5")
