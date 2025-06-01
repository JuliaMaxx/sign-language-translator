import os
import shutil
import random
from tqdm import tqdm

# === Configuration ===
SOURCE_DIR = 'data/processed_combine_asl_dataset'  # Your original dataset
DEST_DIR = 'asl_data_split'  # Where split folders will be created
SPLIT_RATIOS = {
    'train': 0.8,
    'val': 0.1,
    'test': 0.1
}
SEED = 42  # For reproducibility

# === Setup ===
random.seed(SEED)

for split in SPLIT_RATIOS:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

# === Process each class folder ===
classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for cls in tqdm(classes, desc="Splitting dataset"):
    cls_dir = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
    random.shuffle(images)

    total = len(images)
    train_end = int(SPLIT_RATIOS['train'] * total)
    val_end = train_end + int(SPLIT_RATIOS['val'] * total)

    split_data = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, split_images in split_data.items():
        split_cls_dir = os.path.join(DEST_DIR, split, cls)
        os.makedirs(split_cls_dir, exist_ok=True)
        for img in split_images:
            src_path = os.path.join(cls_dir, img)
            dst_path = os.path.join(split_cls_dir, img)
            shutil.copy2(src_path, dst_path)

print(f"\n✅ Dataset split complete! Output in: {DEST_DIR}")
