import os, shutil, random

source_dir = "/home/hamid/ML/Datasets/cat-vs-dog/PetImages"
base_dir   = "/home/hamid/ML/Datasets/cat-vs-dog/split_data"

classes = ["Cat", "Dog"]

# Create directory structure
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# Prepare split
train_ratio = 0.7
val_ratio   = 0.2
test_ratio  = 0.1

for cls in classes:
    folder = os.path.join(source_dir, cls)
    files = os.listdir(folder)

    random.shuffle(files)

    train_end = int(len(files) * train_ratio)
    val_end   = train_end + int(len(files) * val_ratio)

    train_files = files[:train_end]
    val_files   = files[train_end:val_end]
    test_files  = files[val_end:]

    for f in train_files:
        shutil.copy(os.path.join(folder, f), os.path.join(base_dir, "train", cls, f))

    for f in val_files:
        shutil.copy(os.path.join(folder, f), os.path.join(base_dir, "val", cls, f))

    for f in test_files:
        shutil.copy(os.path.join(folder, f), os.path.join(base_dir, "test", cls, f))

print("Split complete.")