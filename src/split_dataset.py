import csv
import hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split

random_seed = 42  # fixed seed so the split is the same every time we run

project_root_folder = Path(__file__).resolve().parent.parent
bottle_folder = project_root_folder / "data" / "raw" / "bottle"
outputs_folder = project_root_folder / "outputs"
outputs_folder.mkdir(exist_ok=True)

# Collect all images with their class name (same pooling as analyze_dataset.py)
all_image_paths = []
all_class_names = []

normal_image_paths = sorted((bottle_folder / "train" / "good").glob("*.png"))
normal_image_paths += sorted((bottle_folder / "test" / "good").glob("*.png"))
for normal_image_path in normal_image_paths:
    all_image_paths.append(normal_image_path)
    all_class_names.append("normal")

for defect_type_folder in sorted((bottle_folder / "test").iterdir()):
    if defect_type_folder.name == "good":
        continue
    for defective_image_path in sorted(defect_type_folder.glob("*.png")):
        all_image_paths.append(defective_image_path)
        all_class_names.append("defective")

# Duplicate check: two files with the same content have the same hash.
# If a duplicate went to train AND test, test score would be fake (leakage).
seen_file_hashes = set()
unique_image_paths = []
unique_class_names = []
duplicate_count = 0
for image_path, class_name in zip(all_image_paths, all_class_names):
    file_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
    if file_hash in seen_file_hashes:
        duplicate_count += 1
        print(f"Duplicate skipped: {image_path}")
        continue
    seen_file_hashes.add(file_hash)
    unique_image_paths.append(image_path)
    unique_class_names.append(class_name)
print(f"Duplicates found: {duplicate_count}")
print(f"Unique images: {len(unique_image_paths)}")

# Split 1: 70% train, 30% temporary (stratify keeps the normal:defective ratio equal)
train_image_paths, temp_image_paths, train_class_names, temp_class_names = train_test_split(
    unique_image_paths, unique_class_names,
    test_size=0.30, stratify=unique_class_names, random_state=random_seed,
)

# Split 2: temporary 30% -> half validation (15%), half test (15%)
val_image_paths, test_image_paths, val_class_names, test_class_names = train_test_split(
    temp_image_paths, temp_class_names,
    test_size=0.50, stratify=temp_class_names, random_state=random_seed,
)

# Save the split to a CSV so training/evaluation use exactly the same files
split_file_path = outputs_folder / "split_summary.csv"
with open(split_file_path, "w", newline="") as split_file:
    csv_writer = csv.writer(split_file)
    csv_writer.writerow(["image_path", "class_name", "split_name"])
    for split_name, split_paths, split_classes in [
        ("train", train_image_paths, train_class_names),
        ("val", val_image_paths, val_class_names),
        ("test", test_image_paths, test_class_names),
    ]:
        for image_path, class_name in zip(split_paths, split_classes):
            relative_image_path = image_path.relative_to(project_root_folder).as_posix()
            csv_writer.writerow([relative_image_path, class_name, split_name])
        print(f"{split_name}: {len(split_paths)} images, "
              f"normal={split_classes.count('normal')}, defective={split_classes.count('defective')}")
print(f"Split saved to {split_file_path}")
