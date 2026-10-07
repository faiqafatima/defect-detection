from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # no popup window, just save the image file
import matplotlib.pyplot as plt

# Folder paths (built from this file's location so it works from anywhere)
project_root_folder = Path(__file__).resolve().parent.parent
bottle_folder = project_root_folder / "data" / "raw" / "bottle"
outputs_folder = project_root_folder / "outputs"
outputs_folder.mkdir(exist_ok=True)

# Normal = train/good + test/good (we pool them, then make our own split later)
normal_image_paths = sorted((bottle_folder / "train" / "good").glob("*.png"))
normal_image_paths += sorted((bottle_folder / "test" / "good").glob("*.png"))

# Defective = every test folder except "good"
defective_image_paths = []
for defect_type_folder in sorted((bottle_folder / "test").iterdir()):
    if defect_type_folder.name == "good":
        continue
    defect_type_image_paths = sorted(defect_type_folder.glob("*.png"))
    print(f"Defect type {defect_type_folder.name}: {len(defect_type_image_paths)} images")
    defective_image_paths += defect_type_image_paths

# Class counts and imbalance ratio
normal_image_count = len(normal_image_paths)
defective_image_count = len(defective_image_paths)
print(f"Normal images: {normal_image_count}")
print(f"Defective images: {defective_image_count}")
print(f"Imbalance ratio (normal : defective) = {normal_image_count / defective_image_count:.2f} : 1")

# Check image sizes (model needs one fixed input size, so we must know them)
all_image_paths = normal_image_paths + defective_image_paths
unique_image_sizes = set()
for image_path in all_image_paths:
    with Image.open(image_path) as opened_image:
        unique_image_sizes.add((opened_image.size, opened_image.mode))
print(f"Unique (width, height) and color mode: {unique_image_sizes}")

# Sample grid: top row = 4 normal, bottom row = 4 defective (spread over defect types)
sample_normal_paths = normal_image_paths[:4]
sample_defective_paths = defective_image_paths[::16][:4]
figure, axes = plt.subplots(2, 4, figsize=(12, 6))
for column_index in range(4):
    axes[0, column_index].imshow(Image.open(sample_normal_paths[column_index]))
    axes[0, column_index].set_title("normal")
    axes[1, column_index].imshow(Image.open(sample_defective_paths[column_index]))
    axes[1, column_index].set_title(sample_defective_paths[column_index].parent.name)
for axis in axes.flatten():
    axis.axis("off")
sample_grid_path = outputs_folder / "sample_grid.png"
plt.savefig(sample_grid_path, bbox_inches="tight")
print(f"Sample grid saved to {sample_grid_path}")
