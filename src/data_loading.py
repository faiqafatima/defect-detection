import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

project_root_folder = Path(__file__).resolve().parent.parent
split_file_path = project_root_folder / "outputs" / "split_summary.csv"

image_size = 224  # ResNet18 was pretrained on 224x224 images

# Class name -> number (the model works with numbers, not words)
class_name_to_label = {"normal": 0, "defective": 1}

# Random changes, used ONLY on training images, so the model sees more variety.
# A bottle seen from above is round, so flips and rotations are still a valid bottle.
train_augmentation_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # small lighting changes
])

# Image -> tensor, then normalize with ImageNet mean/std (what ResNet18 expects)
tensor_and_normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transform = transforms.Compose([train_augmentation_transform, tensor_and_normalize_transform])

# Validation and test: NO randomness, only resize + normalize (so results are fair and repeatable)
eval_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    tensor_and_normalize_transform,
])


class DefectDataset(Dataset):
    """Loads images of one split (train, val or test) from split_summary.csv."""

    def __init__(self, split_name, transform):
        self.transform = transform
        self.image_paths = []
        self.class_names = []
        with open(split_file_path, newline="") as split_file:
            for csv_row in csv.DictReader(split_file):
                if csv_row["split_name"] == split_name:
                    self.image_paths.append(project_root_folder / csv_row["image_path"])
                    self.class_names.append(csv_row["class_name"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, image_index):
        image_path = self.image_paths[image_index]
        opened_image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(opened_image)
        label = class_name_to_label[self.class_names[image_index]]
        return image_tensor, label


# Quick check: run this file directly to test it
if __name__ == "__main__":
    train_dataset = DefectDataset("train", train_transform)
    val_dataset = DefectDataset("val", eval_transform)
    test_dataset = DefectDataset("test", eval_transform)
    print(f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    first_image_tensor, first_label = train_dataset[0]
    print(f"One image tensor shape: {tuple(first_image_tensor.shape)}, label: {first_label}")

    # Save the same defective image 4 times with random augmentation, to SEE what augmentation does
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    defective_image_path = train_dataset.image_paths[train_dataset.class_names.index("defective")]
    original_image = Image.open(defective_image_path).convert("RGB")
    figure, axes = plt.subplots(1, 4, figsize=(12, 3))
    for axis in axes:
        axis.imshow(train_augmentation_transform(original_image))
        axis.axis("off")
    preview_path = project_root_folder / "outputs" / "augmentation_preview.png"
    plt.savefig(preview_path, bbox_inches="tight")
    print(f"Augmentation preview saved to {preview_path}")
