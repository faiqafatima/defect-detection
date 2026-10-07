import csv
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_loading import DefectDataset, eval_transform

project_root_folder = Path(__file__).resolve().parent.parent
model_save_path = project_root_folder / "models" / "resnet18_defect.pt"
outputs_folder = project_root_folder / "outputs"

class_names = ["normal", "defective"]  # index 0 = normal, index 1 = defective

# Rebuild the same model shape, then load the trained weights into it
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Test data: only resize + normalize, no augmentation
test_dataset = DefectDataset("test", eval_transform)
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

true_labels = []
predicted_labels = []
defective_probabilities = []
with torch.no_grad():
    for image_batch, label_batch in test_data_loader:
        model_outputs = model(image_batch)
        # softmax turns the 2 raw scores into probabilities that add up to 1
        probabilities = torch.softmax(model_outputs, dim=1)
        true_labels += label_batch.tolist()
        predicted_labels += model_outputs.argmax(dim=1).tolist()
        defective_probabilities += probabilities[:, 1].tolist()

# Precision, recall, F1 for each class
print(classification_report(true_labels, predicted_labels, target_names=class_names, digits=3))

# Confusion matrix: rows = true class, columns = predicted class
confusion_matrix_values = confusion_matrix(true_labels, predicted_labels)
true_negative_count, false_positive_count, false_negative_count, true_positive_count = confusion_matrix_values.ravel()
print(f"True normal predicted normal (TN): {true_negative_count}")
print(f"Normal wrongly called defective (FP, false alarm): {false_positive_count}")
print(f"Defective wrongly called normal (FN, missed defect): {false_negative_count}")
print(f"Defective correctly found (TP): {true_positive_count}")

confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix_values, display_labels=class_names)
confusion_matrix_display.plot(cmap="Blues")
plt.title("Test set confusion matrix")
confusion_matrix_path = outputs_folder / "confusion_matrix.png"
plt.savefig(confusion_matrix_path, bbox_inches="tight")
print(f"Confusion matrix saved to {confusion_matrix_path}")

# Save every test prediction (used next step for error analysis)
predictions_path = outputs_folder / "test_predictions.csv"
with open(predictions_path, "w", newline="") as predictions_file:
    csv_writer = csv.writer(predictions_file)
    csv_writer.writerow(["image_path", "true_class", "predicted_class", "defective_probability"])
    for image_path, true_label, predicted_label, defective_probability in zip(
        test_dataset.image_paths, true_labels, predicted_labels, defective_probabilities
    ):
        csv_writer.writerow([image_path, class_names[true_label], class_names[predicted_label], f"{defective_probability:.4f}"])
print(f"Predictions saved to {predictions_path}")
