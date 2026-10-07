import csv
import shutil
from pathlib import Path

project_root_folder = Path(__file__).resolve().parent.parent
predictions_path = project_root_folder / "outputs" / "test_predictions.csv"
error_analysis_folder = project_root_folder / "outputs" / "error_analysis"
error_analysis_folder.mkdir(exist_ok=True)

with open(predictions_path, newline="") as predictions_file:
    prediction_rows = list(csv.DictReader(predictions_file))

# Confidence = probability of the class the model chose
for prediction_row in prediction_rows:
    defective_probability = float(prediction_row["defective_probability"])
    if prediction_row["predicted_class"] == "defective":
        prediction_row["confidence"] = defective_probability
    else:
        prediction_row["confidence"] = 1 - defective_probability

# 1) Wrong predictions: false positives and false negatives
wrong_rows = [row for row in prediction_rows if row["true_class"] != row["predicted_class"]]
print(f"Wrong predictions on test set: {len(wrong_rows)}")
for wrong_row in wrong_rows:
    if wrong_row["true_class"] == "normal":
        error_type = "FP"   # normal called defective (false alarm)
    else:
        error_type = "FN"   # defective called normal (missed defect)
    print(f"  {error_type}: {wrong_row['image_path']} (confidence {wrong_row['confidence']:.2f})")
    shutil.copy(wrong_row["image_path"], error_analysis_folder / f"{error_type}_{Path(wrong_row['image_path']).parent.name}_{Path(wrong_row['image_path']).name}")

# 2) Near-misses: correct but least confident (these would fail first on new data)
correct_rows = [row for row in prediction_rows if row["true_class"] == row["predicted_class"]]
least_confident_rows = sorted(correct_rows, key=lambda row: row["confidence"])[:5]
print("5 least confident correct predictions:")
for least_confident_row in least_confident_rows:
    image_path = Path(least_confident_row["image_path"])
    print(f"  {least_confident_row['true_class']} | {image_path.parent.name}/{image_path.name} | confidence {least_confident_row['confidence']:.2f}")
    shutil.copy(image_path, error_analysis_folder / f"nearmiss_{image_path.parent.name}_{image_path.name}")

# 3) Which defect type is hardest? (average confidence per defect folder)
print("Average confidence per defect type (test set):")
confidence_values_per_defect_type = {}
for prediction_row in prediction_rows:
    if prediction_row["true_class"] == "defective":
        defect_type_name = Path(prediction_row["image_path"]).parent.name
        confidence_values_per_defect_type.setdefault(defect_type_name, []).append(prediction_row["confidence"])
for defect_type_name, confidence_values in confidence_values_per_defect_type.items():
    print(f"  {defect_type_name}: {sum(confidence_values) / len(confidence_values):.2f} ({len(confidence_values)} images)")
print(f"Images saved in {error_analysis_folder}")
