from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import f1_score, recall_score
from data_loading import DefectDataset, train_transform, eval_transform, class_name_to_label

torch.manual_seed(42)  # same random start every run, so results can be reproduced

project_root_folder = Path(__file__).resolve().parent.parent
model_save_path = project_root_folder / "models" / "resnet18_defect.pt"
model_save_path.parent.mkdir(exist_ok=True)

batch_size = 16
epochs_last_layer_only = 5   # phase 1: only the new last layer learns
epochs_fine_tuning = 5       # phase 2: last block (layer4) also learns, slowly

# Load data (num_workers=0 is the safest setting on Windows)
train_dataset = DefectDataset("train", train_transform)
val_dataset = DefectDataset("val", eval_transform)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Class weights for imbalance: the rarer class gets a bigger weight.
# weight = total_images / (number_of_classes * images_in_that_class)
train_labels = [class_name_to_label[class_name] for class_name in train_dataset.class_names]
normal_train_count = train_labels.count(0)
defective_train_count = train_labels.count(1)
total_train_count = len(train_labels)
class_weights = torch.tensor([
    total_train_count / (2 * normal_train_count),
    total_train_count / (2 * defective_train_count),
], dtype=torch.float32)
print(f"Class weights -> normal: {class_weights[0]:.2f}, defective: {class_weights[1]:.2f}")
loss_function = nn.CrossEntropyLoss(weight=class_weights)

# Model: ResNet18 pretrained on ImageNet, everything frozen at first
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for parameter in model.parameters():
    parameter.requires_grad = False
# New last layer: 512 features -> 2 outputs (normal, defective). New layers are trainable by default.
model.fc = nn.Linear(model.fc.in_features, 2)


def train_one_epoch(optimizer):
    model.train()
    total_loss = 0.0
    for image_batch, label_batch in train_data_loader:
        optimizer.zero_grad()                       # clear old gradients
        model_outputs = model(image_batch)          # forward pass
        batch_loss = loss_function(model_outputs, label_batch)
        batch_loss.backward()                       # compute gradients
        optimizer.step()                            # update weights
        total_loss += batch_loss.item() * len(label_batch)
    return total_loss / len(train_dataset)


def evaluate_on_validation():
    model.eval()
    total_loss = 0.0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():                           # no learning here, just checking
        for image_batch, label_batch in val_data_loader:
            model_outputs = model(image_batch)
            total_loss += loss_function(model_outputs, label_batch).item() * len(label_batch)
            true_labels += label_batch.tolist()
            predicted_labels += model_outputs.argmax(dim=1).tolist()
    return total_loss / len(val_dataset), true_labels, predicted_labels


best_val_f1 = -1.0
best_val_loss = float("inf")
total_epochs = epochs_last_layer_only + epochs_fine_tuning
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

for epoch_number in range(1, total_epochs + 1):
    # Switch to phase 2: unfreeze layer4 and use a smaller learning rate
    if epoch_number == epochs_last_layer_only + 1:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_parameters, lr=1e-4)
        print("--- Phase 2: fine-tuning layer4 ---")

    train_loss = train_one_epoch(optimizer)
    val_loss, val_true_labels, val_predicted_labels = evaluate_on_validation()
    # pos_label=1 means we measure the DEFECTIVE class (the one we care about)
    val_defective_recall = recall_score(val_true_labels, val_predicted_labels, pos_label=1, zero_division=0)
    val_defective_f1 = f1_score(val_true_labels, val_predicted_labels, pos_label=1, zero_division=0)
    print(f"Epoch {epoch_number}/{total_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
          f"| val_defective_recall={val_defective_recall:.2f} | val_defective_f1={val_defective_f1:.2f}")

    # Keep only the best model according to validation F1 (never the test set)
    # same F1? then the model with the lower validation loss is better (more stable)
    if val_defective_f1 > best_val_f1 or (val_defective_f1 == best_val_f1 and val_loss < best_val_loss):
        best_val_f1 = val_defective_f1
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"   -> new best model saved (val F1 = {best_val_f1:.2f})")

print(f"Training finished. Best validation F1 (defective): {best_val_f1:.2f}")
print(f"Model saved at {model_save_path}")

