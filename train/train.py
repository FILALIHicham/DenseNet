import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data.dataloader import get_dataloaders
from models.densenet import DenseNet
import yaml
import argparse
import os
from tqdm import tqdm
import time

def train_epoch(model, loader, criterion, optimizer, device, epoch_pbar):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += outputs.argmax(dim=1).eq(labels).sum().item()
        total += labels.size(0)

        # Update inner progress bar
        epoch_pbar.update(1)
    return running_loss / total, correct / total

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def run_experiment(exp_config, device):
    exp_name = exp_config['name']
    print(f"\n=== Starting experiment: {exp_name} ===")

    start_time = time.time()

    checkpoint_dir = exp_config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(exp_config['logging']['log_dir'], exp_name))

    # Set up data loaders
    dataset_cfg = exp_config['dataset']
    train_loader, test_loader = get_dataloaders(
        dataset_name=dataset_cfg['name'],
        batch_size=dataset_cfg['batch_size'],
        augment=dataset_cfg['augment'],
        data_dir=dataset_cfg['data_dir'],
        num_workers=dataset_cfg['num_workers']
    )

    # Determine number of classes based on dataset
    if dataset_cfg['name'].upper() in ['CIFAR10', 'SVHN']:
        num_classes = 10
    elif dataset_cfg['name'].upper() == 'CIFAR100':
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_cfg['name']}")

    # Set up model
    model_cfg = exp_config['model']
    model = DenseNet(
        growth_rate=model_cfg['growth_rate'],
        block_layers=tuple(model_cfg['block_layers']),
        num_init_features=model_cfg['num_init_features'],
        bottleneck=model_cfg['bottleneck'],
        compression=model_cfg['compression'],
        drop_rate=model_cfg['drop_rate'],
        num_classes=num_classes
    ).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    train_cfg = exp_config['training']
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        momentum=train_cfg['momentum'],
        weight_decay=train_cfg['weight_decay'],
        nesterov=True
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=train_cfg['milestones'],
        gamma=train_cfg['gamma']
    )

    epochs = train_cfg['epochs']
    patience = train_cfg.get('patience', None)
    epochs_no_improve = 0
    best_val_acc = 0.0

    with tqdm(total=epochs, desc="Overall Training", position=0) as outer_pbar:
        for epoch in range(1, epochs + 1):
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", position=1, leave=False) as epoch_pbar:
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch_pbar)

            val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
            scheduler.step()

            print(f"Exp: {exp_name} | Epoch [{epoch}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0  # reset counter
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Exp: {exp_name} | New best model at epoch {epoch} with val acc: {best_val_acc:.4f}")
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                print(f"Exp: {exp_name} | Early stopping at epoch {epoch}. No improvement for {patience} epochs.")
                break

            outer_pbar.update(1)  

    # Training finished
    end_time = time.time()
    total_time_sec = end_time - start_time
    total_time_hr = total_time_sec / 3600.0
    time_summary = f"Total training time: {total_time_hr:.2f} hours"
    print(f"=== Experiment '{exp_name}' completed. Best Val Accuracy: {best_val_acc:.4f} | {time_summary} ===\n")

    writer.add_text('TrainingTime', time_summary, global_step=0)
    writer.close()

def main():
    parser = argparse.ArgumentParser(description="Run multiple DenseNet experiments sequentially.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file with experiments.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in the config file.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for exp_config in experiments:
        run_experiment(exp_config, device)

if __name__ == "__main__":
    main()