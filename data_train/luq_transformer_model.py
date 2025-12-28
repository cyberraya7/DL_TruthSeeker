import torch
import numpy as np
from luq_preprocess import preprocess
import torch.optim as optim
from luq_transformer import LuqTransformerModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchsummary import summary
import wandb
import yaml


def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs):
    """
    Trains the model for a given number of epochs.

    Args:
        model: The model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        device: The device to train on.
        criterion: The loss function.
        optimizer: The optimizer.
        scheduler: The scheduler.
        num_epochs: The number of epochs to train for.

    Returns:
        The trained model.
    """
    best_loss = float('inf')
    patience = 5
    no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            data, target = data.to(device), target.to(device).long()
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, _, _ = evaluate_model(model, val_loader, device, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'luq_transformer.pth')
            print(f"Model saved to luq_transformer.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Epoch {epoch+1}/{num_epochs}, best_val={best_loss:.6f}")
                break

        scheduler.step(val_loss)
        wandb.log({
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Learning Rate": optimizer.param_groups[0]['lr']
        })
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    model.load_state_dict(torch.load('luq_transformer.pth', map_location=device))
    return model


def evaluate_model(model, dataloader, device, criterion=None):
    """
    Evaluates the model on a given dataloader.
    Can compute loss and/or return predictions and labels.

    Args:
        model: The model to evaluate.
        dataloader: The data loader to evaluate on.
        device: The device to evaluate on.
        criterion: The loss function.

    Returns:
        The average loss, all predictions, and all labels.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for samples, labels in dataloader:
            samples = samples.to(device).float()
            labels = labels.to(device)

            predictions = model(samples)

            if criterion:
                loss = criterion(predictions, labels.long())
                total_loss += loss.item()

            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if criterion and len(dataloader) > 0 else 0
    return avg_loss, all_preds, all_labels


def test_and_report(model, test_loader, device, class_names):
    """
    Evaluates a model on the test set and prints a classification report
    and confusion matrix.

    Args:
        model: The model to evaluate.
        test_loader: The test data loader.
        device: The device to evaluate on.
        class_names: The class names.

    Returns:
        The accuracy of the model on the test set.
    """
    print("\n--- Starting Final Test (Transformer Model) ---")
    model.eval()

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for samples, labels in test_loader:
            samples = samples.to(device)
            labels = labels.to(device).long()

            predictions = model(samples)

            preds = torch.argmax(predictions, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Final Test Accuracy: {acc*100:.2f}%")

    wandb.log({"Accuracy": acc*100})

    print('--- Classification Report ---')

    # Determine which class indices actually appear in the test set
    unique_labels = sorted(set(all_labels))

    # Align names with labels that are actually present to avoid size mismatch errors.
    if len(unique_labels) == len(class_names):
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    else:
        # Assume class indices correspond to positions in class_names.
        used_class_names = [class_names[i] for i in unique_labels]
        report = classification_report(
            all_labels,
            all_preds,
            labels=unique_labels,
            target_names=used_class_names,
            digits=4,
            zero_division=0,
        )

    print(report)

    print('--- Confusion Matrix ---')
    print(confusion_matrix(all_labels, all_preds))
    return acc


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Configs
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train = np.load('../data_process/train.npy')
    test = np.load('../data_process/test.npy')
    val = np.load('../data_process/val.npy')
    print("Finished loading data")  

    # Preprocess data
    train_loader, test_loader, val_loader = preprocess(
        train, test, val, config['batch_size'], scaler_save_path='scaler.pkl'
    )

    # Create Transformer-based model
    model = LuqTransformerModel(
        input_features=train.shape[1] - 1,
        num_classes=len(config['class_names'])
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    num_epochs = config['num_epochs']
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )
    # `torchsummary.summary` can conflict with Transformer-based models because some
    # internal modules return tuples / None, which breaks its hooks.
    # It is safe to skip the summary and train directly.
    # print(summary(model, input_size=(train.shape[1] - 1,), device=device))

    wandb.init(
        project="Transformer",
        config={
            "architecture": "LuqTransformer",
            "batch_size": config['batch_size'],
            "num_epochs": config['num_epochs'],
            "learning_rate": config['learning_rate'],
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
        },
    )

    # Train model
    wandb.watch(model, log="all")
    model = train_model(
        model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs
    )

    # Test model
    test_and_report(model, test_loader, device, config['class_names'])


