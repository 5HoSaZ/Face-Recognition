import torch


def train_model(epoch, model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0
    train_accuracy = 0
    batch = 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        # Get predition
        outputs = model(images)
        probs = torch.exp(outputs)
        preds = probs.max(dim=1).indices
        # Calculate loss
        loss = loss_fn(outputs, targets)
        # Update parameters
        loss.backward()
        optimizer.step()
        # Update metrics
        train_loss += loss.item()
        train_accuracy += torch.sum(targets.data == preds).item()
        # Logging
        batch += 1
        print(f"Epoch {epoch} --- Training: Batch {batch}/{len(dataloader)}", end="\r")
    # Epoch results
    train_loss /= len(dataloader.dataset)
    train_accuracy /= len(dataloader.dataset)
    return train_loss, train_accuracy


def test_model(epoch, model, dataloader, loss_fn, device):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    batch = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            # Get predition
            outputs = model(images)
            probs = torch.exp(outputs)
            preds = probs.max(dim=1).indices
            # Calculate loss
            loss = loss_fn(outputs, targets)
            # Update metrics
            test_loss += loss.item()
            test_accuracy += torch.sum(targets.data == preds).item()
            batch += 1
            print(
                f"Epoch {epoch} --- Test: Batch {batch}/{len(dataloader)}         ",
                end="\r",
            )
    test_loss /= len(dataloader.dataset)
    test_accuracy /= len(dataloader.dataset)
    return test_loss, test_accuracy
