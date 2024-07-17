import torch

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str):

  # Put the model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss = 0

  # Loop through data loader and data batches
  for X, y in dataloader:
    X, y = X.to(device), y.unsqueeze(1).to(device)

    y_logits = model(X)

    loss = loss_fn(y_logits, y)
    train_loss += loss

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  # Adjust metrics to get average loss and average accuracy per batch
  total_loss = train_loss / len(dataloader)

  return total_loss