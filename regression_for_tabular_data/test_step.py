import torch

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str):

  # Put model in eval mode
  model.eval()

  # Setup the test loss and test accuracy values
  test_loss = 0

  # Turn on inference context manager
  with torch.inference_mode():
    for X, y in dataloader:
      X, y = X.to(device), y.unsqueeze(1).to(device)

      # 1. Forward pass
      y_logits = model(X)

      # 2. Calculuate and accumulate loss
      loss = loss_fn(y_logits, y)
      test_loss += loss

  total_loss = test_loss / len(dataloader)

  return total_loss