from tqdm.auto import tqdm
from .train_step import train_step
from .test_step import test_step
import torch
from .early_stopper import EarlyStopper
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          file_name: str,
          lr: float = 0.001,
          epochs: int = 5,
          device: str = 'cuda',
          use_early_stopping: bool = False,
          early_stopping_patience: int = 3,
          early_stopping_min_delta: int = 10):

      if use_early_stopping:
            early_stopper = EarlyStopper(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

      # Create results dictionary
      results = {"train_loss": [],
                  "test_loss": []}
      
      best_mse = np.inf

      optimizer = optim.Adam(model.parameters(), lr)

      for epoch in tqdm(range(epochs)):

            # lr ajustable
            if (epoch % 250 == 0) and (lr > 0.00001):
                  lr /= 5
                  optimizer = optim.Adam(model.parameters(), lr)

            # Train step
            train_loss = train_step(model=model,
                                    dataloader=train_dataloader,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer,
                                    device=device)
            # Test step
            test_loss = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

            # Print out what's happening
            if (epoch % 50 == 0):

                  print(f"Epoch: {epoch+1} | "
                        f"train_loss: {train_loss:.4f} | "
                        f"test_loss: {test_loss:.4f} | "
                  )

            # Update the results dictionary
            results["train_loss"].append(train_loss)
            results["test_loss"].append(test_loss)

            if test_loss < best_mse:
                  best_mse = test_loss
                  torch.save(model.state_dict(), f'{file_name}_best_model.pth')

                  if (epoch % 10 == 0):
                        print(f"New best MSE: {best_mse} | Saving")

            # early stopping
            if use_early_stopping:
                  if early_stopper.early_stop(test_loss):             
                        break

            # writer

            if writer:
            # Add results to SummaryWriter
                  writer.add_scalars(main_tag="Loss", 
                                    tag_scalar_dict={"train_loss": train_loss,
                                                      "test_loss": test_loss},
                                    global_step=epoch)

                  # Close the writer
                  writer.close()
            else:
                  pass

      torch.save(model.state_dict(), f'{file_name}_epoch_{epoch}.pth')

      # Return the results dictionary
      return results