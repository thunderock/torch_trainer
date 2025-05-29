import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Optional, Dict, Any

# # Set environment variables for distributed training
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"


class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int, input_dim: int, output_dim: int = 1):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Generate synthetic data
        self.X = torch.randn(num_samples, input_dim)
        # Create a simple linear relationship with some noise
        weights = torch.randn(input_dim, output_dim)
        self.y = torch.matmul(self.X, weights) + 0.1 * torch.randn(
            num_samples, output_dim
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ExampleModel(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def train_model(
    model: pl.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 100,
    num_nodes: int = 1,
    gpus_per_node: int = 8,
    strategy: str = "ddp",
    precision: str = "16-mixed",
    accelerator: str = "gpu",
    logger: Optional[pl.loggers.Logger] = None,
) -> pl.Trainer:
    """
    Train a PyTorch Lightning model across multiple nodes with multiple GPUs.
    """
    # # Configure distributed training
    # if strategy == "ddp":
    #     if "RANK" in os.environ:
    #         rank = int(os.environ["RANK"])
    #         world_size = int(os.environ["WORLD_SIZE"])
    #         local_rank = int(os.environ.get("LOCAL_RANK", 0))

    #         # Set device for this process
    #         torch.cuda.set_device(local_rank)

    #         # Initialize process group with timeout
    #         try:
    #             dist.init_process_group(
    #                 backend="nccl",
    #                 init_method="env://",
    #                 world_size=world_size,
    #                 rank=rank,
    #                 timeout=timedelta(seconds=300),  # 5 minute timeout
    #             )
    #             print(
    #                 f"Initialized process group: rank={rank}, world_size={world_size}, local_rank={local_rank}"
    #             )
    #         except Exception as e:
    #             print(f"Failed to initialize process group: {e}")
    #             raise

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_nodes=num_nodes,
        devices=gpus_per_node,
        strategy=strategy,
        precision=precision,
        accelerator=accelerator,
        logger=logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                filename="{epoch}-{val_loss:.2f}",
            ),
        ],
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return trainer


def create_data_loaders(
    batch_size: int,
    input_dim: int,
    output_dim: int = 1,
    train_size: int = 10000,
    val_size: int = 2000,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders with synthetic data.

    Args:
        batch_size: Batch size for training
        input_dim: Input dimension
        output_dim: Output dimension
        train_size: Number of training samples
        val_size: Number of validation samples
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create synthetic datasets
    train_dataset = SyntheticDataset(train_size, input_dim, output_dim)
    val_dataset = SyntheticDataset(val_size, input_dim, output_dim)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--input_dim", type=int, default=10)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    args = parser.parse_args()

    # # Set up distributed environment
    # if "RANK" in os.environ:
    #     torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Create model
    model = ExampleModel(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        learning_rate=args.learning_rate,
    )

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
    )

    logger = pl.loggers.CSVLogger(
        save_dir="/sensei-fs/users/astiwari/logs",
        name="torch_trainer",
    )

    # Train the model
    trainer = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        max_epochs=args.max_epochs,
        logger=logger,
    )
