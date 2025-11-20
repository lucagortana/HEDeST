from __future__ import annotations

import os
from typing import List
from typing import Tuple

import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hedest.analysis.plots import plot_history
from hedest.model.cell_classifier import CellClassifier
from hedest.utils import set_seed


class ModelTrainer:
    """
    A class for training and evaluating a deep learning model for cell classification tasks.
    """

    def __init__(
        self,
        model: CellClassifier,
        ct_list: List[str],
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        divergence: str = "l2",
        alpha: float = 0.0,
        num_epochs: int = 60,
        out_dir: str = "results",
        tb_dir: str = "runs",
        rs: int = 42,
    ) -> None:
        """
        Initializes the ModelTrainer with the given parameters.

        Args:
            model (CellClassifier): The cell classifier to train and evaluate.
            ct_list (List[str]): List of cell type names.
            optimizer (optim.Optimizer): Optimizer for model training.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            test_loader (DataLoader): DataLoader for the test dataset.
            divergence (str): Type of divergence used in loss calculation.
            alpha (float): Weight parameter for loss components.
            num_epochs (int): Number of training epochs.
            out_dir (str): Directory to save model checkpoints and results.
            tb_dir (str): Directory for TensorBoard logs.
            rs (int): Random seed for reproducibility.
        """

        self.model = model
        self.ct_list = ct_list
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.p_c = self.train_loader.dataset.spot_prop_df.mean().values
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.divergence = divergence
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.out_dir = out_dir
        self.tb_dir = tb_dir
        self.rs = rs

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.best_model_path = os.path.join(self.out_dir, "best_model.pth")

    def init_model(self) -> None:
        """
        Initializes training history attributes as empty lists.
        """

        self.history_train = []
        self.history_val = []

    def _setup_device(self) -> None:
        """
        Sets up the computation device (CPU or GPU) for training.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device used: {self.device}")

    def prepare_training(self) -> None:
        """
        Prepares the model for training by setting the random seed,
        configuring the computation device, and initializing the history.
        """

        set_seed(self.rs)
        self._setup_device()
        self.p_c = torch.tensor(self.p_c, dtype=torch.float32, device=self.device)
        self.init_model()

    def train(self) -> None:
        """
        Trains the model using the specified parameters and logs the performance.
        """

        # Prepare for training
        self.prepare_training()

        tb_file = os.path.join(self.tb_dir, os.path.basename(self.out_dir))
        writer = SummaryWriter(tb_file)

        # Begin training
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            train_loss_half1 = 0.0
            train_loss_half2 = 0.0

            # Training loop
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                images = batch["images"].to(self.device)
                proportions = batch["proportions"].to(self.device)
                bag_indices = batch["bag_indices"]
                unique_indices = torch.unique(bag_indices, sorted=False).flip(0)
                mapping = {val.item(): idx for idx, val in enumerate(unique_indices)}
                new_bag_indices = torch.tensor([mapping[val.item()] for val in bag_indices]).to(self.device)

                cell_probs = self.model(images)
                loss, loss_half1, loss_half2 = self.model.compute_loss(
                    cell_probs,
                    new_bag_indices,
                    proportions,
                    divergence=self.divergence,
                    alpha=self.alpha,
                )
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() / len(self.train_loader)
                train_loss_half1 += loss_half1.item() / len(self.train_loader)
                train_loss_half2 += loss_half2.item() / len(self.train_loader)

                torch.cuda.empty_cache()

            val_loss, val_loss_half1, val_loss_half2 = self.evaluate(self.val_loader)

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

            torch.cuda.empty_cache()

            # TensorBoard logging
            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Val", val_loss, epoch + 1)
            writer.add_scalar("Loss/Train_Half1", train_loss_half1, epoch + 1)
            writer.add_scalar("Loss/Train_Half2", train_loss_half2, epoch + 1)
            writer.add_scalar("Loss/Val_Half1", val_loss_half1, epoch + 1)
            writer.add_scalar("Loss/Val_Half2", val_loss_half2, epoch + 1)

            self.history_train.append(train_loss)
            self.history_val.append(val_loss)

            # Save best model based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                torch.save(self.best_model_state, self.best_model_path)
                logger.info(
                    f"-> Validation loss improved. Saving best model at {self.best_model_path} (epoch {epoch + 1})."
                )

        writer.close()

        logger.info("Training complete. Evaluating on test set...")

        # Evaluate the final and best models on the test set
        self.test()

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        Evaluate the model on the validation or test set.

        Args:
            dataloader (DataLoader): The DataLoader for the dataset to evaluate.

        Returns:
            Tuple[float, float, float]: Total loss, loss component 1, and loss component 2.
        """

        self.model.eval()
        running_loss = 0.0
        running_loss_half1 = 0.0
        running_loss_half2 = 0.0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(self.device)
                proportions = batch["proportions"].to(self.device)
                bag_indices = batch["bag_indices"]
                unique_indices = torch.unique(bag_indices, sorted=False).flip(0)
                mapping = {val.item(): idx for idx, val in enumerate(unique_indices)}
                new_bag_indices = torch.tensor([mapping[val.item()] for val in bag_indices]).to(self.device)

                cell_probs = self.model(images)
                loss, loss_half1, loss_half2 = self.model.compute_loss(
                    cell_probs,
                    new_bag_indices,
                    proportions,
                    divergence=self.divergence,
                    alpha=self.alpha,
                )
                running_loss += loss.item() / len(dataloader)
                running_loss_half1 += loss_half1.item() / len(dataloader)
                running_loss_half2 += loss_half2.item() / len(dataloader)

        return running_loss, running_loss_half1, running_loss_half2

    def test(self) -> None:
        """
        Evaluate the final model and the best model on the test set.
        """

        # Evaluate the final model
        final_test_loss, _, _ = self.evaluate(self.test_loader)
        logger.info(f"Test Loss on final model: {final_test_loss:.4f}")

        # Load and evaluate the best model
        logger.info("Loading best model for final test evaluation...")
        best_model = type(self.model)(
            model_name=self.model.model_name,
            num_classes=self.model.num_classes,
            hidden_dims=self.model.hidden_dims,
            device=self.device,
        )
        best_model.load_state_dict(torch.load(self.best_model_path))
        best_model = best_model.to(self.device)
        self.model = best_model

        test_loss_best, _, _ = self.evaluate(self.test_loader)
        logger.info(f"Test Loss on best model: {test_loss_best:.4f}")

    def save_history(self) -> None:
        """
        Save training and validation history as a plot.

        This method saves a visualization of the training history (losses per epoch)
        as a PNG file in the output directory.
        """

        history_filedir = os.path.join(self.out_dir, "history.png")
        plot_history(
            history_train=self.history_train,
            history_val=self.history_val,
            savefig=history_filedir,
        )
        logger.info(f"History saved at {history_filedir}")
