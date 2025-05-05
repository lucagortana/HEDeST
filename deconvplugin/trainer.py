from __future__ import annotations

import os
import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from deconvplugin.analysis.plots import plot_history
from deconvplugin.basics import set_seed
from deconvplugin.model.cell_classifier import CellClassifier

# from deconvplugin.analysis.plots import plot_grid_celltype
# from deconvplugin.predict import predict_slide


class ModelTrainer:
    """
    A class for training and evaluating a deep learning model for cell classification tasks.

    Attributes:
        model (nn.Module): The model to train and evaluate.
        ct_list (List[str]): List of cell type names.
        optimizer (optim.Optimizer): Optimizer for model training.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        weights (torch.Tensor, optional): Class weights for loss calculation, if applicable.
        agg (str): Aggregation method for loss calculation ('proba', 'onehot').
        divergence (str): Type of divergence used in loss calculation ('l1', 'l2', 'kl', 'rot').
        alpha (float): Weight parameter for loss components.
        num_epochs (int): Number of training epochs.
        out_dir (str): Directory to save model checkpoints and results.
        tb_dir (str): Directory for TensorBoard logs.
        rs (int): Random seed for reproducibility.
        device (torch.device): Computation device ('cuda' or 'cpu').
        best_val_loss (float): Best validation loss observed during training.
        best_model_state (Optional[Dict[str, Any]]): State dictionary of the best model.
        best_model_path (str): Path to save the best model.
        final_model_path (str): Path to save the final model.
        history_train (List[float]): List of training losses per epoch.
        history_val (List[float]): List of validation losses per epoch.
    """

    def __init__(
        self,
        model: CellClassifier,
        ct_list: List[str],
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        weights: Optional[torch.Tensor] = None,
        agg: str = "proba",
        divergence: str = "l1",
        alpha: float = 0.5,
        beta: float = 0.0,
        num_epochs: int = 25,
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
            weights (torch.Tensor, optional): Class weights for loss calculation.
            agg (str): Aggregation method for loss calculation.
            divergence (str): Type of divergence used in loss calculation.
            alpha (float): Weight parameter for loss components.
            beta (float): Regularization parameter for Bayesian adjustment.
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
        self.weights = weights
        self.agg = agg
        self.divergence = divergence
        self.reduction = "mean"
        self.alpha = alpha
        self.beta = beta
        self.num_epochs = num_epochs
        self.out_dir = out_dir
        self.tb_dir = tb_dir
        self.rs = rs

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.best_model_path = os.path.join(self.out_dir, "best_model.pth")
        self.final_model_path = os.path.join(self.out_dir, "final_model.pth")

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
        if self.weights is not None:
            self.weights = self.weights.to(self.device)

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
                adjusted_probs = bayesian_adjustment(cell_probs, new_bag_indices, proportions, self.p_c, beta=self.beta)
                loss, loss_half1, loss_half2 = self.model.compute_loss(
                    adjusted_probs,
                    new_bag_indices,
                    proportions,
                    weights=self.weights,
                    agg=self.agg,
                    divergence=self.divergence,
                    reduction=self.reduction,
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

            # if epoch % 10 == 0:

            #     n_img_train = sum(element[0].size(0) for element in self.train_loader.dataset)
            #     n_img_val = sum(element[0].size(0) for element in self.val_loader.dataset)

            #     img_plot_train = self._extract_images_for_tb(self.train_loader, n_img_train)
            #     img_plot_val = self._extract_images_for_tb(self.val_loader, n_img_val)
            #     cell_prob_train = predict_slide(self.model, img_plot_train,
            #                                     self.ct_list, batch_size=256, verbose=False)
            #     cell_prob_val = predict_slide(self.model, img_plot_val, self.ct_list, batch_size=256, verbose=False)

            #     for ct in self.ct_list:
            #         writer.add_figure(
            #             f"Train - {ct}",
            #             plot_grid_celltype(
            #                 cell_prob_train, img_plot_train, ct, n=20, selection="max", show_probs=True, display=False
            #             ),
            #             global_step=epoch + 1,
            #         )
            #         writer.add_figure(
            #             f"Val - {ct}",
            #             plot_grid_celltype(
            #                 cell_prob_val, img_plot_val, ct, n=20, selection="max", show_probs=True, display=False
            #             ),
            #             global_step=epoch + 1,
            #         )

            # Save best model based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                torch.save(self.best_model_state, self.best_model_path)
                logger.info(
                    f"-> Validation loss improved. Saving best model at {self.best_model_path} (epoch {epoch + 1})."
                )

        writer.close()

        # Save best and final models
        if self.best_val_loss != val_loss:
            torch.save(self.model.state_dict(), self.final_model_path)
            logger.info(f"Best model and final model are different. Final model saved at {self.final_model_path}.")
        else:
            logger.info("Best model and final model are the same.")

        logger.info("Training complete. Evaluating on test set...")

        # Evaluate the final model on the test set
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
                adjusted_probs = bayesian_adjustment(cell_probs, new_bag_indices, proportions, self.p_c, beta=self.beta)
                loss, loss_half1, loss_half2 = self.model.compute_loss(
                    adjusted_probs,
                    new_bag_indices,
                    proportions,
                    weights=self.weights,
                    agg=self.agg,
                    divergence=self.divergence,
                    reduction=self.reduction,
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

    def _extract_images_for_tb(self, dataloader: DataLoader, n: int) -> Dict[str, torch.Tensor]:
        """
        Extracts a subset of images from a dataloader for visualization in TensorBoard.

        Args:
            dataloader (DataLoader): The DataLoader to extract images from.
            n (int): The number of images to extract.

        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are indices and values are image tensors.
        """

        global_dict = {}
        for spot in dataloader.dataset:
            for image in spot[0]:
                global_dict[str(len(global_dict))] = image

        if n > len(global_dict):
            raise ValueError(f"Only {len(global_dict)} images are available, but you asked for {n}.")

        random_indices = random.sample(range(len(global_dict)), n)

        image_dict = {}
        for idx in random_indices:
            image_dict[str(len(image_dict))] = (global_dict[str(idx)] * 255).to(torch.uint8)

        return image_dict


def bayesian_adjustment(
    cell_probs: torch.Tensor,  # shape: (N_cells, N_classes)
    bag_indices: torch.Tensor,  # shape: (N_cells,), maps each cell to a spot
    proportions: torch.Tensor,  # shape: (N_spots, N_classes)
    p_c: torch.Tensor,  # shape: (N_classes,), global class proportions
    eps: float = 1e-6,  # avoid division by zero
    beta: float = 0.0,  # regularization term: 0 = full adjustment, 1 = original probabilities
) -> torch.Tensor:
    """
    Bayesian adjustment of cell probabilities with optional recursive regularization.

    Args:
        cell_probs: Predicted cell probabilities.
        bag_indices: Mapping of each cell to its corresponding spot.
        proportions: Spot-level cell type proportions.
        p_c: Global class proportions.
        eps: Small constant to avoid division by zero.
        regularization: If >0, recursively averages adjusted_probs with original cell_probs.

    Returns:
        Adjusted (and optionally regularized) cell probability tensor.
    """
    p_tilde_c = proportions[bag_indices]
    p_c = p_c.clamp(min=eps)

    ratio = p_tilde_c / p_c
    alpha_x = 1.0 / (torch.sum(cell_probs * ratio, dim=1) + eps)
    adjusted_probs = cell_probs * alpha_x.unsqueeze(1) * ratio

    adjusted_probs = (1 - beta) * adjusted_probs + beta * cell_probs

    return adjusted_probs
