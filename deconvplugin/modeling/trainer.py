from __future__ import annotations

import os
import random

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from deconvplugin.basics import set_seed
from deconvplugin.modeling.predict import predict_slide
from deconvplugin.plots import plot_grid_celltype
from deconvplugin.plots import plot_history


class ModelTrainer:
    def __init__(
        self,
        model,
        ct_list,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        weights=None,
        agg="proba",
        divergence="l1",
        reduction="mean",
        alpha=0.5,
        num_epochs=25,
        out_dir="results",
        tb_dir="runs",
        rs=42,
    ):
        """
        ModelTrainer class for training a VAE model.

        Attributes:
            model (CellClassifier): The VAE model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use for training the model.
            train_loader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
            val_loader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
            test_loader (torch.utils.data.DataLoader): The DataLoader for the test dataset.
            agg_loss (str): The type of loss aggregation to use ('mean' or 'sum'). Default is 'mean'.
            alpha (float): The weight parameter for the KL divergence loss. Default is 0.5.
            num_epochs (int): The number of epochs to train the model. Default is 25.
            out_dir (str): The output directory to save the trained model. Default is 'results'.
            rs (int): The random seed value to set for reproducibility. Default is 42.

        Example:
            >>> trainer = ModelTrainer(model, optimizer, train_loader, val_loader, test_loader)
            >>> trainer.train()
        """
        self.model = model
        self.ct_list = ct_list
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.weights = weights
        self.agg = agg
        self.divergence = divergence
        self.reduction = reduction
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.out_dir = out_dir
        self.tb_dir = tb_dir
        self.rs = rs

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.best_model_path = os.path.join(self.out_dir, "best_model.pth")
        self.final_model_path = os.path.join(self.out_dir, "final_model.pth")

    def init_model(self):
        """
        Initializes the history_train and history_val attributes as empty lists.
        """
        self.history_train = []
        self.history_val = []

    def _setup_device(self):
        """
        Sets up the device for training.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device used: {self.device}")

    def prepare_training(self):
        """
        Prepares the model for training by setting the seed, setting up the device, and setting the optimizer.
        """
        set_seed(self.rs)
        self._setup_device()
        self.init_model()

    def train(self):
        """
        Trains the model using the specified optimizer and data loaders.
        """

        self.prepare_training()
        if self.weights is not None:
            self.weights = self.weights.to(self.device)

        tb_file = os.path.join(self.tb_dir, os.path.basename(self.out_dir))
        writer = SummaryWriter(tb_file)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            train_loss_half1 = 0.0
            train_loss_half2 = 0.0

            for images_list, true_proportions in self.train_loader:
                self.optimizer.zero_grad()
                true_proportions = true_proportions.to(self.device)
                images = images_list[0].to(self.device)
                outputs = self.model(images)
                loss, loss_half1, loss_half2 = self.model.loss_comb(
                    outputs,
                    true_proportions[0],
                    weights=self.weights,
                    agg=self.agg,
                    divergence=self.divergence,
                    reduction=self.reduction,
                    alpha=self.alpha,
                )
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_loss_half1 += loss_half1.item()
                train_loss_half2 += loss_half2.item()

                torch.cuda.empty_cache()

            val_loss, val_loss_half1, val_loss_half2 = self.evaluate(self.val_loader)

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

            torch.cuda.empty_cache()

            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Val", val_loss, epoch + 1)
            writer.add_scalar("Loss/Train_Half1", train_loss_half1, epoch + 1)
            writer.add_scalar("Loss/Train_Half2", train_loss_half2, epoch + 1)
            writer.add_scalar("Loss/Val_Half1", val_loss_half1, epoch + 1)
            writer.add_scalar("Loss/Val_Half2", val_loss_half2, epoch + 1)

            self.history_train.append(train_loss)
            self.history_val.append(val_loss)

            if epoch % 10 == 0:

                n_img_train = sum(element[0].size(0) for element in self.train_loader.dataset)
                n_img_val = sum(element[0].size(0) for element in self.val_loader.dataset)

                img_plot_train = self._extract_images_for_tb(self.train_loader, n_img_train)
                img_plot_val = self._extract_images_for_tb(self.val_loader, n_img_val)
                pred_on_train = predict_slide(self.model, img_plot_train, self.ct_list, batch_size=256, verbose=False)
                pred_on_val = predict_slide(self.model, img_plot_val, self.ct_list, batch_size=256, verbose=False)

                for ct in self.ct_list:
                    writer.add_figure(
                        f"Train - {ct}",
                        plot_grid_celltype(
                            pred_on_train, img_plot_train, ct, n=20, selection="max", show_probs=True, display=False
                        ),
                        global_step=epoch + 1,
                    )
                    writer.add_figure(
                        f"Val - {ct}",
                        plot_grid_celltype(
                            pred_on_val, img_plot_val, ct, n=20, selection="max", show_probs=True, display=False
                        ),
                        global_step=epoch + 1,
                    )

            # Save best model based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                torch.save(self.best_model_state, self.best_model_path)
                logger.info(
                    f"-> Validation loss improved. Saving best model at {self.best_model_path} (epoch {epoch + 1})."
                )

        writer.close()

        # If the best model is different from the final model
        if self.best_val_loss != val_loss:
            # Save the final model at the end of training
            torch.save(self.model.state_dict(), self.final_model_path)
            logger.info(f"Best model and final model are different. Final model saved at {self.final_model_path}.")
        else:
            logger.info("Best model and final model are the same.")

        logger.info("Training complete. Evaluating on test set...")
        self.test()

    def evaluate(self, dataloader):
        """
        Evaluate the model on the validation or test set.
        """

        self.model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        running_loss_half1 = 0.0
        running_loss_half2 = 0.0

        with torch.no_grad():
            for images_list, true_proportions in dataloader:
                true_proportions = true_proportions.to(self.device)
                images = images_list[0].to(self.device)
                outputs = self.model(images)
                loss, loss_half1, loss_half2 = self.model.loss_comb(
                    outputs,
                    true_proportions[0],
                    weights=self.weights,
                    agg=self.agg,
                    divergence=self.divergence,
                    reduction=self.reduction,
                    alpha=self.alpha,
                )
                running_loss += loss.item()
                running_loss_half1 += loss_half1.item()
                running_loss_half2 += loss_half2.item()

        return running_loss, running_loss_half1, running_loss_half2

    def test(self):
        """
        Evaluate the final model and the best model on the test set.
        """

        # Evaluate the final model
        final_test_loss, _, _ = self.evaluate(self.test_loader)
        logger.info(f"Test Loss on final model: {final_test_loss:.4f}")

        # Load and evaluate the best model
        logger.info("Loading best model for final test evaluation...")
        best_model = type(self.model)(
            size_edge=self.model.size_edge,
            num_classes=self.model.num_classes,
            mtype=self.model.mtype,
            device=self.device,
        )
        best_model.load_state_dict(torch.load(self.best_model_path))
        best_model = best_model.to(self.device)
        self.model = best_model

        test_loss_best, _, _ = self.evaluate(self.test_loader)
        logger.info(f"Test Loss on best model: {test_loss_best:.4f}")

    def save_history(self):

        history_filedir = os.path.join(self.out_dir, "/history.png")
        plot_history(
            history_train=self.history_train,
            history_val=self.history_val,
            savefig=history_filedir,
        )
        logger.info(f"History saved at {history_filedir}")

    def _extract_images_for_tb(self, dataloader, n):
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
