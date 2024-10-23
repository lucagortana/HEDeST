from __future__ import annotations

import os

import torch
from tools.analysis import plot_history
from tools.basics import set_seed


class ModelTrainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        weights=None,
        agg_loss="mean",
        alpha=0.5,
        num_epochs=25,
        out_dir="results",
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
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.weights = weights
        self.agg_loss = agg_loss
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.out_dir = out_dir
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
        print("Device used : ", self.device)

    def prepare_training(self):
        """
        Prepares the model for training by setting the seed, setting up the device, and setting the optimizer.
        """
        set_seed(self.rs)
        self._setup_device()
        self.init_model()
        print("Beginning training...")

    def train(self):
        """
        Trains the model using the specified optimizer and data loaders.
        """

        self.prepare_training()

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for images_list, true_proportions in self.train_loader:
                self.optimizer.zero_grad()
                # spot_outputs = []
                true_proportions = true_proportions.to(self.device)
                images = images_list[0].to(self.device)
                outputs = self.model(images)
                # spot_outputs.append(outputs)
                # outputs = torch.cat(spot_outputs, dim=0)
                loss = self.model.loss_comb(
                    outputs, true_proportions[0], weights=self.weights, agg=self.agg_loss, alpha=self.alpha
                )
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(self.train_loader)
            val_loss = self.evaluate(self.val_loader)

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            self.history_train.append(train_loss)
            self.history_val.append(val_loss)

            # Save best model based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                torch.save(self.best_model_state, self.best_model_path)
                print(f"Validation loss improved. Saving best model at {self.best_model_path} (epoch {epoch + 1}).")

        # If the best model is different from the final model
        if self.best_val_loss != val_loss:
            # Save the final model at the end of training
            torch.save(self.model.state_dict(), self.final_model_path)
            print("Best model and final model are different. Both models have been saved.")
            print(f"Final model saved at {self.final_model_path}.")
        else:
            print("Best model and final model are the same.")

        print("Training complete. Evaluating on test set...")
        self.test()

    def evaluate(self, dataloader):
        """
        Evaluate the model on the validation or test set.
        """

        self.model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        with torch.no_grad():
            for images_list, true_proportions in dataloader:
                # spot_outputs = []
                true_proportions = true_proportions.to(self.device)
                images = images_list[0].to(self.device)
                outputs = self.model(images)
                # spot_outputs.append(outputs)
                # outputs = torch.cat(spot_outputs, dim=0)  # Concatenate spot_outputs
                loss = self.model.loss_comb(
                    outputs, true_proportions[0], weights=self.weights, agg=self.agg_loss, alpha=self.alpha
                )
                running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        return avg_loss

    def test(self):
        """
        Evaluate the final model and the best model on the test set.
        """

        # Evaluate the final model
        final_test_loss = self.evaluate(self.test_loader)
        print(f"Test Loss on final model: {final_test_loss:.4f}")

        # Load and evaluate the best model
        print("\nLoading best model for final test evaluation...")
        best_model = type(self.model)(
            size_edge=self.model.size_edge, num_classes=self.model.num_classes
        )  # Instantiate new model with the same architecture
        best_model.load_state_dict(torch.load(self.best_model_path))
        best_model = best_model.to(self.device)
        self.model = best_model

        test_loss_best = self.evaluate(self.test_loader)
        print(f"Test Loss on best model: {test_loss_best:.4f}")

    def save_history(self):

        history_filedir = self.out_dir + "/history.png"
        plot_history(
            history_train=self.history_train,
            history_val=self.history_val,
            savefig=history_filedir,
        )
        print(f"History saved at {history_filedir}")
