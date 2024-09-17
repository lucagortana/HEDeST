import torch
import os

from tools.basics import set_seed
from tools.analysis import plot_history

class ModelTrainer:
    def __init__(self, model, optimizer, train_loader, val_loader, test_loader, 
                 agg_loss='mean', alpha=0.5, num_epochs=25, out_dir='models', rs=42):
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
            out_dir (str): The output directory to save the trained model. Default is 'models'.
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
        self.agg_loss = agg_loss
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.out_dir = out_dir
        self.rs = rs
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_model_path = os.path.join(self.out_dir, 'best_model.pth')
        self.final_model_path = os.path.join(self.out_dir, 'final_model.pth')
        
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
                spot_outputs = []
                true_proportions = true_proportions.to(self.device)
                
                for images in images_list:
                    images = images.to(self.device)
                    outputs = self.model(images)
                    spot_outputs.append(outputs)
                
                outputs = torch.cat(spot_outputs, dim=0)
                loss = self.model.loss_comb(outputs, true_proportions, agg=self.agg_loss, alpha=self.alpha)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            train_loss = running_loss / len(self.train_loader)
            val_loss = self.evaluate(self.val_loader)
            
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            
            self.history_train.append(train_loss)
            self.history_val.append(val_loss)
            
            # Save best model based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                torch.save(self.best_model_state, self.best_model_path)
                print(f'Validation loss improved. Saving best model at {self.best_model_path} (epoch {epoch + 1}).')

        # If the best model is different from the final model
        if self.best_val_loss != val_loss:
            # Save the final model at the end of training
            torch.save(self.model.state_dict(), self.final_model_path)
            print("Best model and final model are different. Both models have been saved.")
            print(f'Final model saved at {self.final_model_path}.')
        else:
            print("Best model and final model are the same.")

        print('Training complete. Evaluating on test set...')
        self.test()
        
    def evaluate(self, dataloader):
        """ 
        Evaluate the model on the validation or test set.
        """
        
        self.model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        with torch.no_grad():
            for images_list, true_proportions in dataloader:
                spot_outputs = []
                true_proportions = true_proportions.to(self.device)
                
                for images in images_list:
                    images = images.to(self.device)
                    outputs = self.model(images)
                    spot_outputs.append(outputs)
                
                outputs = torch.cat(spot_outputs, dim=0)  # Concatenate spot_outputs
                loss = self.model.loss_comb(outputs, true_proportions, agg=self.agg_loss, alpha=self.alpha)
                running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        return avg_loss

    def test(self):
        """
        Evaluate the final model and the best model on the test set.
        """
        
        # Evaluate the final model
        final_test_loss = self.evaluate(self.test_loader)
        print(f'Test Loss on final model: {final_test_loss:.4f}')

        # Load and evaluate the best model
        print("\nLoading best model for final test evaluation...")
        best_model = type(self.model)(num_classes=self.model.num_classes)  # Instantiate new model with the same architecture
        best_model.load_state_dict(torch.load(self.best_model_path))
        best_model = best_model.to(self.device)

        test_loss_best = self.evaluate(self.test_loader)
        print(f'Test Loss on best model: {test_loss_best:.4f}')
        
    def save_history(self):
        
        history_filedir = self.out_dir + "/history.png"
        plot_history(
            history_train=self.history_train,
            history_val=self.history_val,
            savefig=history_filedir,
        )
        print(f"History saved at {history_filedir}")


# def evaluate(model, dataloader, agg_loss = 'mean', alpha = 0.5, device=torch.device("cpu")):
#     model.eval()  # Set model to evaluation mode
#     running_loss = 0.0
#     with torch.no_grad():
#         for images_list, true_proportions in dataloader: # images -> images_list
#             spot_outputs = [] #created spot_outputs
#             true_proportions = true_proportions.to(device)
            
#             for images in images_list: #created loop over images_list
#                 images = images.to(device)
#                 outputs = model(images)
#                 spot_outputs.append(outputs)
                
#             outputs = torch.cat(spot_outputs, dim=0) #concatenated spot_outputs
            
#             loss = model.loss_comb(outputs, true_proportions, agg=agg_loss, alpha=alpha)
#             running_loss += loss.item()
    
#     avg_loss = running_loss / len(dataloader)
#     return avg_loss

# def train(model, train_loader, val_loader, test_loader, optimizer, agg_loss='mean', alpha=0.5, num_epochs=25, out_dir='models'):
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Training on {device}.")
    
#     best_val_loss = float('inf')
#     best_model_state = None
    
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
        
#         for images_list, true_proportions in train_loader: # images -> images_list
#             optimizer.zero_grad()
#             spot_outputs = [] #created spot_outputs
#             true_proportions = true_proportions.to(device)
            
#             for images in images_list: #created loop over images_list
#                 images = images.to(device)
#                 outputs = model(images)
#                 spot_outputs.append(outputs)
                
#             outputs = torch.cat(spot_outputs, dim=0) #concatenated spot_outputs
            
#             loss = model.loss_comb(outputs, true_proportions, agg=agg_loss, alpha=alpha)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
        
#         train_loss = running_loss / len(train_loader)
#         val_loss = evaluate(model, val_loader, agg_loss=agg_loss, alpha=alpha, device=device)
        
#         print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_state = model.state_dict()
#             best_model_path = os.path.join(out_dir, 'best_model.pth')
#             torch.save(best_model_state, best_model_path)
#             print(f'Validation loss improved. Saving best model at {best_model_path} (epoch {epoch + 1}).')
    
#     if val_loss != best_val_loss:
#         final_model_state = model.state_dict()
#         final_model_path = os.path.join(out_dir, 'final_model.pth')
#         torch.save(final_model_state, final_model_path)
#         print("Best model and final model are different. Both models have been saved.")
#     else:
#         print("Best model and final model are the same. Only the final model has been saved.")
    
#     print('Training complete. Evaluating on test set...')
    
#     test_loss = evaluate(model, test_loader, agg_loss=agg_loss, alpha=alpha, device=device)
#     print(f'Test Loss on final model: {test_loss:.4f}')

#     best_model = type(model)(num_classes = outputs.shape[1])  # Create a new model instance with the same architecture
#     best_model.load_state_dict(torch.load(best_model_path))
#     best_model = best_model.to(device)

#     test_loss_best = evaluate(best_model, test_loader, agg_loss=agg_loss, alpha=alpha, device=device)
#     print(f'Test Loss on best model: {test_loss_best:.4f}')