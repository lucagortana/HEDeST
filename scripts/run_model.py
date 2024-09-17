from data.load_data import SpotDataset, split_data, collate_fn
from model.cell_classifier import CellClassifier
import torch
from torch import optim
import os
from tqdm import tqdm

from train.LLP_trainer import train
from utils.hovernet_tools import map_cells_to_spots

def run_plugin(adata,
               adata_name,
               json_path,
               image_dict,
               proportions,
               batch_size=1,
               lr=0.001,
               agg_loss='mean',
               alpha=0.5,
               epochs=25,
               train_size=0.5,
               val_size=0.25,
               out_dir='models'):
    
    print("Parameters :")
    print(f"Dataset : {adata_name}")
    print(f"Batch size : {batch_size}")
    print(f"Learning rate : {lr}")
    print(f"Aggregation loss : {agg_loss}")
    print(f"Alpha : {alpha}")
    print(f"Number of epochs : {epochs}")
    print(f"Train size : {train_size}")
    print(f"Validation size : {val_size}")
    print(f"Output directory : {out_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used : ", device)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    spot_dict = map_cells_to_spots(adata, adata_name, json_path)
    
    train_spot_dict, train_proportions, val_spot_dict, val_proportions, test_spot_dict, test_proportions = split_data(
        spot_dict, proportions, train_size=train_size, val_size=val_size)
    
    # Create datasets
    train_dataset = SpotDataset(train_spot_dict, train_proportions, image_dict)
    val_dataset = SpotDataset(val_spot_dict, val_proportions, image_dict)
    test_dataset = SpotDataset(test_spot_dict, test_proportions, image_dict)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) #added collate_fn
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) #added collate_fn
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) #added collate_fn

    model = CellClassifier(num_classes=proportions.shape[1], device=device)
    model = model.to(device)
    
    print(f"{proportions.shape[1]} classes detected !")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train(model, train_loader, val_loader, test_loader, optimizer, agg_loss=agg_loss, alpha=alpha, num_epochs=epochs, out_dir=out_dir)
    

def predict_slide(model, image_dict, batch_size=32):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used : ", device)
    
    model.eval()
    model = model.to(device)
    predictions = {}

    dataloader = torch.utils.data.DataLoader(list(image_dict.items()), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting on cells", unit="batch"):
            cell_ids, images = batch
            images = images.to(device).float() / 255.0

            outputs = model(images)
            predicted_classes = torch.argmax(outputs, dim=1)

            # Store the predicted class and the full probability vector for each cell
            for cell_id, pred_class, prob_vector in zip(cell_ids, predicted_classes, outputs):
                predictions[cell_id] = {
                    'predicted_class': pred_class.item(),
                    'probabilities': prob_vector.cpu().tolist()  # Convert tensor to a list for easier handling
                }

    return predictions
