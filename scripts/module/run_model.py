from module.load_data import SpotDataset, split_data, collate_fn
from module.cell_classifier import CellClassifier
import torch
from torch import optim
import os

from module.trainer import ModelTrainer
from tools.basics import set_seed

def run_plugin(image_dict,
               spot_dict,
               proportions,
               batch_size=8,
               lr=0.001,
               agg_loss='mean',
               alpha=0.5,
               epochs=25,
               train_size=0.5,
               val_size=0.25,
               out_dir='models',
               rs = 42):
    
    print("\n")
    print("NEW TRAINING")
    print("Parameters :")
    print("------------")
    print(f"Batch size : {batch_size}")
    print(f"Learning rate : {lr}")
    print(f"Aggregation loss : {agg_loss}")
    print(f"Alpha : {alpha}")
    print(f"Number of epochs : {epochs}")
    print(f"Train size : {train_size}")
    print(f"Validation size : {val_size}")
    print(f"Output directory : {out_dir}")
    print(f"Random state : {rs}")
    print("------------")
    print("\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used : ", device, "\n")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    train_spot_dict, train_proportions, val_spot_dict, val_proportions, test_spot_dict, test_proportions = split_data(
        spot_dict, proportions, train_size=train_size, val_size=val_size, rs=rs)
    
    # Create datasets
    set_seed(rs)
    train_dataset = SpotDataset(train_spot_dict, train_proportions, image_dict)
    val_dataset = SpotDataset(val_spot_dict, val_proportions, image_dict)
    test_dataset = SpotDataset(test_spot_dict, test_proportions, image_dict)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) #added collate_fn
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) #added collate_fn
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) #added collate_fn

    model = CellClassifier(num_classes=proportions.shape[1], device=device)
    model = model.to(device)
    
    print(f"{proportions.shape[1]} classes detected !\n")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = ModelTrainer(model, optimizer, train_loader, val_loader, test_loader, 
                           agg_loss=agg_loss, alpha=alpha, num_epochs=epochs, out_dir=out_dir, rs=rs)
    trainer.train()
