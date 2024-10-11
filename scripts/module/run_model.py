from __future__ import annotations

import os
import pickle

import pandas as pd
import torch
from module.cell_classifier import CellClassifier
from module.load_data import collate_fn
from module.load_data import split_data
from module.load_data import SpotDataset
from module.trainer import ModelTrainer
from tools.basics import set_seed
from torch import optim
from tqdm import tqdm

# def run_pri_deconv()


def run_sec_deconv(
    image_dict,
    spot_dict,
    proportions,
    batch_size=8,
    lr=0.001,
    agg_loss="mean",
    alpha=0.5,
    epochs=25,
    train_size=0.5,
    val_size=0.25,
    out_dir="models",
    rs=42,
):

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
        spot_dict, proportions, train_size=train_size, val_size=val_size, rs=rs
    )

    # Create datasets
    set_seed(rs)
    train_dataset = SpotDataset(train_spot_dict, train_proportions, image_dict)
    val_dataset = SpotDataset(val_spot_dict, val_proportions, image_dict)
    test_dataset = SpotDataset(test_spot_dict, test_proportions, image_dict)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )  # added collate_fn
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )  # added collate_fn
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )  # added collate_fn

    num_classes = proportions.shape[1]
    ct_list = list(proportions.columns)
    model = CellClassifier(num_classes=num_classes, device=device)
    model = model.to(device)

    print(f"{proportions.shape[1]} classes detected !\n")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = ModelTrainer(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        agg_loss=agg_loss,
        alpha=alpha,
        num_epochs=epochs,
        out_dir=out_dir,
        rs=rs,
    )
    trainer.train()
    trainer.save_history()

    # Predict on the whole slide
    model4pred = CellClassifier(num_classes=num_classes, device=device)
    model4pred.load_state_dict(torch.load(trainer.best_model_path))
    pred = predict_slide(model4pred, image_dict, ct_list)

    # Save model infos
    info_dir = f"{out_dir}/info.pickle"
    print(f"Saving objects to {info_dir}")
    with open(info_dir, "wb") as f:
        pickle.dump({"image_dict": image_dict, "spot_dict": spot_dict, "proportions": proportions, "pred": pred}, f)


def predict_slide(model, image_dict, ct_list, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used : ", device)

    model.eval()
    model = model.to(device)
    predictions = []

    dataloader = torch.utils.data.DataLoader(list(image_dict.items()), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting on cells", unit="batch"):
            cell_ids, images = batch
            images = images.to(device).float() / 255.0

            outputs = model(images)

            # Convertir les résultats en liste de dictionnaires pour chaque cellule
            for cell_id, prob_vector in zip(cell_ids, outputs):
                predictions.append(
                    {
                        "cell_id": cell_id,
                        **{
                            ct_list[i]: prob for i, prob in enumerate(prob_vector.cpu().tolist())
                        },  # Utilisation des noms de classes
                    }
                )

    # Convertir la liste de dictionnaires en DataFrame
    predictions_df = pd.DataFrame(predictions)
    predictions_df.set_index("cell_id", inplace=True)

    return predictions_df
