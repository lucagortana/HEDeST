from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm


def predict_slide(model, image_dict, ct_list, batch_size=32, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("Device used : ", device)

    model.eval()
    model = model.to(device)
    predictions = []

    dataloader = torch.utils.data.DataLoader(list(image_dict.items()), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting on cells", unit="batch", disable=(not verbose)):
            cell_ids, images = batch
            images = images.to(device).float() / 255.0

            outputs = model(images)

            for cell_id, prob_vector in zip(cell_ids, outputs):
                predictions.append(
                    {
                        "cell_id": cell_id,
                        **{ct_list[i]: prob for i, prob in enumerate(prob_vector.cpu().tolist())},
                    }
                )

    predictions_df = pd.DataFrame(predictions)
    predictions_df.set_index("cell_id", inplace=True)

    return predictions_df
