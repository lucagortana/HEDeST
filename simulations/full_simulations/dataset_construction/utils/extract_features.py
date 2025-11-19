from __future__ import annotations

import argparse

from model_features import extract_embeddings
from model_features import save_embedding_dict


def main(model_name: str, image_dict_path: str, out_dir: str, batch_size: int):

    embeddings_dict = extract_embeddings(model_name=model_name, image_dict_path=image_dict_path, batch_size=batch_size)

    save_embedding_dict(embeddings_dict, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--image_dict_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(
        model_name=args.model_name,
        image_dict_path=args.image_dict_path,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
    )
