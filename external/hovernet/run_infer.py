"""run_infer.py

Usage:
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)

Options:
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name,
                              and expected overlaid color. [default: '']
  --adata_path=<path>         Path to anndata .h5ad file for ST filtering. [default: '']

  --size_px=<n>               Output tile size in pixels. [default: 64]
  --size_um=<f>               Crop size in micrometers. [default: ]
  --mpp=<f>                   Resolution of the original WSI (Microns per pixel). [default: ]

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used PanNuke and MoNuSAC,
                              'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size per 1 GPU. [default: 32]

Two command mode are `tile` and `wsi` to enter corresponding inference mode
    tile  run the inference on tile
    wsi   run the inference on wsi

Use `run_infer.py <command> --help` to show their options and usage.
"""
from __future__ import annotations

tile_cli = """
Arguments for processing tiles.

usage:
    tile (--input_dir=<path>) (--output_dir=<path>) \
         [--draw_dot] [--save_qupath] [--save_raw_map] [--mem_usage=<n>]

options:
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --mem_usage=<n>        Declare how much memory (physical + swap) should be used for caching.
                          By default it will load as many tiles as possible till reaching the
                          declared limit. [default: 0.2]
   --draw_dot             To draw nuclei centroid on overlay. [default: False]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map         To save raw prediction or not. [default: False]
"""

wsi_cli = """
Arguments for processing wsi

usage:
    wsi (--input_dir=<path>) (--output_dir=<path>) (--image_dict_path=<path>) \
        [--input_mask_dir=<path>] [--cache_path=<path>] [--proc_mag=<n>] [--ambiguous_size=<n>] \
        [--chunk_shape=<n>] [--tile_shape=<n>] [--save_thumb] [--save_mask] [--save_geojson]

options:
    --input_dir=<path>       Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>      Path to output directory.
    --image_dict_path=<path> Path to save extracted image dictionary.
    --input_mask_dir=<path>  Path to directory containing tissue masks.
                             Should have the same name as corresponding WSIs. [default: '']
    --cache_path=<path>      Path for cache. Should be placed on SSD with at least 100GB. [default: cache]

    --proc_mag=<n>           Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>   Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>        Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>         Shape of tiles for processing. [default: 2048]
    --save_thumb             To save thumb. [default: False]
    --save_mask              To save mask. [default: False]
    --save_geojson           To save QuPath-compatible GeoJSON. [default: False]
"""

import torch
import logging
import os
import copy
from pathlib import Path
import json #added
from glob import glob #added
from misc.utils import log_info
from docopt import docopt

# -------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    sub_cli_dict = {"tile": tile_cli, "wsi": wsi_cli}
    args = docopt(__doc__, help=False, options_first=True, version="HoVer-Net Pytorch Inference v1.0")
    sub_cmd = args.pop("<command>")
    sub_cmd_args = args.pop("<args>")

    # ! TODO: where to save logging
    logging.basicConfig(
        level=logging.INFO,
        format="|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )

    if args["--help"] and sub_cmd is not None:
        if sub_cmd in sub_cli_dict:
            print(sub_cli_dict[sub_cmd])
        else:
            print(__doc__)
        exit()
    if args["--help"] or sub_cmd is None:
        print(__doc__)
        exit()

    sub_args = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_args, help=True)

    args.pop("--version")
    gpu_list = args.pop("--gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

    nr_gpus = torch.cuda.device_count()
    log_info("Detect #GPUS: %d" % nr_gpus)

    args = {k.replace("--", ""): v for k, v in args.items()}

    size_px = int(args.get("size_px", 64))

    size_um = args.get("size_um")
    size_um = float(size_um) if size_um not in (None, "") else None

    mpp = args.get("mpp")
    mpp = float(mpp) if mpp not in (None, "") else None

    # --- Consistency check ---
    if size_um is not None and mpp is None:
        raise ValueError(
            "Error: You specified --size_um but did not provide --mpp. "
            "Microns per pixel (mpp) is required to convert size_um into pixels. "
            "Segmentation will not run until both are provided."
        )

    sub_args = {k.replace("--", ""): v for k, v in sub_args.items()}
    if args["model_path"] == None:
        raise Exception("A model path must be supplied as an argument with --model_path.")

    nr_types = int(args["nr_types"]) if int(args["nr_types"]) > 0 else None
    method_args = {
        "method": {
            "model_args": {
                "nr_types": nr_types,
                "mode": args["model_mode"],
            },
            "model_path": args["model_path"],
        },
        "type_info_path": None if args["type_info_path"] == "" else args["type_info_path"],
    }

    # ***
    run_args = {
        "batch_size": int(args["batch_size"]) * nr_gpus,
        "nr_inference_workers": int(args["nr_inference_workers"]),
        "nr_post_proc_workers": int(args["nr_post_proc_workers"]),
    }

    if args["model_mode"] == "fast":
        run_args["patch_input_shape"] = 256
        run_args["patch_output_shape"] = 164
    else:
        run_args["patch_input_shape"] = 270
        run_args["patch_output_shape"] = 80

    if sub_cmd == "tile":
        run_args.update(
            {
                "input_dir": sub_args["input_dir"],
                "output_dir": sub_args["output_dir"],
                "mem_usage": float(sub_args["mem_usage"]),
                "draw_dot": sub_args["draw_dot"],
                "save_qupath": sub_args["save_qupath"],
                "save_raw_map": sub_args["save_raw_map"],
            }
        )

    if sub_cmd == "wsi":
        run_args.update(
            {
                "input_dir": sub_args["input_dir"],
                "output_dir": sub_args["output_dir"],
                "image_dict_path": sub_args["image_dict_path"],
                "input_mask_dir": sub_args["input_mask_dir"],
                "cache_path": sub_args["cache_path"],
                "proc_mag": int(sub_args["proc_mag"]),
                "ambiguous_size": int(sub_args["ambiguous_size"]),
                "chunk_shape": int(sub_args["chunk_shape"]),
                "tile_shape": int(sub_args["tile_shape"]),
                "save_thumb": sub_args["save_thumb"],
                "save_mask": sub_args["save_mask"],
                "save_geojson": sub_args["save_geojson"],
            }
        )
    # ***

    if sub_cmd == "tile":
        from infer.tile import InferManager

        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    else:
        out_path = Path(run_args["output_dir"])
        json_files = list(out_path.glob("*.json"))

        # ---- Segmentation (only if no JSON exists) ----
        if not json_files:
            from infer.wsi import InferManager
            from external.hovernet.seg_postprocessing import filter_by_st_proximity

            infer = InferManager(**method_args)
            infer.process_wsi_list(run_args)

            # refresh JSON list after segmentation
            json_files = list(out_path.glob("*.json"))

            # Process data
            if len(json_files) != 1:
                raise ValueError(f"Unexpected number of JSON files: {len(json_files)}")
            
            json_path = json_files[0]
            with open(json_path, "r") as f:
                data = json.load(f)

            if "nuc" not in data:
                raise KeyError(f"'nuc' key not found in {json_path}. Cannot reindex.")

            nuc_dict = data["nuc"]

            ## Optional ST filtering step
            adata_path = args.get("adata_path")
            # docopt fills the unset option with the literal default "''"; treat
            # empty / quote-only values as "no ST filtering".
            if adata_path in (None, "", "''", '""'):
                adata_path = None
            if adata_path and mpp:
                nuc_dict = filter_by_st_proximity(nuc_dict, adata_path, mpp)
            elif adata_path and not mpp:
                logging.warning("adata_path provided but mpp is missing. Skipping ST filtering.")

            ## Reindex the 'nuc' keys in the output JSON files
            new_nuc = {str(i): value for i, value in enumerate(nuc_dict.values())}
            data["nuc"] = new_nuc

            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

            logging.info(f"Reindexed 'nuc' keys in {json_path}")

            #-- Optional GeoJSON output for QuPath --
            if run_args.get("save_geojson"):
                from external.hovernet.seg_postprocessing import hovernet_to_geojson
                geojson_path = json_path.with_suffix(".geojson")
                hovernet_to_geojson(json_path, geojson_path)
                logging.info(f"GeoJSON saved to {geojson_path}")
            
        else:
            if len(json_files) != 1:
                raise ValueError(f"Unexpected number of JSON files: {len(json_files)}")
            
            json_path = json_files[0]
            logging.info("Existing JSON file(s) found. Skipping segmentation step.")

        ## Extract cell images
        from external.hovernet.seg_postprocessing import extract_images_hn
        
        input_path = Path(run_args["input_dir"])
        image_files = list(input_path.iterdir())

        if len(image_files) != 1:
            raise ValueError(f"Unexpected number of files: {len(image_files)}")

        image_path = image_files[0]

        try:
            image_dict = extract_images_hn(
                image_path=image_path,
                json_path=json_path,
                size_px=size_px,
                size_um=size_um,
                mpp=mpp,
                save_dict=os.path.join(run_args["image_dict_path"]),
            )
            logging.info(f"-> Image extraction completed successfully ({len(image_dict)} cells).")

        except Exception as e:
            raise ValueError(
                "Failed to extract images. Please check the image format.\n"
                "It must be in one of the following formats:\n"
                ".tif, .tiff, .svs, .dcm, or .ndpi.\n"
                "Also, ensure that the json_path is correct and contains "
                "valid segmentation data."
            ) from e
