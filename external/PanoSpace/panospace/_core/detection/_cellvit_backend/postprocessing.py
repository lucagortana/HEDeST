# -*- coding: utf-8 -*-
# Post-processing utilities for CellViT cell instance segmentation.
#
# Handles watershed-based instance separation, small object removal,
# and packaging results into instance-wise dictionaries.
#
# @ PanoSpace adaptation
from __future__ import annotations

import logging
from os import environ
from typing import Dict
from typing import List
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import ray
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

logger = logging.getLogger(__name__)

from numba import jit
from scipy.ndimage import binary_fill_holes, measurements
from skimage.segmentation import watershed

from shapely.geometry import Polygon, MultiPolygon
from shapely import strtree
from collections import deque

from .tools import get_bounding_box, remove_small_objects, remap_label
from scipy.ndimage import label


class DetectionCellPostProcessor:
    def __init__(
        self,
        nr_types: int,
    ) -> None:
        """DetectionCellPostProcessor for postprocessing prediction maps and get detected cells, based on cupy

        Args:
            nr_types (int):  Number of cell types, including background (background = 0). Defaults to None.

        Raises:
            NotImplementedError: Unknown
        """
        self.nr_types = nr_types

    def check_network_output(self, predictions_: dict) -> None:
        """Check if the network output is valid

        Args:
            predictions_ (dict): Network predictions with tokens. Keys (required):
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, H, W, 2)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)
        """
        b, h, w, _ = predictions_["nuclei_binary_map"].shape
        assert isinstance(predictions_, dict), "predictions_ must be a dictionary"
        assert "nuclei_binary_map" in predictions_, "nuclei_binary_map must be in predictions_"
        assert "nuclei_type_map" in predictions_, "nuclei_binary_map must be in predictions_"
        assert "hv_map" in predictions_, "nuclei_binary_map must be in predictions_"
        assert predictions_["nuclei_binary_map"].shape == (
            b,
            h,
            w,
            2,
        ), "nuclei_binary_map must have shape (B, H, W, 2)"
        assert predictions_["nuclei_type_map"].shape == (
            b,
            h,
            w,
            self.nr_types,
        ), "nuclei_type_map must have shape (B, H, W, self.nr_types)"
        assert predictions_["hv_map"].shape == (
            b,
            h,
            w,
            2,
        ), "hv_map must have shape (B, H, W, 2)"

    def post_process_batch(self, predictions_: dict) -> Tuple[torch.Tensor, List[dict]]:
        """Post process a batch of predictions and generate cell dictionary and instance predictions for each image in a list

        Args:
            predictions_ (dict): Network predictions with tokens. Keys (required):
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, H, W, 2)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
        """
        b, h, w, _ = predictions_["nuclei_binary_map"].shape
        # checking
        self.check_network_output(predictions_)

        # batch wise
        pred_maps = self._prepare_pred_maps(predictions_)

        # image wise
        cell_dicts = []
        instance_predictions = []
        for i in range(b):
            pred_inst, cells = self.post_process_single_image(pred_maps[i])
            instance_predictions.append(pred_inst)
            cell_dicts.append(cells)

        return torch.Tensor(np.stack(instance_predictions)), cell_dicts

    def post_process_single_image(self, pred_map: np.ndarray) -> Tuple[np.ndarray, dict[int, dict]]:
        """Process one single image and generate cell dictionary and instance predictions

        Args:
            pred_map (np.ndarray): Combined output of tp, np and hv branches, in the same order. Shape: (H, W, 4)
        Returns:
            Tuple[np.ndarray, dict[int, dict]]: _description_
        """
        pred_inst, pred_type = self._get_pred_inst_tensor(pred_map)
        cells = self._create_cell_dict(pred_inst, pred_type)
        return (pred_inst, cells)

    def _prepare_pred_maps(self, predictions_: dict) -> np.ndarray:
        """Prepares the prediction maps for post-processing.

        This function takes a dictionary of PyTorch tensors, clones it,
        moves the tensors to the CPU, converts them to numpy arrays, and
        then stacks them along the last axis.

        Args:
            predictions_ (Dict[str, torch.Tensor]): Network predictions with tokens. Keys (required):
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, H, W, 2)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)

        Returns:
            np.ndarray: A numpy array containing the stacked prediction maps.
                * shape: B, H, W, 4
                * The last dimension contains the following maps:
                    * channel 0: Type prediction of nuclei
                    * channel 1: Binary Nucleus Predictions
                    * channel 2: Horizontal-Vertical nuclei mapping (X)
                    * channel 3: Horizontal-Vertical nuclei mapping (Y)
        """
        predictions = predictions_.copy()
        predictions["nuclei_type_map"] = predictions["nuclei_type_map"].detach().cpu().numpy()
        predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].detach().cpu().numpy()
        predictions["hv_map"] = predictions["hv_map"].detach().cpu().numpy()

        return self._stack_pred_maps(
            predictions["nuclei_type_map"],
            predictions["nuclei_binary_map"],
            predictions["hv_map"],
        )

    def _stack_pred_maps(
        self,
        nuclei_type_map: np.ndarray,
        nuclei_binary_map: np.ndarray,
        hv_map: np.ndarray,
    ) -> np.ndarray:
        """Creates the prediction map for HoVer-Net post-processing

        Args:
        nuclei_binary_map:
            nuclei_type_map (np.ndarray):  Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
            nuclei_binary_map (np.ndarray): Binary Nucleus Predictions. Shape: (B, H, W, 2)
            hv_map (np.ndarray): Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)

        Returns:
            np.ndarray: A numpy array containing the stacked prediction maps. Shape [B, H, W, 4]
        """
        # Assert that the shapes of the inputs are as expected
        assert nuclei_type_map.ndim == 4, "nuclei_type_map must be a 4-dimensional array"
        assert nuclei_binary_map.ndim == 4, "nuclei_binary_map must be a 4-dimensional array"
        assert hv_map.ndim == 4, "hv_map must be a 4-dimensional array"
        assert (
            nuclei_type_map.shape[:-1] == nuclei_binary_map.shape[:-1] == hv_map.shape[:-1]
        ), "The first three dimensions of all input arrays must be the same"
        assert nuclei_binary_map.shape[-1] == 2, "The last dimension of nuclei_binary_map must have a size of 2"
        assert hv_map.shape[-1] == 2, "The last dimension of hv_map must have a size of 2"
        assert isinstance(nuclei_type_map, np.ndarray), "nuclei_type_map must be a cupy array"
        assert isinstance(nuclei_binary_map, np.ndarray), "nuclei_binary_map must be a cupy array"
        assert isinstance(hv_map, np.ndarray), "hv_map must be a cupy array"

        nuclei_type_map = np.argmax(nuclei_type_map, axis=-1)  # argmax: cupy argmax
        nuclei_binary_map = np.argmax(nuclei_binary_map, axis=-1)  # argmax: cupy argmax
        pred_map = np.stack(
            (nuclei_type_map, nuclei_binary_map, hv_map[..., 0], hv_map[..., 1]),
            axis=-1,
        )

        assert pred_map.shape[-1] == 4, "The last dimension of pred_map must have a size of 4"

        return pred_map

    def _get_pred_inst_tensor(
        self,
        pred_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process Nuclei Prediction and generate instance map (each instance has unique integer)

        Args:
            pred_map (np.ndarray): Combined output of tp, np and hv branches, in the same order. Shape: (H, W, 4)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                * np.ndarray: Instance array with shape (H, W), each instance has unique integer
                * np.ndarray: Type array with shape (H, W), each pixel has the type of the instance
        """
        assert isinstance(pred_map, np.ndarray), "pred_map must be a numpy array"
        assert pred_map.ndim == 3, "pred_map must be a 3-dimensional array"
        assert pred_map.shape[-1] == 4, "The last dimension of pred_map must have a size of 4"

        pred_type = pred_map[..., :1]
        pred_inst = pred_map[..., 1:]
        pred_type = pred_type.astype(np.int32)

        pred_inst = np.squeeze(pred_inst)
        pred_inst = remap_label(self._proc_np_hv(pred_inst))

        # return as numpy array
        return pred_inst, pred_type.squeeze()

    def _proc_np_hv(self, pred_inst: np.ndarray, object_size: int = 10, ksize: int = 21) -> np.ndarray:
        """Process Nuclei Prediction with XY Coordinate Map and generate instance map (each instance has unique integer)

        Separate Instances (also overlapping ones) from binary nuclei map and hv map by using morphological operations and watershed

        Args:
            pred (np.ndarray): Prediction output, assuming. Shape: (H, W, 3)
                * channel 0 contain probability map of nuclei
                * channel 1 containing the regressed X-map
                * channel 2 containing the regressed Y-map
            object_size (int, optional): Smallest oject size for filtering. Defaults to 10
            k_size (int, optional): Sobel Kernel size. Defaults to 21

        Returns:
            np.ndarray: Instance map for one image. Each nuclei has own integer. Shape: (H, W)
        """

        # Check input types and values
        assert isinstance(pred_inst, np.ndarray), "pred_inst must be a numpy array"
        assert pred_inst.ndim == 3, "pred_inst must be a 3-dimensional array"
        assert pred_inst.shape[2] == 3, "The last dimension of pred_inst must have a size of 3"
        assert isinstance(object_size, int), "object_size must be an integer"
        assert object_size > 0, "object_size must be greater than 0"
        assert isinstance(ksize, int), "ksize must be an integer"
        assert ksize > 0, "ksize must be greater than 0"

        # ensure dtype and extract individual channels
        pred = np.array(pred_inst, dtype=np.float32)
        blb_raw = pred[..., 0]
        h_dir_raw = pred[..., 1]
        v_dir_raw = pred[..., 2]

        blb = np.array(blb_raw >= 0.5, dtype=np.int32)
        blb = label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1  # background is 0 already

        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        # Apply Sobel filter to the direction maps
        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        # Combine the Sobel filtered images
        overall = np.maximum(np.asarray(sobelh), np.asarray(sobelv))
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        # Create distance map
        dist = (1.0 - overall) * blb
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)
        marker = blb - overall
        marker[marker < 0] = 0

        # Apply all
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = label(np.asarray(marker))[0]
        marker = remove_small_objects(marker, min_size=object_size)

        # Separate instances
        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred

    ### Methods related to cell dictionary
    def _create_cell_dict(self, pred_inst: np.ndarray, pred_type: np.ndarray) -> dict[int, dict]:
        """Create cell dictionary from instance and type predictions

        Keys of the dictionary:
            * bbox: Bounding box of the cell
            * centroid: Centroid of the cell
            * contour: Contour of the cell
            * type_prob: Probability of the cell type
            * type: Type of the cell

        Args:
            pred_inst (np.ndarray): Instance array with shape (H, W), each instance has unique integer
            pred_type (np.ndarray): Type array with shape (H, W), each pixel has the type of the instance

        Returns:
            dict [int, dict]: Dictionary containing the cell information
        """
        assert isinstance(pred_inst, np.ndarray), "pred_inst must be a numpy array"
        assert pred_inst.ndim == 2, "pred_inst must be a 2-dimensional array"
        assert isinstance(pred_type, np.ndarray), "pred_type must be a numpy array"
        assert pred_type.ndim == 2, "pred_type must be a 2-dimensional array"
        assert pred_inst.shape == pred_type.shape, "pred_inst and pred_type must have the same shape"

        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}

        for inst_id in inst_id_list:
            inst_id, cell_dict = self._create_single_instance_entry(inst_id, pred_inst, pred_type)
            if cell_dict is not None:
                inst_info_dict[inst_id] = cell_dict

        return inst_info_dict

    def _create_single_instance_entry(
        self, inst_id: int, pred_inst: np.ndarray, pred_type: np.ndarray
    ) -> Tuple[int, dict]:
        """Create a single cell dictionary entry from instance and type predictions

        Args:
            inst_id (int): _description_
            pred_inst (np.ndarray): Instance array with shape (H, W), each instance has unique integer
            pred_type (np.ndarray): Type array with shape (H, W), each pixel has the type of the instance

        Returns:
            Tuple[int, dict]:
                * int: Instance ID
                * dict: Dictionary containing the cell information
                    Keys are: "bbox", "centroid", "contour", "type_prob", "type"
        """
        inst_map_global = pred_inst == inst_id
        inst_bbox = self._get_instance_bbox(inst_map_global)
        inst_map_local = self._get_local_instance_map(inst_map_global, inst_bbox)
        inst_centroid_local, inst_contour_local = self._get_instance_centroid_contour(inst_map_local)

        if inst_centroid_local is None:
            return inst_id, None

        inst_centroid, inst_contour = self._correct_instance_position(
            inst_centroid_local, inst_contour_local, inst_bbox
        )
        inst_type, inst_type_prob = self._get_instance_type(inst_bbox, pred_type, inst_map_local)

        return inst_id, {  # inst_id should start at 1
            "bbox": inst_bbox,
            "centroid": inst_centroid,
            "contour": inst_contour,
            "type_prob": inst_type_prob,
            "type": inst_type,
        }

    def _get_instance_bbox(self, inst_map_global: np.ndarray) -> np.ndarray:
        """Get the bounding box of an instance from global instance map (instance map is binary)

        Args:
            inst_map_global (np.ndarray): Binary instance map, Shape: (H, W)

        Returns:
            np.ndarray: Bounding box of the instance. Shape: (2, 2)
                Interpretation: [[rmin, cmin], [rmax, cmax]]
        """
        rmin, rmax, cmin, cmax = get_bounding_box(inst_map_global)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        return inst_bbox

    def _get_local_instance_map(self, inst_map_global: np.ndarray, inst_bbox: np.ndarray) -> np.ndarray:
        """Get the local instance map from the global instance map, crop it with the bounding box

        Args:
            inst_map_global (np.ndarray): Binary instance map, Shape: (H, W)
            inst_bbox (np.ndarray): Bounding box of the instance. Shape: (2, 2)

        Returns:
            np.ndarray: Local instance map. Shape: (H', W')
        """
        inst_map_local = inst_map_global[inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]]
        inst_map_local = inst_map_local.astype(np.uint8)
        return inst_map_local

    def _get_instance_centroid_contour(self, inst_map_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get the centroid and contour of an instance from the local instance map

        Coordinates are relative to the local instance map

        Args:
            inst_map_local (np.ndarray): Local instance map. Shape: (H', W')

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                * np.ndarray: Centroid of the instance. Shape: (2,)
                * np.ndarray: Contour of the instance. Shape: (N, 2)
        """
        inst_moment = cv2.moments(inst_map_local)
        inst_contour = cv2.findContours(inst_map_local, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))

        if inst_contour.shape[0] < 3 or len(inst_contour.shape) != 2:
            return None, None

        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)

        return inst_centroid, inst_contour

    def _correct_instance_position(
        self, inst_centroid: np.ndarray, inst_contour: np.ndarray, inst_bbox: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct the position of the centroid and contour of an instance to the global image

        Args:
            inst_centroid (np.ndarray): Centroid of the instance. Shape: (2,)
            inst_contour (np.ndarray): Contour of the instance. Shape: (N, 2)
            inst_bbox (np.ndarray): Bounding box of the instance. Shape: (2, 2)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                * np.ndarray: Centroid of the instance (global cs). Shape: (2,)
                * np.ndarray: Contour of the instance (global cs). Shape: (N, 2)
        """
        inst_contour[:, 0] += inst_bbox[0][1]  # X
        inst_contour[:, 1] += inst_bbox[0][0]  # Y
        inst_centroid[0] += inst_bbox[0][1]  # X
        inst_centroid[1] += inst_bbox[0][0]  # Y

        return inst_centroid, inst_contour

    def _get_instance_type(
        self, inst_bbox: np.ndarray, pred_type: np.ndarray, inst_map_local: np.ndarray
    ) -> Tuple[int, float]:
        """Get the type of an instance from the local instance map and the type prediction map

        Args:
            inst_bbox (np.ndarray): Bounding box of the instance. Shape: (2, 2)
            pred_type (np.ndarray): Type prediction of nuclei. Shape: (H, W)
            inst_map_local (np.ndarray): Local instance map. Shape: (H', W')

        Returns:
            Tuple[int, float]:
                * int: Type of the instance
                * float: Probability of the instance type
        """
        inst_map_local = inst_map_local.astype(bool)
        inst_type_local = pred_type[inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]][
            inst_map_local
        ]
        type_list, type_pixels = np.unique(inst_type_local, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]

        # if type is background, select, the 2nd most dominant if exist
        if inst_type == 0:
            if len(type_list) > 1:
                inst_type = type_list[1][0]
        type_dict = {v[0]: v[1] for v in type_list}
        type_prob = type_dict[inst_type] / (np.sum(inst_map_local) + 1.0e-6)

        return int(inst_type), float(type_prob)


def create_batch_pooling_actor(num_cpus: int = 8):
    num_gpus = 0.1
    if environ.get("RAY_GPUS_DEACTIVATION") is not None:
        if environ.get("RAY_GPUS_DEACTIVATION") == "1":
            num_gpus = 0

    @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
    class BatchPoolingActor:
        def __init__(
            self,
            detection_cell_postprocessor: DetectionCellPostProcessor,
            run_conf: dict,
        ) -> None:
            """Ray Actor for coordinating the postprocessing of **one** batch

            The postprocessing is done in a separate process to avoid blocking the main process.
            The calculation is done with the help of the `DetectionCellPostProcessorCupy` class.
            This actor acts as a coordinator for the postprocessing of one batch and a wrapper for the `DetectionCellPostProcessorCupy` class.

            Args:
                detection_cell_postprocessor (DetectionCellPostProcessorCupy): Instance of the `DetectionCellPostProcessorCupy` class
                run_conf (dict): Run configuration
            """
            assert "dataset_config" in run_conf, "dataset_config must be in run_conf"
            assert "nuclei_types" in run_conf["dataset_config"], "nuclei_types must be in run_conf['dataset_config']"
            assert "model" in run_conf, "model must be in run_conf"
            assert "token_patch_size" in run_conf["model"], "token_patch_size must be in run_conf['model']"

            self.detection_cell_postprocessor = detection_cell_postprocessor
            self.run_conf = run_conf

        def convert_batch_to_graph_nodes(
            self, predictions: dict, metadata: List[dict]
        ) -> Tuple[List[dict], List[dict], List[torch.Tensor], List[torch.Tensor]]:
            """Postprocess a batch of predictions and convert it to graph nodes

            Returns the complete graph nodes (cell dictionary), the detection nodes (cell detection dictionary), the cell tokens and the cell positions


            Args:
                predictions (dict): predictions_ (dict): Network predictions with tokens. Keys (required):
                    * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, H, W, 2)
                    * nuclei_type_map: Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
                    * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)
                metadata List[(dict)]: List of metadata dictionaries for each patch.
                    Each dictionary needs to contain the following keys:
                    * row: Row index of the patch
                    * col: Column index of the patch
                    Other keys are optional

            Returns:
                Tuple[List[dict], List[dict], List[torch.Tensor], List[torch.Tensor]]:
                    * List[dict]: Complete graph nodes (cell dictionary)
                    * List[dict]: Detection nodes (cell detection dictionary)
                    * List[torch.Tensor]: Cell tokens
                    * List[torch.Tensor]: Cell positions (centroid)
            """
            _, cell_dict_batch = self.detection_cell_postprocessor.post_process_batch(predictions)
            tokens = predictions["tokens"].detach().to("cpu")

            batch_complete = []
            batch_detection = []
            batch_cell_tokens = []
            batch_cell_positions = []

            for idx, (patch_cell_dict, patch_metadata) in enumerate(zip(cell_dict_batch, metadata)):
                (
                    patch_complete,
                    patch_detection,
                    patch_cell_tokens,
                    patch_cell_positions,
                ) = self.convert_patch_to_graph_nodes(patch_cell_dict, patch_metadata, tokens[idx])
                batch_complete = batch_complete + patch_complete
                batch_detection = batch_detection + patch_detection
                batch_cell_tokens = batch_cell_tokens + patch_cell_tokens
                batch_cell_positions = batch_cell_positions + patch_cell_positions

            if self.detection_cell_postprocessor.classifier is not None:
                if len(batch_cell_tokens) > 0:
                    batch_cell_tokens_pt = torch.stack(batch_cell_tokens)
                    updated_preds = self.detection_cell_postprocessor.classifier(batch_cell_tokens_pt)
                    updated_preds = F.softmax(updated_preds, dim=1)
                    updated_classes = torch.argmax(updated_preds, dim=1)
                    updated_class_preds = updated_preds[torch.arange(updated_classes.shape[0]), updated_classes]

                    for f, z in zip(batch_complete, updated_classes):
                        f["type"] = int(z)
                    for f, z in zip(batch_complete, updated_class_preds):
                        f["type_prob"] = int(z)
                    for f, z in zip(batch_detection, updated_classes):
                        f["type"] = int(z)
            if self.detection_cell_postprocessor.binary:
                for f in batch_complete:
                    f["type"] = 1
                for f in batch_detection:
                    f["type"] = 1
                pass

            return (
                batch_complete,
                batch_detection,
                batch_cell_tokens,
                batch_cell_positions,
            )

        def convert_patch_to_graph_nodes(
            self,
            patch_cell_dict: dict,
            patch_metadata: dict,
            patch_tokens: torch.Tensor,
        ) -> Tuple[List[dict], List[dict], List[torch.Tensor], List[torch.Tensor]]:
            """Extract information from a single patch and convert it to graph nodes for a global view

            Args:
                patch_cell_dict (dict): Dictionary containing the cell information.
                    Each dictionary needs to contain the following keys:
                    * bbox: Bounding box of the cell
                    * centroid: Centroid of the cell
                    * contour: Contour of the cell
                    * type_prob: Probability of the cell type
                    * type: Type of the cell
                patch_metadata (dict): Metadata dictionary for the patch.
                    Each dictionary needs to contain the following keys:
                    * row: Row index of the patch
                    * col: Column index of the patch
                    Other keys are optional but are stored in the graph nodes for later use
                patch_tokens (torch.Tensor): Tokens of the patch. Shape: (D, H, W)

            Returns:
                Tuple[List[dict], List[dict], List[torch.Tensor], List[torch.Tensor]]:
                    * List[dict]: Complete graph nodes (cell dictionary) of the patch
                    * List[dict]: Detection nodes (cell detection dictionary) of the patch
                    * List[torch.Tensor]: Cell tokens of the patch
                    * List[torch.Tensor]: Cell positions (centroid) of the patch
            """
            wsi = self.detection_cell_postprocessor.wsi
            patch_cell_detection = {}
            patch_cell_detection["patch_metadata"] = patch_metadata
            patch_cell_detection["type_map"] = self.run_conf["dataset_config"]["nuclei_types"]

            wsi_scaling_factor = wsi.metadata["downsampling"]
            patch_size = wsi.metadata["patch_size"]
            x_global = int(
                patch_metadata["row"] * patch_size - (patch_metadata["row"] + 0.5) * wsi.metadata["patch_overlap"]
            )
            y_global = int(
                patch_metadata["col"] * patch_size - (patch_metadata["col"] + 0.5) * wsi.metadata["patch_overlap"]
            )

            cell_tokens = []
            cell_positions = []
            cell_complete = []
            cell_detections = []

            # extract cell information
            for cell in patch_cell_dict.values():
                if cell["type"] == self.run_conf["dataset_config"]["nuclei_types"]["Background"]:
                    continue
                offset_global = np.array([x_global, y_global])
                centroid_global = np.rint((cell["centroid"] + np.flip(offset_global)) * wsi_scaling_factor)
                contour_global = (cell["contour"] + np.flip(offset_global)) * wsi_scaling_factor
                bbox_global = (cell["bbox"] + offset_global) * wsi_scaling_factor
                cell_dict = {
                    "bbox": bbox_global.tolist(),
                    "centroid": centroid_global.tolist(),
                    "contour": contour_global.tolist(),
                    "type_prob": cell["type_prob"],
                    "type": cell["type"],
                    "patch_coordinates": [
                        patch_metadata["row"],
                        patch_metadata["col"],
                    ],
                    "cell_status": get_cell_position_marging(
                        bbox=cell["bbox"],
                        patch_size=wsi.metadata["patch_size"],
                        margin=64,
                    ),
                    "offset_global": offset_global.tolist(),
                }
                cell_detection = {
                    "bbox": bbox_global.tolist(),
                    "centroid": centroid_global.tolist(),
                    "type": cell["type"],
                }
                if (
                    np.max(cell["bbox"]) == wsi.metadata["patch_size"] or np.min(cell["bbox"]) == 0
                ):  # Use overlap and patch size
                    position = get_cell_position(cell["bbox"], wsi.metadata["patch_size"])
                    cell_dict["edge_position"] = True
                    cell_dict["edge_information"] = {}
                    cell_dict["edge_information"]["position"] = position
                    cell_dict["edge_information"]["edge_patches"] = get_edge_patch(
                        position, patch_metadata["row"], patch_metadata["col"]
                    )
                else:
                    cell_dict["edge_position"] = False

                bb_index = cell["bbox"] / self.run_conf["model"]["token_patch_size"]
                bb_index[0, :] = np.floor(bb_index[0, :])
                bb_index[1, :] = np.ceil(bb_index[1, :])
                bb_index = bb_index.astype(np.uint8)
                cell_token = patch_tokens[:, bb_index[0, 0] : bb_index[1, 0], bb_index[0, 1] : bb_index[1, 1]]
                cell_token = torch.mean(rearrange(cell_token, "D H W -> (H W) D"), dim=0)

                cell_tokens.append(cell_token)
                cell_positions.append(torch.Tensor(centroid_global))
                cell_complete.append(cell_dict)
                cell_detections.append(cell_detection)

            return cell_complete, cell_detections, cell_tokens, cell_positions

    return BatchPoolingActor


def get_cell_position(bbox: np.ndarray, patch_size: int = 1024) -> List[int]:
    """Get cell position as a list

    Entry is 1, if cell touches the border: [top, right, down, left]

    Args:
        bbox (np.ndarray): Bounding-Box of cell
        patch_size (int, optional): Patch-size. Defaults to 1024.

    Returns:
        List[int]: List with 4 integers for each position
    """
    # bbox = 2x2 array in h, w style
    # bbox[0,0] = upper position (height)
    # bbox[1,0] = lower dimension (height)
    # boox[0,1] = left position (width)
    # bbox[1,1] = right position (width)
    # bbox[:,0] -> x dimensions
    top, left, down, right = False, False, False, False
    if bbox[0, 0] == 0:
        top = True
    if bbox[0, 1] == 0:
        left = True
    if bbox[1, 0] == patch_size:
        down = True
    if bbox[1, 1] == patch_size:
        right = True
    position = [top, right, down, left]
    position = [int(pos) for pos in position]

    return position


def get_cell_position_marging(bbox: np.ndarray, patch_size: int = 1024, margin: int = 64) -> int:
    """Get the status of the cell, describing the cell position

    A cell is either in the mid (0) or at one of the borders (1-8)

    # Numbers are assigned clockwise, starting from top left
    # i.e., top left = 1, top = 2, top right = 3, right = 4, bottom right = 5 bottom = 6, bottom left = 7, left = 8
    # Mid status is denoted by 0

    Args:
        bbox (np.ndarray): Bounding Box of cell
        patch_size (int, optional): Patch-Size. Defaults to 1024.
        margin (int, optional): Margin-Size. Defaults to 64.

    Returns:
        int: Cell Status
    """
    cell_status = None
    if np.max(bbox) > patch_size - margin or np.min(bbox) < margin:
        if bbox[0, 0] < margin:
            # top left, top or top right
            if bbox[0, 1] < margin:
                # top left
                cell_status = 1
            elif bbox[1, 1] > patch_size - margin:
                # top right
                cell_status = 3
            else:
                # top
                cell_status = 2
        elif bbox[1, 1] > patch_size - margin:
            # top right, right or bottom right
            if bbox[1, 0] > patch_size - margin:
                # bottom right
                cell_status = 5
            else:
                # right
                cell_status = 4
        elif bbox[1, 0] > patch_size - margin:
            # bottom right, bottom, bottom left
            if bbox[0, 1] < margin:
                # bottom left
                cell_status = 7
            else:
                # bottom
                cell_status = 6
        elif bbox[0, 1] < margin:
            # bottom left, left, top left, but only left is left
            cell_status = 8
    else:
        cell_status = 0

    return cell_status


def get_edge_patch(position: List[int], row: int, col: int) -> List[List[int]]:
    """Get the edge patches of a cell located at the border

    Args:
        position (List[int]): Position of the cell encoded as a list
            -> See below for a list of positions (1-8)
        row (int): Row position of the patch
        col (int): Col position of the patch

    Returns:
        List[List[int]]: List of edge patches, each patch encoded as list of row and col
    """
    # row starting on bottom or on top?
    if position == [1, 0, 0, 0]:
        # top
        return [[row - 1, col]]
    if position == [1, 1, 0, 0]:
        # top and right
        return [[row - 1, col], [row - 1, col + 1], [row, col + 1]]
    if position == [0, 1, 0, 0]:
        # right
        return [[row, col + 1]]
    if position == [0, 1, 1, 0]:
        # right and down
        return [[row, col + 1], [row + 1, col + 1], [row + 1, col]]
    if position == [0, 0, 1, 0]:
        # down
        return [[row + 1, col]]
    if position == [0, 0, 1, 1]:
        # down and left
        return [[row + 1, col], [row + 1, col - 1], [row, col - 1]]
    if position == [0, 0, 0, 1]:
        # left
        return [[row, col - 1]]
    if position == [1, 0, 0, 1]:
        # left and top
        return [[row, col - 1], [row - 1, col - 1], [row - 1, col]]


def process_cell_instance(
    instance_types,
    offset_global,
    row,
    col,
    tile_size=256,
    overlap=64,
):

    all_cell_dicts = []
    all_cell_detections = []
    for cell in instance_types.values():
        # Calculate global coordinates
        centroid_global = cell["centroid"] + np.flip(offset_global)
        contour_global = cell["contour"] + np.flip(offset_global)
        bbox_global = cell["bbox"] + offset_global

        # Build cell information dictionary
        cell_dict = {
            "bbox": bbox_global.tolist(),
            "centroid": centroid_global.tolist(),
            "contour": contour_global.tolist(),
            "type": cell["type"],
            "type_prob": float(cell.get("type_prob", 1.0)),
            "patch_coordinates": [row, col],
            "cell_status": get_cell_position_marging(cell["bbox"], tile_size, overlap),
            "offset_global": offset_global.tolist(),
        }

        # If bbox extends beyond patch, mark as boundary cell
        if np.max(cell["bbox"]) == tile_size or np.min(cell["bbox"]) == 0:
            position = get_cell_position(cell["bbox"], tile_size)
            cell_dict["edge_position"] = True
            cell_dict["edge_information"] = {}
            cell_dict["edge_information"]["position"] = position
            cell_dict["edge_information"]["edge_patches"] = get_edge_patch(position, row, col)
        else:
            cell_dict["edge_position"] = False

        # Build simple detection information dictionary
        cell_detection = {
            "bbox": bbox_global.tolist(),
            "centroid": centroid_global.tolist(),
            "type": cell["type"],
        }
        all_cell_dicts.append(cell_dict)
        all_cell_detections.append(cell_detection)

    # Return list of all cell information
    return all_cell_dicts, all_cell_detections


# @jit(nopython=True)
# def stack_pred_maps(
#     nuclei_type_map: np.ndarray, nuclei_binary_map: np.ndarray, hv_map: np.ndarray
# ) -> np.ndarray:
#     """Creates the prediction map for HoVer-Net post-processing

#     Args:
#     nuclei_binary_map:
#         nuclei_type_map (np.ndarray):  Type prediction of nuclei. Shape: (B, H, W, self.num_nuclei_classes,)
#         nuclei_binary_map (np.ndarray): Binary Nucleus Predictions. Shape: (B, H, W, 2)
#         hv_map (np.ndarray): Horizontal-Vertical nuclei mapping. Shape: (B, H, W, 2)

#     Returns:
#         np.ndarray: A numpy array containing the stacked prediction maps. Shape [B, H, W, 4]
#     """
#     # Assert that the shapes of the inputs are as expected
#     assert nuclei_type_map.ndim == 4, "nuclei_type_map must be a 4-dimensional array"
#     assert (
#         nuclei_binary_map.ndim == 4
#     ), "nuclei_binary_map must be a 4-dimensional array"
#     assert hv_map.ndim == 4, "hv_map must be a 4-dimensional array"
#     assert (
#         nuclei_type_map.shape[:-1] == nuclei_binary_map.shape[:-1] == hv_map.shape[:-1]
#     ), "The first three dimensions of all input arrays must be the same"
#     assert (
#         nuclei_binary_map.shape[-1] == 2
#     ), "The last dimension of nuclei_binary_map must have a size of 2"
#     assert hv_map.shape[-1] == 2, "The last dimension of hv_map must have a size of 2"

#     nuclei_type_map = np.argmax(nuclei_type_map, axis=-1)
#     nuclei_binary_map = np.argmax(nuclei_binary_map, axis=-1)
#     pred_map = np.stack(
#         (nuclei_type_map, nuclei_binary_map, hv_map[..., 0], hv_map[..., 1]), axis=-1
#     )

#     assert (
#         pred_map.shape[-1] == 4
#     ), "The last dimension of pred_map must have a size of 4"

#     return pred_map


def convert_coordinates(row: pd.Series) -> pd.Series:
    """Convert a row from x,y type to one string representation of the patch position for fast querying
    Repr: x_y

    Args:
        row (pd.Series): Row to be processed

    Returns:
        pd.Series: Processed Row
    """
    x, y = row["patch_coordinates"]
    row["patch_row"] = x
    row["patch_col"] = y
    row["patch_coordinates"] = f"{x}_{y}"
    return row


class CellPostProcessor:
    def __init__(self, cell_list: List[dict], logger=None) -> None:
        self.cell_df = pd.DataFrame(cell_list)
        self.cell_df = self.cell_df.apply(convert_coordinates, axis=1)
        self.mid_cells = self.cell_df[self.cell_df["cell_status"] == 0]
        self.cell_df_margin = self.cell_df[self.cell_df["cell_status"] != 0]
        self.logger = logger

    def post_process_cells(self) -> pd.DataFrame:

        cleaned_edge_cells = self._clean_edge_cells()
        cleaned_edge_cells = self._remove_overlap(cleaned_edge_cells)

        # merge with mid cells
        postprocessed_cells = pd.concat([self.mid_cells, cleaned_edge_cells]).sort_index()
        return postprocessed_cells

    def _clean_edge_cells(self) -> pd.DataFrame:

        margin_cells = self.cell_df_margin[
            self.cell_df_margin["edge_position"] == 0
        ]  # cells at the margin, but not touching the border
        edge_cells = self.cell_df_margin[self.cell_df_margin["edge_position"] == 1]  # cells touching the border
        existing_patches = list(set(self.cell_df_margin["patch_coordinates"].to_list()))

        edge_cells_unique = pd.DataFrame(
            columns=self.cell_df_margin.columns
        )  # cells torching the border without having an overlap from other patches

        for idx, cell_info in tqdm(edge_cells.iterrows(), total=len(edge_cells)):
            edge_information = dict(cell_info["edge_information"])
            edge_patch = edge_information["edge_patches"][0]
            edge_patch = f"{edge_patch[0]}_{edge_patch[1]}"
            if edge_patch not in existing_patches:
                edge_cells_unique.loc[idx, :] = cell_info

        cleaned_edge_cells = pd.concat([margin_cells, edge_cells_unique])

        return cleaned_edge_cells.sort_index()

    # def _remove_overlap(self, cleaned_edge_cells: pd.DataFrame) -> pd.DataFrame:

    #     merged_cells = cleaned_edge_cells

    #     for iteration in range(20):
    #         poly_list = []
    #         uid_to_poly = {}
    #         for idx, cell_info in merged_cells.iterrows():
    #             if len(cell_info["contour"]) < 4:
    #                 continue

    #             poly = Polygon(cell_info["contour"])
    #             if not poly.is_valid:
    #                 multi = poly.buffer(0)
    #                 if isinstance(multi, MultiPolygon):
    #                     if len(multi.geoms) > 1:
    #                         poly_idx = np.argmax([p.area for p in multi.geoms])
    #                         poly = multi.geoms[poly_idx]
    #                         poly = Polygon(poly)
    #                     else:
    #                         poly = multi.geoms[0]
    #                         poly = Polygon(poly)
    #                 else:
    #                     poly = Polygon(multi)

    #             # poly = PolygonWithID(poly, uid=idx)
    #             # poly.uid = idx
    #             poly_list.append(poly)
    #             uid_to_poly[idx] = poly

    #         # use an strtree for fast querying
    #         tree = strtree.STRtree(poly_list)
    #         poly_to_uid = {v: k for k, v in uid_to_poly.items()}

    #         merged_idx = deque()
    #         iterated_cells = set()
    #         overlaps = 0

    #         for query_poly in poly_list:
    #             uid = poly_to_uid[poly]
    #             if uid not in iterated_cells:
    #                 intersected_polygons = tree.query(
    #                     query_poly
    #                 )  # this also contains a self-intersection
    #                 if (
    #                     len(intersected_polygons) > 1
    #                 ):  # we have more at least one intersection with another cell
    #                     submergers = []  # all cells that overlap with query
    #                     for inter_poly in intersected_polygons:
    #                         if (
    #                             inter_poly.uid != query_poly.uid
    #                             and inter_poly.uid not in iterated_cells
    #                         ):
    #                             if (
    #                                 query_poly.intersection(inter_poly).area
    #                                 / query_poly.area
    #                                 > 0.01
    #                                 or query_poly.intersection(inter_poly).area
    #                                 / inter_poly.area
    #                                 > 0.01
    #                             ):
    #                                 overlaps = overlaps + 1
    #                                 submergers.append(inter_poly)
    #                                 iterated_cells.add(inter_poly.uid)
    #                     # catch block: empty list -> some cells are touching, but not overlapping strongly enough
    #                     if len(submergers) == 0:
    #                         merged_idx.append(query_poly.uid)
    #                     else:  # merging strategy: take the biggest cell, other merging strategies needs to get implemented
    #                         selected_poly_index = np.argmax(
    #                             np.array([p.area for p in submergers])
    #                         )
    #                         selected_poly_uid = submergers[selected_poly_index].uid
    #                         merged_idx.append(selected_poly_uid)
    #                 else:
    #                     # no intersection, just add
    #                     merged_idx.append(query_poly.uid)
    #                 iterated_cells.add(query_poly.uid)

    #         if overlaps == 0:

    #             break
    #         elif iteration == 20:
    #             ''
    #         merged_cells = cleaned_edge_cells.loc[
    #             cleaned_edge_cells.index.isin(merged_idx)
    #         ].sort_index()

    #     return merged_cells.sort_index()

    def _remove_overlap(self, cleaned_edge_cells: pd.DataFrame) -> pd.DataFrame:
        merged_cells = cleaned_edge_cells.copy()

        for iteration in range(20):
            if self.logger is not None:
                self.logger.info(f"Overlap removal iteration {iteration}")
            else:
                print(f"\rOverlap removal iteration {iteration}", end="")
            poly_list = []
            uid_list = []

            # Build polygon and uid lists
            for idx, cell_info in merged_cells.iterrows():
                contour = np.asarray(cell_info["contour"])

                if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 4:
                    continue  # Skip invalid or too small contours

                try:
                    poly = Polygon(contour)
                except Exception:
                    continue

                if not poly.is_valid:
                    poly = poly.buffer(0)
                    if isinstance(poly, MultiPolygon):
                        poly = max(poly.geoms, key=lambda p: p.area)
                    if not isinstance(poly, Polygon) or poly.is_empty:
                        continue

                poly_list.append(poly)
                uid_list.append(idx)

            if not poly_list:
                break  # No valid polygons, stop

            tree = strtree.STRtree(poly_list)
            uid_by_geom = dict(zip(tree.geometries, uid_list))  # Build uid mapping via .geometries

            merged_idx = deque()
            iterated_uids = set()
            overlaps = 0

            for query_geom in tree.geometries:
                query_uid = uid_by_geom[query_geom]

                if query_uid in iterated_uids:
                    continue

                intersected_geoms = tree.query(query_geom)

                submergers = []
                for other_geom in intersected_geoms:
                    other_uid = uid_by_geom.get(other_geom)
                    if other_uid is None or other_uid == query_uid or other_uid in iterated_uids:
                        continue

                    inter_area = query_geom.intersection(other_geom).area
                    if inter_area / query_geom.area > 0.01 or inter_area / other_geom.area > 0.01:
                        overlaps += 1
                        submergers.append((other_uid, other_geom))
                        iterated_uids.add(other_uid)

                if not submergers:
                    merged_idx.append(query_uid)
                else:
                    # Merge strategy: keep the one with largest area
                    selected_uid = max(submergers, key=lambda x: x[1].area)[0]
                    merged_idx.append(selected_uid)

                iterated_uids.add(query_uid)

            # If no overlaps, stop early
            if overlaps == 0:
                logger.info("No overlaps, stop early.")
                break

            # Generate next round cell table
            merged_cells = cleaned_edge_cells.loc[cleaned_edge_cells.index.isin(merged_idx)].sort_index()
        logger.info("Overlap removal completed.")

        return merged_cells.sort_index()
