from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import tangram as tg


def nucleixspot(
    tissue_positions,
    spot_diameter,
    measurements,
    position_coords,
    measurements_coords=["Centroid X µm", "Centroid Y µm"],
    qs=[0.5],
    selected_features=["Nucleus: Area"],
    pixels=True,
    resolution=1,
):
    qs = [qs] if isinstance(qs, (float, int)) else qs
    spot_radius = spot_diameter // 2
    aggregated_measurements = pd.DataFrame()
    for i, row in tissue_positions.iterrows():
        x, y = row[position_coords[0]], row[position_coords[1]]
        mask = (measurements[measurements_coords[0]] - x) ** 2 + (
            measurements[measurements_coords[1]] - y
        ) ** 2 <= spot_radius**2
        spot_measurements = measurements.loc[mask, selected_features]
        if not spot_measurements.empty:
            spot_aggregated = pd.DataFrame(index=[i])
            spot_aggregated["n_detections"] = int(len(spot_measurements))
            if pixels:
                spot_aggregated["Nucleus: Area_percentage"] = (
                    spot_measurements["Nucleus: Area"].sum() / (math.pi * (spot_radius**2))
                ) * 100
            else:
                spot_aggregated["Nucleus: Area_percentage"] = (
                    (spot_measurements["Nucleus: Area"].sum() / resolution) / (math.pi * (spot_radius**2))
                ) * 100
            for feature in spot_measurements.columns:
                for q in qs:
                    quantile_val = spot_measurements[feature].quantile(q)
                    new_col_name = f"{feature}_{int(q*100)}"
                    spot_aggregated[new_col_name] = [quantile_val]
            aggregated_measurements = pd.concat([aggregated_measurements, spot_aggregated])
    return aggregated_measurements


def nucleixspot_adata(adata, **kwargs):
    aggregated_measurements = nucleixspot(**kwargs)
    adata.obs = pd.concat([adata.obs, aggregated_measurements], axis=1)
    adata.obs.iloc[:, -aggregated_measurements.shape[1] :] = adata.obs.iloc[
        :, -aggregated_measurements.shape[1] :
    ].fillna(0)
    return adata


print("\n Loading spatial data...")
ad_sp = sq.datasets.visium_hne_adata()
print(ad_sp)

print("\n Cropping region...")
scale = ad_sp.uns["spatial"]["V1_Adult_Mouse_Brain"]["scalefactors"]["tissue_hires_scalef"]
img = sq.im.ImageContainer(ad_sp.uns["spatial"]["V1_Adult_Mouse_Brain"]["images"]["hires"], layer="image", scale=scale)

measurements = pd.read_csv("measurements.csv")

measurements["Centroid X µm"] *= 1.3125
measurements["Centroid Y µm"] *= 1.3125

xx = (measurements["Centroid X µm"].min()) / (img.shape[1] / scale)
yy = (measurements["Centroid Y µm"].min()) / (img.shape[0] / scale)
size_x = measurements["Centroid X µm"].max() - measurements["Centroid X µm"].min()
size_y = measurements["Centroid Y µm"].max() - measurements["Centroid Y µm"].min()

img_crop = img.crop_corner(x=xx, y=yy, size=(int(size_y * scale), int(size_x * scale)))

ad_sp_crop = img_crop.subset(ad_sp, copy=True)

print("\n Loading single-cell data...")
ad_sc = sq.datasets.sc_mouse_cortex()

print("\n Preprocessing...")
ad_sc = ad_sc[ad_sc.obs.groupby("cell_subclass").filter(lambda x: len(x) > 1).index]
sc.tl.rank_genes_groups(ad_sc, groupby="cell_subclass", use_raw=False)
markers_df = pd.DataFrame(ad_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
markers = list(np.unique(markers_df.melt().value.values))
print(len(markers))
tg.pp_adatas(ad_sc, ad_sp_crop, genes=markers)

print("\n Loading morphology...")
tissue_positions = pd.DataFrame(ad_sp_crop.obsm["spatial"], columns=["img_x", "img_y"], index=ad_sp_crop.obs.index)
spot_diameter_fullres = ad_sp_crop.uns["spatial"]["V1_Adult_Mouse_Brain"]["scalefactors"]["spot_diameter_fullres"]
ad_sp_crop = nucleixspot_adata(
    ad_sp_crop,
    tissue_positions=tissue_positions,
    spot_diameter=spot_diameter_fullres,
    measurements=measurements,
    position_coords=["img_x", "img_y"],
    measurements_coords=["Centroid X µm", "Centroid Y µm"],
    qs=[0.05, 0.5, 0.95],
    selected_features=["Nucleus: Area"],
    pixels=False,
    resolution=1,
)

print("\n Deconvolution via alignment...")
ad_map = tg.map_cells_to_space(
    ad_sc,
    ad_sp_crop,
    mode="constrained",
    target_count=ad_sp_crop.obs.n_detections.sum(),
    density_prior=np.array(ad_sp_crop.obs.n_detections) / ad_sp_crop.obs.n_detections.sum(),
    num_epochs=10000,
    device="cpu",
)

tg.project_cell_annotations(ad_map, ad_sp_crop, annotation="cell_subclass")
ad_sp_crop.obs = pd.concat([ad_sp_crop.obs, ad_sp_crop.obsm["tangram_ct_pred"]], axis=1)

print("\n Adding morphology to adata...")
spot_r = spot_diameter_fullres // 2
measurements = measurements[["Centroid X µm", "Centroid Y µm"]]
ad_sp_crop.uns["tangram_cell_segmentation"] = pd.DataFrame(columns=["spot_idx", "x", "y", "centroids"])

for i, row in tissue_positions.iterrows():
    spot_x = row["img_x"]
    spot_y = row["img_y"]
    dist = np.sqrt((measurements["Centroid X µm"] - spot_x) ** 2 + (measurements["Centroid Y µm"] - spot_y) ** 2)
    within_radius = measurements[dist <= spot_r]
    spot_idx = row.name
    for j, (jj, within_row) in enumerate(within_radius.iterrows()):
        ad_sp_crop.uns["tangram_cell_segmentation"] = pd.concat(
            [
                ad_sp_crop.uns["tangram_cell_segmentation"],
                pd.DataFrame.from_dict(
                    {
                        "spot_idx": [spot_idx],
                        "x": [within_row["Centroid X µm"]],
                        "y": [within_row["Centroid Y µm"]],
                        "centroids": [spot_idx + "_" + str(j)],
                    }
                ),
            ],
            ignore_index=True,
        )

xs = ad_sp_crop.obsm["spatial"][:, 1]
ys = ad_sp_crop.obsm["spatial"][:, 0]

tangram_spot_centroids = pd.pivot_table(
    ad_sp_crop.uns["tangram_cell_segmentation"],
    index=["spot_idx"],
    values=["centroids"],
    aggfunc=lambda x: list(map(str, ", ".join(map(str, x)).split(", "))),
)

list_to_array = lambda x: np.array(x, dtype=str)
tangram_spot_centroids["array_centroids"] = tangram_spot_centroids["centroids"].apply(list_to_array)

temp = ad_sp_crop.obs[["in_tissue"]]
temp = temp.join(tangram_spot_centroids)
temp["centroids"].fillna(0, inplace=True)

ad_sp_crop.obsm["image_features"] = pd.DataFrame(
    data=ad_sp_crop.obs.n_detections.values, columns=["segmentation_label"], index=ad_sp_crop.obs.index, dtype="int"
)
ad_sp_crop.obsm["tangram_spot_centroids"] = temp.array_centroids

cell_count = ad_sp_crop.obsm["image_features"]["segmentation_label"]
df_segmentation = ad_sp_crop.uns["tangram_cell_segmentation"]
centroids = ad_sp_crop.obsm["tangram_spot_centroids"]

df_vox_cells = df_vox_cells = pd.DataFrame(
    data={"x": xs, "y": ys, "cell_n": cell_count, "centroids": centroids},
    index=list(ad_sp_crop.obs.index),
)

tg.count_cell_annotations(
    ad_map,
    ad_sc,
    ad_sp_crop,
    annotation="cell_subclass",
)

ad_sp_mod = ad_sp_crop[ad_sp_crop.obs.n_detections != 0]
df_vox_cells = ad_sp_mod.obsm["tangram_ct_count"]
filter_cell_annotation = pd.unique(list(ad_sp_mod.obsm["tangram_ct_pred"].columns))
cell_types = filter_cell_annotation
df = df_vox_cells
df_cum_sums = df[cell_types].cumsum(axis=1)
df_c = df.copy()
for i in df_cum_sums.columns:
    df_c[i] = df_cum_sums[i]

from collections import defaultdict

cell_types_mapped = defaultdict(list)
for i_index, i in enumerate(cell_types):
    for j_index, j in df_c.iterrows():
        start_ind = 0 if i_index == 0 else j[cell_types[i_index - 1]]
        end_ind = j[i]
        cell_types_mapped[i].extend(j["centroids"][start_ind:end_ind].tolist())

adata_segment = tg.deconvolve_cell_annotations(ad_sp_crop[ad_sp_crop.obs.n_detections != 0])

print("\n Saving...")
adata_segment.obs.to_csv("per_cell.csv")
