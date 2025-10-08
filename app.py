import os
import io
import numpy as np
from PIL import Image
from scipy import stats
import pydicom
import pandas as pd
import magic
import streamlit as st
from ultralytics import YOLO
from pydicom.errors import InvalidDicomError

from src.constants import PATH_TO_BEST_MODEL, DISC_LABELS
from src.tools import (
    group_uploaded_files_by_series,
    filter_dicom_series,
    load_dicom_series_from_files,
    get_yolo_pred,
)

DEVICE = "cpu"
st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache_resource
def cached_model(model_path):
    model = YOLO(model_path)
    model.to(DEVICE)
    return model


def plot_result(pred):
    result_bgr = pred.plot()
    result_rgb = Image.fromarray(result_bgr[..., ::-1])  # RGB-order PIL image
    orig_img = pred.orig_img
    tab1, tab2 = st.tabs(["Prediction", "Image"])
    tab1.image(result_rgb)
    tab2.image(orig_img)


def _make_uploadedfile_like(paths):
    files_like = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                data = f.read()
        except Exception:
            continue
        bio = io.BytesIO(data)
        bio.name = os.path.basename(p)
        try:
            bio.type = magic.from_buffer(data, mime=True) or "application/octet-stream"
        except Exception:
            bio.type = "application/octet-stream"
        files_like.append(bio)
    return files_like


def process_uploaded_files(uploaded_files_like, *, model):
    results_dict = {}

    series_dict, _ = group_uploaded_files_by_series(uploaded_files_like)
    filtered_series = filter_dicom_series(series_dict)

    if len(filtered_series) > 0:
        for series_uid, series_files in filtered_series.items():
            volume = load_dicom_series_from_files(series_files, series_uid)
            if volume is None:
                continue

            num_slices = volume.shape[0]
            if num_slices < 6:
                continue

            mid = num_slices // 2
            outputs, confidences, sum_whs = [], [], []
            preds = {}

            for s in [-2, -1, 0, 1, 2]:
                slice_num = mid + s
                if slice_num < 0 or slice_num >= num_slices:
                    continue
                img = volume[slice_num]
                pred, sorted_classes, sorted_class_conf = get_yolo_pred(img, model)
                preds[slice_num] = pred

                if len(sorted_classes) == 5:
                    outputs.append(sorted_classes)
                    confidences.append(sorted_class_conf)
                    xywh = pred.boxes.xywhn.detach().cpu().numpy()
                    sum_wh = np.sum([100 * wh[0] * wh[1] for wh in xywh[:, 2:]]) if xywh.size else 0.0
                    sum_whs.append(sum_wh)

            if len(outputs) != 0:
                output = stats.mode(outputs, axis=0)[0][0].astype(int)
                total_metrics = np.sum(confidences) + np.sum(sum_whs)
                results_dict[total_metrics] = (series_uid, num_slices, preds, output)

    if len(results_dict) != 0:
        best_series, num_slices, best_preds, best_output = results_dict[max(results_dict.keys())]
        best_output = [i + 1 for i in best_output]
        avg_df = pd.DataFrame({"Disc": DISC_LABELS, "Pfirrmann Grade": best_output})
        avg_df.set_index("Disc", inplace=True)
        st.dataframe(avg_df)

        slices = [f"Slice {i + 1}" for i in best_preds.keys()]
        for tab, key in zip(st.tabs(slices), best_preds.keys()):
            with tab:
                plot_result(best_preds[key])

        st.write(f"Series Instance UID: {best_series}")
        st.write(f"Number of slices: {num_slices}")
    else:
        st.error(
            "Error: Unable to identify appropriate lumbar spine MRI slices for disc analysis. "
            "Please verify the MRI scan includes the lumbar region."
        )


# --------------------- UI ---------------------

model = cached_model(PATH_TO_BEST_MODEL)

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

st.image("logo/logo.png")
uploaded_files = st.file_uploader(
    label="Select the MRI folder",
    accept_multiple_files=True,
    label_visibility="hidden",
    key=st.session_state["file_uploader_key"],
)

if st.button("Reset"):
    st.session_state["file_uploader_key"] += 1
    st.experimental_rerun()

if len(uploaded_files) != 0:
    process_uploaded_files(uploaded_files, model=model)
else:
    if st.button("Example (Image)"):
        img = Image.open("examples/166_LKN_28.jpg").convert("RGB")
        pred = model([img], verbose=None)[0]
        plot_result(pred)

    if st.button("Example (DICOM folder)"):
        all_paths = []
        for root, _, files in os.walk("examples/197_MAN"):
            for fn in files:
                all_paths.append(os.path.join(root, fn))

        uploaded_like = _make_uploadedfile_like(all_paths)
        process_uploaded_files(uploaded_like, model=model)
