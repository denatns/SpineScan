import re
import cv2
import tempfile
import shutil
import SimpleITK as sitk
import numpy as np
import torch
import pydicom
from PIL import Image
from pydicom.errors import InvalidDicomError

from src.constants import SAGITTAL, BANNED_TAGS



def is_t2_series(dicom):
    t2_flag = False
    if hasattr(dicom, "SeriesDescription") and ('T2' in str(dicom.SeriesDescription).upper()):
        series_description = str(dicom.SeriesDescription).upper()
        t2_flag = all(t not in series_description for t in BANNED_TAGS)
    elif hasattr(dicom, "ProtocolName") and ('T2' in str(dicom.ProtocolName).upper()):
        t2_flag = True
    else:
        for element in dicom:
            if hasattr(element, "VR") and element.VR == "SQ":
                for sub_element in element.value:
                    for e in sub_element:
                        if e.keyword == "AcquisitionContrast" and e.value == "T2":
                            t2_flag = True
                            break
            if t2_flag:
                break
    return t2_flag


def group_uploaded_files_by_series(uploaded_files):
    series_dict = {}
    non_dicom_files = []

    for uploaded_file in uploaded_files:
        try:
            dicom = pydicom.dcmread(uploaded_file, stop_before_pixels=True)
            series_uid = dicom.SeriesInstanceUID

            if series_uid not in series_dict:
                series_dict[series_uid] = []

            uploaded_file.seek(0)
            series_dict[series_uid].append(uploaded_file)
        except InvalidDicomError:
            non_dicom_files.append(uploaded_file)
        except Exception as e:
            continue

    return series_dict, non_dicom_files


def filter_dicom_series(series_files):
    filtered_series = {}
    for series_uid, files in series_files.items():
        try:
            dicom = pydicom.dcmread(files[0])

            if not hasattr(dicom, 'ImageOrientationPatient'):
                continue

            if not all(np.around(dicom.ImageOrientationPatient) == SAGITTAL):
                continue

            if not is_t2_series(dicom):
                continue

            filtered_series[series_uid] = files
        except Exception as e:
            continue

    return filtered_series


def load_dicom_series_from_files(series_files, series_id):
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_paths = []
            for uploaded_file in series_files:
                file_path = f"{tmpdirname}/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(uploaded_file, f)
                file_paths.append(file_path)

            paths = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(tmpdirname, series_id, recursive=True)
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(paths)
            reader.LoadPrivateTagsOn()
            reader.MetaDataDictionaryArrayUpdateOn()
            image = reader.Execute()
            return sitk.GetArrayFromImage(image)
    except Exception as e:
        return None


def percentile_norm(img):
    img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
    max_v = np.max(img)
    denominator = max_v - np.min(img)
    if denominator > 0:
        return (img - np.min(img)) / denominator
    elif max_v > 0:
        return img / max_v
    else:   
        return img


def scale_to_min_height(img, min_height=640):
    h, w, c = img.shape
    if h < min_height:
        ratio = min_height / h
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        img = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img


def get_yolo_pred(img, model):
    img = percentile_norm(img)
    img = (img * 255).astype(np.uint8)
    img_rgb = Image.fromarray(img)
    img_rgb = np.array(img_rgb.convert("RGB"))
    img_scaled = scale_to_min_height(img_rgb)
    preds = model.predict(img_scaled, max_det=5, augment=True, agnostic_nms=True, verbose=False)
    pred = preds[0]
    bbox_coordinates = pred.boxes.xyxy.cpu()
    class_labels = pred.boxes.cls.cpu()
    class_confidences = pred.boxes.conf.cpu()
    if bbox_coordinates.numel() == 0:
        return pred, [], []
    y_coordinates = bbox_coordinates[:, 1]
    sorted_indices = torch.argsort(y_coordinates, descending=False)
    sorted_classes = class_labels[sorted_indices].tolist()
    sorted_class_conf = class_confidences[sorted_indices].tolist()
    
    return pred, sorted_classes, sorted_class_conf
