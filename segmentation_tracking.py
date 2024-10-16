import os
import cv2
import numpy as np
import torch
import logging
import yaml
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import exposure, morphology
from skimage.exposure import match_histograms
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from matplotlib import image as mpimg
from IPython.display import clear_output
from scipy.ndimage import label as nd_label
import pandas as pd
import shutil
import math
import re

# Set logging level
logging.getLogger('PIL').setLevel(logging.WARNING)

# Input parameters
IMSIZE = 768
IMAGE_FORMAT = 'png'
IMG_DPI = 1200

# Data sources
SOURCE3 = r'..\setA1_tiff'
SOURCE4 = r'..\setB1_tiff'
REFERENCE_SOURCE3 = r'..\references\C1-Damage3pct_MMStack_Default.ome-0040001.tif'
REFERENCE_SOURCE4 = r'C:..\photmtericscancelled_MMStack_Default'

# Setup detectron2 logger
setup_logger()

# Configure detectron2
cfg = get_cfg()
cfg.OUTPUT_DIR = r"..\output_directory"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = r'..\model_final.pth'
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 4000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.TEST.DETECTIONS_PER_IMAGE = 300
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Initialize predictor
predictor = DefaultPredictor(cfg)

def load_and_preprocess_dataset(source, reference_source, in_range):
    dataset_names = os.listdir(source)
    sorted_list_dataset = sorted(dataset_names)
    reference = cv2.resize(mpimg.imread(reference_source), (IMSIZE, IMSIZE))
    reference = exposure.rescale_intensity(reference, in_range=in_range)

    dataset = []
    for idx in tqdm(sorted_list_dataset):
        filepath = os.path.join(source, idx)
        if filepath.endswith('.tif'):
            file = mpimg.imread(filepath)
            file = cv2.resize(file, (IMSIZE, IMSIZE))
            file = match_histograms(file, reference)
            dataset.append(file)
    dataset = np.array(dataset)
    dataset = ((dataset - dataset.min()) / dataset.max()) * 255
    return np.uint8(dataset)

# Load datasets
Dataset = load_and_preprocess_dataset(SOURCE3, REFERENCE_SOURCE3, (1023, 4055))
DatasetC = load_and_preprocess_dataset(SOURCE4, REFERENCE_SOURCE4, (600, 52800))

print('Training set shape:', Dataset.shape)
print('Training set max:', Dataset.max())

# Configure and initialize the second predictor
cfg2 = get_cfg()
cfg2.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg2.MODEL.WEIGHTS = r'..\model_final.pth'
cfg2.DATASETS.TRAIN = ("my_dataset_train",)
cfg2.DATASETS.TEST = ()
cfg2.DATALOADER.NUM_WORKERS = 4
cfg2.SOLVER.IMS_PER_BATCH = 2
cfg2.SOLVER.BASE_LR = 0.00025
cfg2.SOLVER.MAX_ITER = 1000
cfg2.SOLVER.STEPS = []
cfg2.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg2.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg2.TEST.DETECTIONS_PER_IMAGE = 300
cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor2 = DefaultPredictor(cfg2)

# Save configuration to a YAML file
config_yaml_path = r"..\config.yaml"
with open(config_yaml_path, 'w') as file:
    yaml.dump(cfg, file)

# Yeast segmentation
NY = DatasetC.shape[0]
D1 = []
t = time.time()

for i in range(NY):
    new_im = np.stack((DatasetC[i],) * 3, axis=-1)
    outputs = predictor2(new_im)
    category = outputs["instances"].pred_classes
    mask = outputs["instances"].pred_masks.to("cpu").numpy().astype(bool)
    Z = np.zeros((mask.shape[1], mask.shape[2]), dtype="uint8")
    for idx in range(mask.shape[0]):
        if mask[idx].sum() < 2000:
            msk = (np.uint8(category[idx].cpu()) + 1) * mask[idx]
            Z = Z * (~(msk > 0)) + msk
    D1.append(Z)
    clear_output(wait=True)
    print('Frame', i + 1, '/', NY)

elapsed = time.time() - t
print('Completed')
print('Time required for processing all data:', elapsed)
print('Time required for processing single frame:', round(elapsed / NY, 1))

D1 = np.uint8(D1)
D1 = np.array(D1)

# Display results
idx = 1
plt.figure(figsize=(24, 12))
plt.subplot(1, 2, 1)
plt.imshow(DatasetC[idx], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(D1[idx], cmap='gray')
plt.show()

# Amoeba Segmentation using Detectron2
Ndata = min(Dataset.shape[0], DatasetC.shape[0])
D2 = []
kernelD2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

t = time.time()

for i in range(Ndata):
    new_im = np.stack((Dataset[i],) * 3, axis=-1)
    outputs = predictor(new_im)
    category = outputs["instances"].pred_classes
    mask = outputs["instances"].pred_masks.to("cpu").numpy().astype(bool)
    Z = np.zeros((mask.shape[1], mask.shape[2]), dtype="uint8")
    
    for idx in range(mask.shape[0]):
        msk = (np.uint8(category[idx].cpu()) + 1) * mask[idx]
        msk2 = cv2.erode(msk.astype('float32'), kernelD2, iterations=1)
        Z = Z * (~(msk2 > 0)) + msk2

    D2.append(Z)

    clear_output(wait=True)
    print('Frame', i + 1, '/', Ndata)

elapsed = time.time() - t
print('Completed')
print('Time required for processing all data:', elapsed)
print('Time required for processing single frame:', round(elapsed / Ndata, 1))

D2 = np.uint8(D2)
D2 = np.array(D2)

# Define datasets with Amoebas and Yeast
print('Classes:', np.unique(D2))
A = D2 == 1
Y = D2 == 2

# Clean dataset with amoebas
for i in range(A.shape[0]):
    L, N = nd_label(A[i] > 0)
    for j in range(1, N):
        if (L == j).sum() < 50:
            A[i] = A[i] * (~(L == j))

# Definition of the risk action
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))

risk = np.zeros((D2.shape[0], D2.shape[1], D2.shape[2], 3), dtype='uint8')
for idx in range(D2.shape[0]):
    risk[idx, :, :, 0] = 255 * (A[idx] + cv2.dilate(Y[idx].astype('uint8'), kernel, iterations=1) > 1)

idx = 2
res = 128 * np.stack((D2[idx],) * 3, axis=-1)
res[:, :, 0] = res[:, :, 0] * (~(risk[idx, :, :, 0] > 0)) + risk[idx, :, :, 0]
res[:, :, 1] = res[:, :, 1] * (~(risk[idx, :, :, 0] > 0))
res[:, :, 2] = res[:, :, 2] * (~(risk[idx, :, :, 0] > 0))

res = np.uint8(res)

D13 = np.stack((D1,) * 3, axis=-1, dtype='uint8')
D23 = np.stack((D2,) * 3, axis=-1, dtype='uint8')
D3 = np.stack((Dataset,) * 3, axis=-1, dtype='uint8')
D3C = np.stack((DatasetC,) * 3, axis=-1, dtype='uint8')
A3 = np.stack((A,) * 3, axis=-1, dtype='uint8')


#################################################
# Initialize parameters
idx = 0  # Select starting slice index
labels, nlabels = nd_label(A[idx] > 0.1)

# Create labeled regions of interest
label_arrays = [np.where(labels == label_num, 1, 0) for label_num in range(1, nlabels + 1)]
print(f'{nlabels} separate objects detected.')

# Plot labels with indexes
reg4 = D2[idx].copy()
cx, cy = np.zeros((1, D2.shape[0])), np.zeros((1, D2.shape[0]))
for idxx, binary in enumerate(label_arrays):
    binary = np.asarray(binary, dtype="uint8")
    contours, _ = cv2.findContours(image=binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    if len(contours[0]) > 5:
        ellipse = cv2.fitEllipse(contours[0])
        cx0, cy0 = ellipse[0]
        reg4 = cv2.drawContours(np.uint8(reg4), contours[0], -1, (120, 0, 0), 3)
        reg5 = cv2.putText(reg4, str(idxx + 1), (int(cx0), int(cy0)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 2)

# Visualization
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(Dataset[idx], cmap='gray')
plt.title('Phase contrast ' + str(idx + 1))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reg5, cmap='gray')
plt.title('Numbered eroded masks for Dicty 6')
plt.axis("off")

# Prepare directories and data structures
amoebas_idx = [5, 16, 14, 19, 7, 1, 15, 6, 21, 13]
dirpath = 'ROIs_selected'
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)
os.mkdir(dirpath)

Z0 = np.zeros(risk.shape[1:], dtype='uint8')
ZT, ZB = np.zeros(risk.shape, dtype='uint8'), np.zeros(risk.shape, dtype='uint8')
M0, N0 = D2.shape[1], D2.shape[2]

# Process each ROI
for roi_idx, amoeba in enumerate(amoebas_idx, start=1):
    track = np.zeros((risk.shape[1], risk.shape[2], 3), dtype='uint8')
    binary = np.asarray(label_arrays[amoeba - 1], dtype="float32")
    binary = np.uint8(binary > 0)
    contours, _ = cv2.findContours(image=binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    
    M = cv2.moments(cnt)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    m, n = cy, cx
    cx0, cy0 = cx, cy

    # Determine half-side dimensions for ROI
    hside = np.uint8(round(150 / 2, 0))
    hsideN, hsideM = hside, hside
    if (N0 - n) < hside and m >= hside:
        hsideN = N0 - n - 1
    if (M0 - m) < hside and n >= hside:
        hsideM = M0 - m - 1
    if (M0 - m) < hside and (N0 - n) < hside:
        hsideM, hsideN = M0 - m - 1, N0 - n - 1
    if m < hside and n >= hside:
        hsideM = m
    if n < hside and m >= hside:
        hsideN = n

    Z0[frame_idx, m - hsideM:m + hsideM, n - hsideN:n + hsideN] = 1
    Z1 = np.zeros((risk.shape[1], risk.shape[2]), dtype='uint8')
    Z1[m - hsideM:m + hsideM, n - hsideN:n + hsideN] = 1

    labels2, nlabels2 = nd_label(A[frame_idx] * (Z1 > 0))
    for ind in range(1, nlabels2 + 1):
        if np.uint8((label_arrays[amoeba - 1]) * (labels2 == ind) * (Z1 > 0)).sum() > 0:
            Z0[frame_idx] += np.uint8((labels2 == ind))
            ZT[frame_idx] = cv2.putText(ZT[frame_idx], 'ROI_' + str(roi_idx), (n, m - hsideM), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    os.mkdir(os.path.join(dirpath, 'ROI' + str(roi_idx)))

    for k in range(1, 591):
        labels3, nlabels3 = nd_label((A[frame_idx + k] * Z0[frame_idx + k - 1]) > 0)
        maxA, idxx = 10, 0
        for kk in range(1, nlabels3 + 1):
            if np.uint8((Z0[frame_idx + k - 1] > 1) * (labels3 == kk) * (Z1 > 0)).sum() > 0:
                if np.uint8((Z0[frame_idx + k - 1] > 1) * (labels3 == kk)).sum() > 0:
                    if np.uint8((labels3 == kk)).sum() > maxA:
                        maxA = np.uint8((labels3 == kk)).sum()
                        idxx = np.uint8(kk)

        binary = np.asarray(labels3 == idxx, dtype="float32")
        binary = np.uint8(binary)
        contours, _ = cv2.findContours(image=binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        if contours:
            M = cv2.moments(contours[0])
            if M['m00'] > 0:
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                m, n = cy, cx

                # Adjust half-side dimensions for current ROI
                if (N0 - n) < hside and m >= hside:
                    hsideN = N0 - n - 1
                if (M0 - m) < hside and n >= hside:
                    hsideM = M0 - m - 1
                if (M0 - m) < hside and (N0 - n) < hside:
                    hsideM, hsideN = M0 - m - 1, N0 - n - 1
                if m < hside and n >= hside:
                    hsideM = m
                if n < hside and m >= hside:
                    hsideN = n

                Z0[frame_idx + k, m - hsideM:m + hsideM, n - hsideN:n + hsideN] = 1
                Z0[frame_idx + k] += np.uint8((labels3 == idxx))
                ZT[frame_idx + k] = cv2.putText(ZT[frame_idx + k], 'ROI_' + str(roi_idx), (n, m - hsideM), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                ZB[frame_idx + k] = cv2.rectangle(ZB[frame_idx + k], (n - hsideN, m - hsideM), (n + hsideN, m + hsideM), (0, 255, 0), 2)
                Z1.fill(0)
                Z1[m - hsideM:m + hsideM, n - hsideN:n + hsideN] = 1

                # Feature extraction
                ellipse = cv2.fitEllipse(contours[0])
                cx1, cy1 = round(ellipse[0][0]), round(ellipse[0][1])
                (xc, yc), (d1, d2), angle = ellipse
                area = cv2.contourArea(contours[0])
                perimeter = cv2.arcLength(contours[0], True)
                # Calculate features
                circularity = 4 * math.pi * area / (perimeter ** 2)
                step = math.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)
                rmajor = max(d1, d2) / 2
                angle1 = angle - 90 if angle > 90 else angle + 90
                hull = cv2.convexHull(contours[0])
                convex_hull_area = cv2.contourArea(hull)
                roundness = 4 * area / (math.pi * max(d1, d2) ** 2)
                solidity = area / convex_hull_area
                aspect_ratio = max(d1, d2) / min(d1, d2) if min(d1, d2) > 0 else 0

                # Update previous center coordinates
                cx0, cy0 = cx1, cy1

                # Calculate tracking overlap
                tracking_overlap = (cv2.subtract(np.uint8((Z1 > 0) * (Z0[frame_idx + k] > 1)),
                                                 np.uint8((Z1 > 0) * (Z0[frame_idx + k - 1] > 1)))).sum()

                # Prepare data for saving
                row_dict = {
                    "Circularity": round(circularity, 1),
                    "Trecking_step": round(step, 1),
                    "Area": round(area, 2),
                    "Tracking_overlap": tracking_overlap,
                    "Roundness": round(roundness, 1),
                    "Solidity": round(solidity, 1),
                    "Aspect_ratio": round(aspect_ratio, 1),
                    "Ellipse_angle": angle,
                    "Ellipse_max_diameter": max(d1, d2)
                }
                dict_list.append(row_dict)

                # Draw tracking path
                track = cv2.line(track, (px0, py0), (px1, py1), (255, 0, 0), 2)

                # Save images in folders
                filename = os.path.join(dirpath, f'ROI{roi_idx}', f'image_{k}.jpg')
                res = 127 * np.stack((Z0[k + frame_idx],) * 3, axis=-1)
                res[:, :, 0] = res[:, :, 0] * (~(track[:, :, 0] > 0))
                res[:, :, 1] = res[:, :, 1] * (~(track[:, :, 0] > 0))
                res[:, :, 2] = res[:, :, 2] * (~(track[:, :, 0] > 0))
                res += track

                # Save composite image
                res1 = cv2.ellipse(D3[frame_idx + k], ellipse, (255, 0, 0), 1)
                img = np.hstack((res1[m - hsideM:m + hsideM, n - hsideN:n + hsideN],
                                 res[m - hsideM:m + hsideM, n - hsideN:n + hsideN],
                                 255 * D13[frame_idx + k, m - hsideM:m + hsideM, n - hsideN:n + hsideN]))
                cv2.imwrite(filename, cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR))

    # Save features to Excel
    df = pd.DataFrame.from_dict(dict_list)
    sheet_name = 'ROI_' + str(roi_idx)
    df.to_excel(writer, sheet_name=sheet_name, index=False)

    clear_output(wait=True)
    print('Processed ROI', ii + 1)

# Finalize Excel writing
writer.close()
print('Completed')
print('Number of processed ROIs:', roi_idx)

# Create GIFs from processed images
source = r'C:\Users\ph1oc\PycharmProjects\DictyProject\ROIs_selected\ROI3'
dataset_names = os.listdir(source)
re_pattern = re.compile(r'.+?(\d+)\.([a-zA-Z0-9]+)')
sorted_list_dataset = sorted(dataset_names, key=lambda x: int(re_pattern.match(x).groups()[0]))

# Load and display the first image from the dataset
Dataset = [mpimg.imread(os.path.join(source, idx)) for idx in tqdm(sorted_list_dataset)]
Dataset = np.array(Dataset)
plt.imshow(Dataset[0])
print('Dataset shape:', Dataset.shape)

### save GIF ###############################
from moviepy.editor import ImageSequenceClip
print('making GIF...')
clip = ImageSequenceClip(list(Dataset), fps=20)# 2 seconds
clip.write_gif(r"..\Gifs\ROI1.gif")
print('Done')
