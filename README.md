# Water Body Segmentation with ResNet-UNet

This project implements a deep learning pipeline for semantic segmentation of water bodies from multi-band Sentinel-2 satellite imagery and aerial imagery. A custom ResNet-style encoder is integrated into a UNet architecture to leverage both high-level contextual features and fine-grained spatial details. The model is trained on georeferenced TIFF data with ground-truth masks, supports mixed-precision GPU training for efficiency, and produces pixel-level water/non-water predictions suitable for large-scale environmental and hydrological analysis.

All three workflows—Waterbody, NAIP, and Sentinel-2 (dset-s2)—solve the same core problem: pixel-level surface water segmentation using deep learning. In each case, a UNet-style semantic segmentation model is trained to classify every pixel as water or non-water. The pipeline is consistent across datasets: images and masks are loaded, normalized, resized to a fixed resolution, passed through a convolutional encoder–decoder network, optimized with cross-entropy loss, and evaluated using validation loss and qualitative visualizations. What differs between the sections is the source imagery, spectral content, spatial resolution, and assumptions about georeferencing, which directly affect model design choices and the kinds of water features the model can learn.

## 1. Surface Water Mapping (Waterbody)

### Data Download

Download the dataset from either of the following:

- **Kaggle**: https://www.kaggle.com/datasets/giswqs/geospatial-waterbody-dataset

- **Hugging Face**: https://huggingface.co/datasets/giswqs/geospatial/resolve/main/waterbody-dataset.zip

After downloading and unzipping, you should have two folders:

- `images/` → satellite image files  
- `masks/` → binary water masks with water encoded as `255` and background as `0`

---

### What’s Going On

The Waterbody section focuses on surface water mapping using non-georeferenced satellite imagery, typically RGB images collected from diverse sources. These images lack consistent spatial resolution and geographic metadata, so the task is treated purely as an image-based segmentation problem rather than a geospatial one.

- Inputs are standard RGB images with no guaranteed scale or projection.
- The model learns water appearance based on visual texture, color, and shape alone.
- This setup emphasizes generalization across heterogeneous imagery rather than physical reflectance properties.
- The workflow is ideal for demonstrating deep learning concepts without requiring GIS preprocessing or remote-sensing expertise.

This section highlights how far computer vision alone can go when geospatial context is unavailable.

---

### Running the Model

To train the waterbody segmentation model, run the following script:

- **Training script**: `waterbody_model.py`

During training, the script will automatically:

- Save a training vs. validation loss curve to `train_test_curve_waterbody.png`

- Save the best model checkpoint to `best_unet_waterbody.pth`

After training, the script generates qualitative results on validation samples and saves them to:

- **Sample predictions**: `sample_val_preds_waterbody.png`

These figures typically show three side-by-side panels for each example: the input image, ground-truth water mask, and the model’s predicted mask — providing a visual assessment of segmentation quality.

## 2. Surface Water Mapping with Sentinel-2 Imagery

### Data Download

You can download the dataset from the following link:

- **Sentinel-2 Waterbody Dataset**: https://zenodo.org/records/5205674/files/dset-s2.zip?download=1

---

### What’s Going On

The dset-s2 section uses multispectral Sentinel-2 imagery, shifting the focus from spatial detail to spectral information.

- Inputs consist of six spectral bands (Blue, Green, Red, NIR, SWIR1, SWIR2).
- Spatial resolution is coarser (10–20 m), so small water bodies may not be resolved.
- Water detection relies heavily on spectral signatures, especially NIR and SWIR bands where water has strong absorption.
- The model learns physically meaningful patterns tied to surface reflectance rather than visual appearance alone.

This section emphasizes how multispectral data enables robust water discrimination even when spatial detail is limited.

---

### Running the Model

To train the Sentinel-2 surface water segmentation model, run the following file:

- **Training script**: `dset_s2_model.py`

During training, the script will automatically:

- Save a training vs. validation loss curve to `train_test_curve_dset_s2.png`

- Save the best model checkpoint to `best_unet_dset_s2.pth`

After training, the script also runs inference on validation samples and saves qualitative results to:

- **Sample predictions**: `sample_val_preds_dset_s2.png`

These outputs typically visualize the input Sentinel-2 imagery, the ground-truth water mask, and the predicted mask side by side, providing a clear qualitative assessment of model performance.

This completes the Sentinel-2 surface water mapping workflow, from dataset download to model training and evaluation.


This directory layout allows the dataset loader to read corresponding image–mask pairs automatically during training.

## 3. Surface water mapping with aerial imagery
### Data Download

This section uses NAIP (National Agriculture Imagery Program) aerial imagery for surface water mapping. NAIP provides high-resolution (~1 m) orthorectified aerial imagery across the United States, enabling detection of fine-scale surface water features such as small streams, ponds, and urban drainage.

The NAIP surface water dataset used in this module is hosted on Hugging Face and consists of georeferenced raster imagery along with corresponding binary water masks.

Download the data using the following links:

- **Training imagery**: https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip/naip_water_train.tif

- **Training masks**: https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip/naip_water_masks.tif

- **Test imagery**: https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip/naip_water_test.tif

The training raster and mask files are spatially aligned and can be tiled into smaller image–mask pairs for deep learning. The test raster can be used for inference and qualitative evaluation of model predictions.

**Note: This dataset does not have any test labels and therefore no test loss is outputted.**

---

### What’s Going On

The NAIP section uses high-resolution aerial imagery, shifting the emphasis from spectral richness to fine spatial detail.

- Inputs consist of four-band aerial imagery (Red, Green, Blue, and Near-Infrared).
- Spatial resolution is very high (approximately 1 m), allowing small and narrow water features to be clearly resolved.
- Water detection relies on a combination of visual appearance and near-infrared response, which helps distinguish water from vegetation, pavement, and shadows.
- The model learns fine-scale boundaries and local spatial patterns rather than broad spectral signatures.

This section demonstrates how high-resolution aerial imagery enables accurate detection of small, complex surface water features that are not visible in coarser satellite data.

---

### Running the Model

To train the aerial imagery surface water segmentation model, run the following script:

- **Training script**: `naip_model.py`

During training, the script will automatically:

- Save a training vs. validation loss curve to `train_curve_naip.png`

- Save the best model checkpoint to `best_unet_naip.pth`

After training, the script also runs inference on validation samples and saves qualitative results to:

- **Sample predictions**: `sample_val_preds_naip.png`

These saved figures typically include the original aerial image, the ground-truth mask, and the model’s predicted water mask side by side, providing an intuitive visual measure of segmentation performance.

This completes the aerial imagery surface water mapping workflow, from dataset acquisition through training and evaluation.