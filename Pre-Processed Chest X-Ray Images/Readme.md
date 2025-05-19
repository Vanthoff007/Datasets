# IN-CXR (pre-processed)

The dataset is derived from the [**IN-CXR: An Open Dataset of Chest Radiograph**](https://nirt.res.in/html/xray.html) and has been optimized for improved usability. The original DICOM images were converted to PNG format and underwent a pre-processing pipeline designed to enhance image quality for model training.

## Chest X-ray Image Preprocessing Pipeline

To enhance the quality and diagnostic relevance of chest X-ray images before feeding them into a deep learning model, we employed the following preprocessing steps:

### 1. Grayscale Loading & Resizing
All input images are loaded in grayscale mode and resized to a fixed resolution of **224×224 pixels** to standardize input dimensions across the dataset.

### 2. Adaptive Masking for Lung Segmentation
To suppress non-lung regions (e.g., diaphragm, bright artifacts), we apply a basic adaptive masking technique:
- Compute the minimum and maximum pixel intensities in the image.
- Apply binary thresholding using a threshold defined as:  
  **threshold = min + 0.9 × (max − min)**
- Perform **morphological closing** using an elliptical kernel to fill small gaps and smooth the mask.
- Invert the mask and apply it to the image, preserving only the relevant lung fields.

### 3. Contrast Enhancement (CLAHE)
To improve local contrast without amplifying noise, we apply **Contrast Limited Adaptive Histogram Equalization (CLAHE)**. CLAHE operates on small regions (tiles) and is especially effective for enhancing subtle patterns in lung tissue.

### 4. Edge-Preserving Denoising
To suppress random noise while preserving important structural edges, we use **Non-Local Means Denoising** (`cv2.fastNlMeansDenoising`). This method averages similar patches across the image, making it well-suited for medical images.

### 5. Saving as 8-bit PNG
The processed images are saved in `.png` format as 8-bit grayscale images, preserving compatibility with most computer vision pipelines while ensuring minimal loss of detail.

---

This preprocessing pipeline helps focus the model on diagnostically relevant regions, improves contrast for subtle abnormalities, and reduces noise, thereby facilitating better training and generalization for downstream classification tasks such as tuberculosis detection.


The processed images are categorized into two directories: **NORMAL** and **ABNORMAL** (tuberculosis positive) chest X-Rays, facilitating easy loading and processing through the ImageFolder method from the torchvision library's datasets module.
