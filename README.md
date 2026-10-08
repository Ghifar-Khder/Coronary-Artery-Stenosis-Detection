# Coronary Artery Stenosis Detection

[**Open the live app**](https://coronary-artery-stenosis-detection.streamlit.app/)

This project uses a ResNet101V2-based model to locate coronary stenosis in X-ray angiography images. It predicts a bounding box around the stenotic region. The Streamlit app accepts individual images or videos and displays the predicted location.

![Coronary stenosis detector interface](https://raw.githubusercontent.com/Ghifar-Khder/portfolio/main/assets/stenosis.jpeg)

## Using the app

1. Open the live app and choose **Image** or **Video** in the sidebar.
2. Upload an angiography image (`.jpg`, `.jpeg`, `.png`, or `.bmp`) or video (`.mp4`, `.avi`, or `.mov`).
3. For an image, view the bounding box and its coordinates. For a video, wait for processing, then view or download the annotated video.

## Methods

### Dataset and preparation

The dataset contains **8,325 grayscale angiography frames from 100 patients**, with image sizes ranging from 512 × 512 to 1000 × 1000 pixels. The images were collected at the Research Institute for Complex Problems of Cardiovascular Diseases in Kemerovo, Russia, using Siemens Coroscop and GE Healthcare Innova systems.

Frames showing contrast passage through stenotic vessels were retained. The stenotic regions were annotated in LabelBox, with bounding boxes stored in XML files.

Preparation included:

- Matching each image to its XML annotation and separating training and test data.
- Setting aside 10% of the training images for validation.
- Converting XML annotations into CSV files containing image information and bounding box coordinates.
- Resizing images to **512 × 512** and adjusting the box coordinates to match.
- Applying random horizontal flips and brightness changes during training. Box coordinates were updated when an image was flipped.

Image sequences were also grouped and ordered by filename, then converted into videos at 10 frames per second for visual inspection.

### Model and training

The model uses a pretrained **ResNet101V2** backbone without its original classification head. The backbone is followed by a convolutional layer, global average pooling, a dense layer, and a bounding box regression output, in this order:

| Layer | Configuration |
| --- | --- |
| ResNet101V2 backbone | Pretrained feature extractor without its original classification head |
| Convolution | 3 × 3 kernel, 1,024 filters |
| Global average pooling | Converts feature maps into a feature vector |
| Dense layer | 512 neurons |
| Bounding box regression | Four coordinate outputs |

Training used the **Adam** optimizer and **Huber loss** for bounding box regression. Training batches were shuffled and prefetched, while validation data stayed in a fixed order.

### Image and video processing

The deployed app converts input images to RGB, resizes them to 512 × 512, and scales pixel values to the range 0–1. Predicted coordinates are mapped back to the original image size.

Video frames pass through the same prediction process. A consistency filter compares consecutive bounding boxes using intersection over union (IoU). The app draws a box after two successive comparisons exceed an IoU of 0.25. This reduces flickering between frames. The output keeps the source video's frame rate, with a fallback of 10 fps when that rate is unavailable.

## Run locally

Install Python and Git LFS, then clone the repository and download the model:

```bash
git lfs install
git clone https://github.com/Ghifar-Khder/Coronary-Artery-Stenosis-Detection.git
cd Coronary-Artery-Stenosis-Detection
git lfs pull
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The file `stenosis_detector_v18.keras` must contain the downloaded model, rather than only the Git LFS pointer.

## Repository contents

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, image prediction, and video processing |
| `stenosis_detector_v18.keras` | Trained model, stored with Git LFS |
| `requirements.txt` | Python dependencies |
| `.streamlit/` | App configuration |

This repository contains the inference app and trained model. The dataset and the scripts for data preparation and training are not included here.

## Scope

The model predicts one bounding box per frame. It does not measure the percentage of artery narrowing or provide a separate no-stenosis decision. The dataset contains selected frames showing stenotic vessels, so those data alone do not establish how the model performs on normal angiograms.

## Author

[Ghifar Khder — Portfolio](https://ghifar-khder.github.io/portfolio/)
