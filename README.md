# Multiview-Video-Summarization for surveillance

This project implements a pipeline for face detection, recognition, and video summarization using deep learning. The workflow leverages the `facenet-pytorch` library for face detection and embedding, and PyTorch for model training and inference. The pipeline processes video frames, detects and recognizes faces, and generates a summarized output video based on recognized identities.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Pipeline Overview](#pipeline-overview)
- [Training](#training)
- [Inference and Video Summarization](#inference-and-video-summarization)
- [Outputs](#outputs)
- [Notes](#notes)
- [References](#references)

---

## Features

- **Face Detection:** Uses MTCNN for robust face detection in images and video frames.
- **Face Recognition:** Trains an InceptionResnetV1 model (pretrained on VGGFace2) for identity classification.
- **Data Augmentation:** Applies various augmentations to improve model robustness.
- **Video Summarization:** Processes multiple videos, recognizes faces, and creates a summary video with annotated frames.
- **Embeddings Storage:** Saves and loads face embeddings for efficient recognition.

---

## Project Structure

```
.
├── main.ipynb                # Main Jupyter notebook with the full pipeline
├── README.md                 # This file
├── Faces/                    # Raw face image dataset (not included in repo)
├── Faces_cropped/            # Cropped and augmented face images (generated)
├── NamitKush30_emb/          # Saved face embeddings and names (generated)
├── videos2/                  # Input videos for summarization (not included in repo)
├── output2_video.mp4         # Example output summary video (generated)
```

---

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Multiview-Video-Summarization
   ```

2. **Install dependencies:**
   - Python 3.8+
   - PyTorch (with CUDA if GPU available)
   - facenet-pytorch
   - torchvision
   - numpy
   - opencv-python
   - pillow
   - tensorboard

   You can install the main dependencies with:
   ```bash
   pip install torch torchvision facenet-pytorch numpy opencv-python pillow tensorboard
   ```

3. **(Optional) Install Jupyter:**
   ```bash
   pip install notebook
   ```

---

## Data Preparation

- **Faces/**: Place your raw face images in subfolders by class (person name), e.g.:
  ```
  Faces/
    ├── Kush/
    │     ├── img1.jpg
    │     └── ...
    └── Namit/
          ├── img1.jpg
          └── ...
  ```
- **videos2/**: Place your input videos as `video1.mp4`, `video2.mp4`, etc.

---

## Pipeline Overview

1. **Face Cropping:**  
   - Detects faces in the raw dataset and saves cropped faces to `Faces_cropped/`.

2. **Data Augmentation:**  
   - Applies random rotations, flips, color jitter, affine transforms, and blurring to increase dataset diversity.

3. **Model Training:**  
   - Fine-tunes an InceptionResnetV1 model for face classification using the cropped and augmented images.

4. **Embedding Extraction:**  
   - Extracts and saves face embeddings for each identity for fast recognition.

5. **Video Processing & Summarization:**  
   - Processes each video, detects and recognizes faces, annotates frames, and collects them in time order.
   - Generates a summary video (`output2_video.mp4`) with recognized faces and timestamps.

---

## Training

- Training is performed in the notebook using the `facenet_pytorch.training` utilities.
- The model is trained for 30 epochs with Adam optimizer and a learning rate scheduler.
- Training and validation metrics (loss, accuracy, FPS) are printed and can be visualized with TensorBoard.

---

## Inference and Video Summarization

- The pipeline loads the trained model and face embeddings.
- For each frame in the input videos:
  - Detects faces and computes embeddings.
  - Compares embeddings to the database and recognizes known identities.
  - Annotates frames with names and distances.
- Frames are sorted by timestamp and compiled into a summary video.

---

## Outputs

- **Faces_cropped/**: Cropped and augmented face images.
- **NamitKush30_emb/**: Pickled lists of face embeddings and corresponding names.
- **output2_video.mp4**: The final summarized video with recognized faces annotated.

---

## Notes

- The code is designed for two classes (`Kush` and `Namit`), but can be extended by adding more folders to `Faces/`.
- GPU is recommended for efficient training and inference.
- The pipeline can be adapted for other face datasets and video sources.

---

## References

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [VGGFace2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- [PyTorch](https://pytorch.org/)

---

**Contact:**  
For questions or contributions, please open an issue or submit a pull request. Would love open suggestions and discussions and issues to impove this project further.
