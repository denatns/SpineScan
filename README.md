![SpineScan Logo](./logo/logo.png)

# SpineScan
**SpineScan** is a deep learning–based service for automatic annotation and Pfirrmann grading of lumbar spine MRI scans.  
The model identifies intervertebral discs and estimates their degeneration grades (I–V) according to the **Pfirrmann scale**.

**Online Service:** [spine-scan.science.nprog.ru](https://spine-scan.science.nprog.ru/)  
**Preprint:** [SpineScan: A Deep Learning Model for Lumbar Spine MRI Annotation and Pfirrmann Grading Assessment](https://www.researchgate.net/publication/393277852_SpineScan_a_deep_learning_model_for_lumbar_spine_MRI_annotation_and_Pfirrmann_grading_assessment)  
**Model Weights:** [spine_scan_yolov8x.pt](https://figshare.com/articles/software/args_yaml/29322854)



## Quick Start (Docker)

1. Download the model weights [`spine_scan_yolov8x.pt`](https://figshare.com/articles/software/args_yaml/29322854)  and place it into the `models/` directory.

2. Build the Docker image
```bash
docker build -t spine-scan:latest .
```

3. Run the container
```bash
docker run -d --rm \
  --name spine-scan \
  --gpus all \
  --ipc=host \
  -e PORT=8501 \
  -p 8501:8501 \
  spine-scan:latest
```
4. Open your browser and navigate to http://localhost:8501



Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
