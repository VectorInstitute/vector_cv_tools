# vector_cv_tools
Vector Computer Vision Project Tooling support

## Introduction
This is a tool-kit provided by the AI Engineering team for the Computer Vision project at Vector Institute. It includes various datasets readily loadable from the shared cluster as well as useful image/video tools such as data augmentation and visualization utilities.

### datasets
Provides a list of dataset used for Object Detection, Image Segmentation, and Video Recognition tasks.

Image:

- MSCOCO 2017: image captioning, detection, and segmentation
- Cityscape: segmentation
- MVTec: Anomoly detection and segmentation for common objects

Video:

- ActivityNet: Videos of human activities
- Kinetics-700: Videos including human-object interactions as well as human-human interactions.


### transforms
* Various data augmentation transforms considered useful for CV tasks

## Requirements

To install the requirements for the package, run
```
pip install -r requirements.txt
pip install pycocotools
```


## Installation

To install the package, run
```
git clone https://github.com/VectorInstitute/vector_cv_tools.git
cd vector_cv_tools
pip install -e .
```
