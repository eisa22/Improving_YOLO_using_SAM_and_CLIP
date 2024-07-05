# YOLO Object Detector with Segment Anything and CLIP Integration

This project implements an improved YOLO object detector by integrating the Segment Anything model and CLIP (OpenAI API) to enhance performance. 
![grafik](https://github.com/eisa22/Improving_YOLO_using_SAM_and_CLIP/assets/146633211/f6909885-e58c-4e02-8f17-2bc531fc8290)


## Features

- **YOLO Object Detection**: Robust object detection using YOLO.
- **Segment Anything Model**: Enhanced segmentation capabilities.
- **CLIP Integration**: Improved object detection accuracy with CLIP from OpenAI.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Credentials](#credentials)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Installation

To set up the environment for this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/yolo-object-detector.git
    cd yolo-object-detector
    ```

2. **Create a Conda environment using the provided YAML file**:
The yaml file is attached in the repository.
    ```bash
    conda env create -f environment.yml
    conda activate yolo-env
    ```


## Setup

After setting up the environment, ensure you have the necessary credentials for the OpenAI API and the Segment Anything model.

## Credentials

1. **OpenAI API Key**:
   - Sign up at [OpenAI](https://www.openai.com/) and obtain your API key.
   - Set your OpenAI API key as an environment variable:
     ```bash
     export OPENAI_API_KEY='your_openai_api_key'
     ```

2. **Segment Anything Model**:
   - Obtain access and credentials for the Segment Anything model from the respective provider.
   - Set your credentials as environment variables:
     ```bash
     export SAM_API_KEY='your_sam_api_key'
     ``

## Usage

To run the YOLO object detector with enhanced capabilities:

1. **Prepare your input data** (images).
2. **Control the pipeline via the control panel in `main.py`**:
   - You can adjust various parameters and control the workflow through the control panel in the `main.py` file.
