# Fruit AI Classifier

> **Archived Project:** This is a school seminar project from 2024. The code is not being maintained anymore and serves as a reference for my first projects

This repository contains the code, datasets, and documentation for a machine learning project developed as part of the **W-Seminar** (school seminar project) and presented at the **Jugend forscht** competition.

The objective of this project was to develop a Convolutional Neural Network using TensorFlow to assist supermarket cashiers by identifying fruits and vegetables captured by a camera. The model was optimized for edge computing by converting it to the .tflite format for real-time classification on a Raspberry Pi using a webcam.


##  Project Overview & Workflow

The development of this project was divided into two phases

1.  **Exploration & Testing:** Initial model training and testing were conducted using the public [Fruits360 dataset](https://github.com/Horea94/Fruit-Images-Dataset) to evaluate different architectures and experiment.
2.  **Custom Implementation:** The final model was trained on a custom-built dataset of fruit images created specifically for this project. The model was then converted to TensorFlow Lite for hardware deployment. The python script running on the Raspberry Pi is a modified version of the [official TensorFlow Lite example repository](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification).

##  Repository Structure

The repository is organized to separate the training environments, the deployment code, and the theoretical documentation:

* `docs/`: Contains the full German W-Seminararbeit (research paper) detailing the project history.
* `models/`: Stores the final, optimized `.tflite` model used for inference.
* `notebooks/`: Jupyter Notebooks containing the model training and evaluation processes.
* `project.py`: Python script for running the live camera inference on the Raspberry Pi.
* `requirements.txt`: Requirements for running the python script.

##  Datasets 

* [**Fruits360 Dataset:**](https://github.com/Horea94/Fruit-Images-Dataset)
* [**Custom Fruit Dataset Regular:**](https://huggingface.co/datasets/luis-bauer/fruits_vegetables_german_regular)
* [**Custom Fruit Dataset Cropped:**](https://huggingface.co/datasets/luis-bauer/fruits_vegetables_german_cropped)


##  Installation & Usage (Raspberry Pi)

Follow these steps to set up the project on your Raspberry Pi:

**1. Clone the repository:**
```bash
git clone https://github.com/luis-bauer/fruit-ai-classifier.git
cd fruit-ai-classifier
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the object detection:**
Execute the following command to start the camera feed and begin real-time classification:
```bash
python3 project.py
```

## Documentation & History
If you are interested in the complete developmental history including the initial tests with the Fruits360 dataset, model comparisons, parameter tuning, and the theoretical background of the neural networks used please refer to my complete **W-Seminararbeit**.