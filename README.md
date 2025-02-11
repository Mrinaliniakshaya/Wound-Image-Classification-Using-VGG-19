# Wound Image Classification using VGG19

## Overview
This project focuses on classifying wound images using deep learning techniques, specifically leveraging the VGG19 architecture. The model is trained to differentiate between various wound types based on image data, aiming to assist in automated medical diagnosis.

## Features
- Utilizes **VGG19** for feature extraction and classification.
- Addresses **dataset scarcity** and **class imbalance** through augmentation and resampling techniques.
- Implements **transfer learning** to enhance accuracy with a limited dataset.
- Focuses on **interpretability** to ensure reliability in medical applications.

## Tech Stack
- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn
- **Model**: VGG19 (Pretrained on ImageNet)
- **Dataset**: Wound images from publicly available medical datasets

## Project Structure
```
📂 wound-classification
├── 📂 data
│   ├── train
│   ├── test
│   ├── validation
├── 📂 models
│   ├── vgg19_model.h5
├── 📂 notebooks
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── evaluation.ipynb
├── 📂 utils
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualization.py
├── requirements.txt
├── README.md
├── train.py
├── test.py
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/wound-classification.git
   cd wound-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and prepare the dataset (place images in `data/train`, `data/test`, and `data/validation`).

## Model Training
To train the VGG19 model, run:
```bash
python train.py
```
This will preprocess the dataset, apply data augmentation, and fine-tune VGG19 on wound images.

## Evaluation
Run the evaluation script to test model performance:
```bash
python test.py
```
Metrics such as accuracy, precision, recall, and confusion matrix will be displayed.

## Results
The trained model achieves **high accuracy** in classifying wound images, effectively distinguishing between different types. The results are visualized using **Grad-CAM** to interpret model predictions.

## Future Enhancements
- Experiment with **other architectures** like ResNet50 and EfficientNet.
- Incorporate **attention mechanisms** for better feature extraction.
- Develop a **web-based interface** for real-time classification.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Inspired by medical imaging research on wound classification.
- Uses **publicly available wound image datasets** for training.

---

For contributions or queries, feel free to reach out!

 
