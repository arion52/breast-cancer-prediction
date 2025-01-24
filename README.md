# breast-cancer-prediction

This repository implements a breast cancer classification with a focus on differentiating between various magnification levels of images (`100X`, `200X`, `400X`, and `40X`). The models used are a custom CNN and a pre-trained ResNet50 with transfer learning.

## Dataset Used
[This](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) is the dataset that was used in this project. Detailed description and the link to download is provided in the link itself.


## Features

### Dataset Loading
- Images are organized into subfolders by class.
- A function processes the dataset structure and stratifies it into training, validation, and testing sets.

![Class Distribution curve](https://github.com/user-attachments/assets/eeb499ea-64ba-4fc1-99a7-e8c51f13efa3)

### Data Preprocessing
- Applied augmentation techniques like rotation, width shift, height shift, and flipping for the training dataset using `ImageDataGenerator`.
- Normalized pixel values and resized images to ensure compatibility with CNN and ResNet50 architectures.

### Models
1. **Custom CNN Model**:
   - A sequential architecture tailored for baseline image classification.
   - Includes multiple convolutional and pooling layers followed by fully connected dense layers.

2. **ResNet50**:
   - Utilized a pre-trained ResNet50 model from `keras.applications`.
   - Fine-tuned with additional dense layers and unfrozen convolutional layers for domain-specific learning.

### Training
- Both models are trained using the `Adam` optimizer and categorical cross-entropy loss function.
- Early stopping is employed to prevent overfitting.

### Evaluation
- Models are evaluated on precision, recall, F1-score, and AUC-ROC metrics.

## Results

### CNN
- **Classification Report**:
  ```
                precision    recall  f1-score   support

          100X       0.70      0.66      0.68       312
          200X       0.84      0.74      0.78       302
          400X       0.96      0.86      0.91       273
           40X       0.73      0.94      0.82       300

      accuracy                           0.80      1187
     macro avg       0.81      0.80      0.80      1187
  weighted avg       0.80      0.80      0.79      1187
  ```
- **AUC-ROC Score**: 0.9501

### ResNet50
- **Classification Report**:
  ```
                precision    recall  f1-score   support

          100X       0.22      0.31      0.26       312
          200X       0.14      0.07      0.10       302
          400X       0.61      0.38      0.47       273
           40X       0.27      0.39      0.32       300

      accuracy                           0.29      1187
     macro avg       0.31      0.29      0.29      1187
  weighted avg       0.30      0.29      0.28      1187
  ```
- **AUC-ROC Score**: 0.5743

## Visualization
![ResNet50 learning curve](https://github.com/user-attachments/assets/bbe0f67a-84de-4b9c-82b2-264cbaf695b6) ![CNN learning curve](https://github.com/user-attachments/assets/8bf5063b-592a-4cae-96ed-bdae8c3420e4)
---
