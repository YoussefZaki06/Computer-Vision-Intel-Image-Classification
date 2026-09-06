# Computer Vision — Intel Image Classification

This repository contains the Computer Vision assignment using the Intel Image Classification dataset.

## Contents

- `Computer Vision — Intel Image Classification.ipynb` — Google Colab/Jupyter notebook containing dataset exploration, preprocessing, augmentation, CNN training, evaluation, error analysis, and MobileNetV2 transfer learning.
- `final_model.keras` — exported Keras model for image classification.

## Dataset

The project uses the **Intel Image Classification** dataset with six scene classes.

The dataset is not included in this repository. Download/access the dataset separately and update the dataset paths in the notebook if necessary.

Expected directory structure:

```text
intel-image-classification/
├── seg_train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── seg_test/
    ├── buildings/
    ├── forest/
    ├── glacier/
    ├── mountain/
    ├── sea/
    └── street/
```

## Requirements

Recommended environment:

- Google Colab
- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Pillow

The notebook imports the required Python libraries. In Google Colab, most dependencies are already available.

## Running the notebook

1. Open the `.ipynb` file in Google Colab.
2. Enable a GPU runtime if available.
3. Mount Google Drive when prompted.
4. Make sure the Intel Image Classification dataset is available at the path configured in the notebook.
5. Run the notebook from top to bottom.
6. The notebook creates the training/validation/test pipelines, trains the baseline CNN, evaluates it, performs error analysis, trains MobileNetV2 using transfer learning, compares the models, and demonstrates image prediction.

## Loading the saved model

```python
import tensorflow as tf

model = tf.keras.models.load_model("final_model.keras")
```

For a single image:

```python
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

img = load_img("path/to/image.jpg", target_size=(150, 150))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

predictions = model.predict(x, verbose=0)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(predicted_class, confidence)
```

## Reproducibility

Random seeds are set in the notebook where appropriate. Results can still vary slightly depending on the runtime, TensorFlow version, hardware, and pretrained-model weight initialization.

## Project requirements

The notebook covers the required workflow:

- Dataset understanding and EDA
- Preprocessing and TensorFlow input pipelines
- Training-only data augmentation
- Baseline CNN
- Training curves
- Test-set evaluation
- Confusion matrix and per-class metrics
- Error analysis
- MobileNetV2 transfer learning
- Model comparison
- Reusable image prediction function
