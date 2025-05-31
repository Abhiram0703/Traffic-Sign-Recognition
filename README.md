# Traffic Sign Recognition using Deep Learning

## Project Overview
This project implements a Traffic Sign Recognition system using deep learning, leveraging the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The system employs a custom Convolutional Neural Network (CNN) and transfer learning models (ResNet50, MobileNetV2) to classify 43 distinct traffic sign classes. A Streamlit-based web application enables users to upload images, manually crop traffic signs, and receive predictions with confidence scores and detailed descriptions. The project includes exploratory data analysis (EDA), preprocessing, model training, evaluation, and a user-friendly interface for real-world applications such as autonomous driving, driver assistance, and traffic management.

The custom CNN achieves **~98% accuracy** on the test set, with low inference time and minimal resource requirements, making it suitable for deployment in resource-constrained environments.

## Table of Contents
1. [Dataset](#dataset)
2. [Folder Structure](#folder-structure)
3. [Setup Instructions](#setup-instructions)
4. [Workflow](#workflow)
5. [Model Architectures](#model-architectures)
6. [Model Comparison and Results](#model-comparison-and-results)
7. [Running the Application](#running-the-application)
8. [Contributing](#contributing)
9. [Contact](#contact)

## Dataset
The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, available on Kaggle. To set up the dataset:
1. Download the dataset from [Kaggle](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
2. Extract the dataset and place it in the `Dataset/` folder with the following structure:
   ```
   Dataset/
   ├── Meta/                   # Metadata for traffic signs (Meta.csv)
   ├── Test/                   # Test images and Test.csv
   └── Train/                  # Training images and Train.csv
   ```
3. Ensure the `Meta.csv`, `Train.csv`, and `Test.csv` files are present, along with the image files organized in subdirectories by class.

## Folder Structure
The project is organized as follows:
```
Traffic-Sign-Recognition/
│
├── Dataset/                    # GTSRB dataset (to be downloaded by the user)
│   ├── Meta/                   # Metadata for traffic signs
│   ├── Test/                   # Test images and Test.csv
│   └── Train/                  # Training images and Train.csv
│
├── Model/                      # Trained model
│   └── custom_cnn_model.h5     # Saved custom CNN model
│
├── EDA plots/                  # Visualizations from EDA
│   └── 1-10 plots              # Plots for classes 1-10
│
├── Traffic_sign.ipynb          # Jupyter Notebook for EDA, preprocessing, and model training
├── app.py                      # Streamlit application for traffic sign recognition
├── model_comparison_results.csv # Model performance metrics
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
|-- custom_cnn_model.h5         # Saved custom CNN model
```

## Setup Instructions
To set up and run the project, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/Abhiram0703/Traffic-Sign-Recognition.git
cd Traffic-Sign-Recognition
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required packages using:
```bash
pip install -r requirements.txt
```
The `requirements.txt` should include:
```
streamlit
opencv-python
numpy
pillow
tensorflow
pandas
matplotlib
scikit-learn
tqdm
psutil
```

### 3. Download and Place the Dataset
- Download the GTSRB dataset from [Kaggle](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
- Extract and place the dataset in the `Dataset/` folder as described in the [Dataset](#dataset) section.

### 4. (Optional) Train the Model
The repository includes a pre-trained model (`custom_cnn_model.h5`) in the `Model/` folder. To train the model from scratch:
1. Open the `Traffic_sign.ipynb` notebook in Jupyter Notebook or Google Colab.
2. Update the `data_path` variable to point to your `Dataset/` folder:
   ```python
   data_path = './Dataset/'
   ```
3. Run the notebook to perform EDA, preprocess the data, train the model, and save it as `custom_cnn_model.h5`.

### 5. Run the Streamlit Application
To launch the web application:
```bash
streamlit run app.py
```
The application will open in your default browser, allowing you to upload images, crop traffic signs, and view predictions.

## Workflow
The project follows a structured workflow, as detailed in the `Traffic_sign.ipynb` notebook:

1. **Data Loading & Exploration**:
   - Load the GTSRB dataset (`Train.csv`, `Test.csv`, `Meta.csv`).
   - Perform EDA to analyze class distributions, image sizes, and other properties (39,209 training images, 12,630 test images).
   - Visualize sample images and class distributions, saved in `EDA plots/`.

2. **Preprocessing & Augmentation**:
   - Resize images to 48x48 pixels (RGB).
   - Normalize pixel values to [0, 1].
   - Apply selective data augmentation (e.g., rotation, zoom) to address class imbalance for minority classes.

3. **Model Building**:
   - Define a custom CNN architecture optimized for low inference time and high accuracy.
   - Experiment with transfer learning using pre-trained ResNet50 and MobileNetV2 models.
   - Use TensorFlow/Keras with batch normalization and dropout to prevent overfitting.

4. **Training & Evaluation**:
   - Train models for 30 epochs with a batch size of 32.
   - Use early stopping and model checkpointing to save the best model based on validation accuracy.
   - Evaluate models on the test set, measuring accuracy, loss, inference time, model size, and memory usage.

5. **Visualization & Comparison**:
   - Plot training and validation accuracy/loss curves (saved in `EDA plots/`).
   - Compare model performance to select the best model (`custom_cnn_model.h5`).

6. **Application Development**:
   - Develop a Streamlit application (`app.py`) for interactive traffic sign recognition.
   - Enable users to upload images, manually crop traffic signs, and view predictions with confidence scores and descriptions.

## Model Architectures
1. **Custom CNN**:
   - Lightweight architecture with batch normalization and dropout for robust learning and minimal overfitting.
   - Input size: 48x48 pixels (RGB).
   - Optimized for low inference time and high accuracy.

2. **Transfer Learning Models**:
   - **ResNet50**: Pre-trained on ImageNet, adapted for traffic sign classification, but shows limited accuracy without extensive fine-tuning.
   - **MobileNetV2**: Also pre-trained on ImageNet, designed for resource-constrained environments but underperforms compared to the custom CNN.

## Model Comparison and Results
The models were evaluated on the GTSRB test set (12,630 images) with the following metrics:

| Model       | Test Accuracy | Test Loss | Inference Time (s) | Model Size (MB) | Memory Usage (MB) |
|-------------|---------------|-----------|---------------------|------------------|--------------------|
| Custom CNN  | **97.91%**    | **0.0940** | **0.3314**          | **13.43**        | **8026.26**        |
| ResNet50    | 35.46%        | 2.7867    | 2.5799              | 102.65           | 8457.20            |
| MobileNetV2 | 52.40%        | 2.4344    | 1.3316              | 16.75            | 8586.24            |

### Relative Comparison
- **Accuracy**: The custom CNN significantly outperforms transfer learning models, achieving **97.91% accuracy** compared to 35.46% (ResNet50) and 52.40% (MobileNetV2). This indicates that the custom CNN is better suited for the specific task of traffic sign recognition, likely due to its tailored architecture and training on the GTSRB dataset.
- **Test Loss**: The custom CNN has the lowest test loss (0.0940), indicating better convergence and prediction confidence compared to ResNet50 (2.7867) and MobileNetV2 (2.4344).
- **Inference Time**: The custom CNN is the fastest, with an inference time of **0.3314 seconds**, making it ideal for real-time applications. ResNet50 (2.5799s) and MobileNetV2 (1.3316s) are significantly slower, likely due to their deeper architectures.
- **Model Size**: The custom CNN has a compact size of **13.43 MB**, compared to 102.65 MB for ResNet50 and 16.75 MB for MobileNetV2, making it more suitable for deployment on resource-constrained devices.
- **Memory Usage**: The custom CNN uses **8026.26 MB** of memory, slightly less than ResNet50 (8457.20 MB) and MobileNetV2 (8586.24 MB), indicating better resource efficiency.

### Analysis
The custom CNN is the optimal choice for this task due to its high accuracy, low inference time, and compact size. Transfer learning models (ResNet50, MobileNetV2) underperform without extensive fine-tuning, likely because their pre-trained weights are optimized for general image classification rather than the specific domain of traffic signs. The custom CNN's lightweight design and tailored training make it ideal for real-world applications like autonomous vehicles and traffic monitoring systems.

## Running the Application
1. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Upload an image containing traffic signs (JPG, JPEG, or PNG format, filename < 100 characters).
3. Use the sliders to crop individual traffic signs from the image.
4. Click **Add Crop for Analysis** to save the cropped region.
5. View the analysis results, including:
   - Predicted traffic sign class
   - Confidence score (in percentage)
   - Description of the traffic sign's meaning
   - Top-5 predictions with confidence scores and descriptions
6. Optionally, click **Clear All Crops** to start over.

The application simulates object detection by allowing manual cropping of traffic signs, ensuring accurate recognition even in complex images.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

Please ensure your code follows the project's coding standards and includes appropriate documentation.

## Contact
Developed by **Abhiram Madam**, B.Tech in Data Science and Artificial Intelligence, IIT Guwahati.

- **GitHub**: [Abhiram0703](https://github.com/Abhiram0703)
- **LinkedIn**: [Abhiram Madam](https://www.linkedin.com/in/abhiram-madam/)

For questions or feedback, feel free to reach out via GitHub or LinkedIn.
