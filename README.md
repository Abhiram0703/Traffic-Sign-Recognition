# Traffic Sign Recognition using Deep Learning

## Project Overview
This project focuses on the task of traffic sign recognition, implementing and evaluating several deep learning models to assess their effectiveness and efficiency. By leveraging both custom Convolutional Neural Networks (CNNs) and transfer learning models (`ResNet50`, `MobileNetV2`), we provide a comprehensive evaluation of model accuracy, inference time, and resource requirements, all essential for real-world deployment.

## Key Results
The project demonstrates high accuracy in the custom CNN model, with the following metrics highlighting each model’s performance:

| Model       | Test Accuracy | Test Loss | Inference Time (s) | Model Size (MB) | Memory Usage (MB) |
|-------------|---------------|-----------|---------------------|------------------|--------------------|
| Custom CNN  | **98.77%**    | 0.0436    | 0.0016             | 4.43            | 2524.48           |
| ResNet50    | 18.97%        | 6.4846    | 0.0127             | 102.65          | 2636.24           |
| MobileNetV2 | 44.41%        | 1.8762    | 0.0078             | 16.75           | 2740.39           |

- **Custom CNN** outperforms the transfer learning models, achieving 98.77% accuracy and minimal inference time, making it suitable for high-performance, low-latency environments.
- **ResNet50** and **MobileNetV2**, while powerful for large-scale image classification tasks, show limited effectiveness here without fine-tuning, given their significantly lower accuracy and larger memory footprint.

## Data Preprocessing
The dataset of categorized images is loaded, preprocessed, and normalized. Each image is resized to a standard dimension, ensuring compatibility across models. Data augmentation techniques are applied to enhance training data variability, contributing to better generalization.

## Model Architectures
1. **Custom CNN:** Optimized with batch normalization and dropout for robust learning and minimal overfitting.
2. **Transfer Learning Models (ResNet50, MobileNetV2):** Pre-trained on large-scale datasets but adapted to the traffic sign dataset, though with limited efficacy in this specific domain.

## Evaluation and Analysis
The custom CNN model demonstrates optimal balance in accuracy, efficiency, and resource utilization, making it the top choice for deployment scenarios requiring high accuracy with limited computational resources.

## Conclusion
This project validates the effectiveness of CNN-based approaches for traffic sign recognition. The impressive accuracy of the custom CNN model, along with its efficient memory and inference requirements, makes it highly suited for real-world applications like autonomous driving or traffic monitoring systems.

## Future Improvements
1. **Cross-Validation**: Implement cross-validation for further model robustness.
2. **Additional Architectures**: Explore models like VGG16 or EfficientNet for potential accuracy improvements.
3. **Gradual Layer Unfreezing**: Fine-tune transfer learning models by unfreezing layers in stages for enhanced accuracy.
4. **Interpretability**: Integrate Grad-CAM or SHAP values to better understand model decision-making.
