# Image Classification on CIFAR-10 with Different Approaches

## Task 1: Apply Filters on CIFAR-10 Images
In this task, we apply two simple filters to the CIFAR-10 dataset images. The dataset is loaded, and the first 10 images are selected for processing. The filters are manually defined as simple 3x3 matrices, and convolution operations are applied using TensorFlow.

### Approach:
1. Load CIFAR-10 dataset.
2. Apply two custom filters to the images using convolution.
3. Plot the original and filtered images side by side for visualization.

The filters used are:
- **Filter 1**: Diagonal identity matrix.
- **Filter 2**: A permutation of the identity matrix.

By applying these filters, we observe how the image features are highlighted based on the filter's structure.

### Results:
The original images and their corresponding filtered versions are displayed for visual comparison. Each filter produces different effects on the images, showcasing the importance of filter design in convolutional operations.

---

## Task 2: Training Deep Learning Models with Different Learning Rate Schedulers

In this task, we build and train two convolutional neural network models on the CIFAR-10 dataset. The models are trained using two different learning rate schedules: **Exponential Decay** and **OneCycle LR**.

### Approach:
1. **Data Preprocessing**: The CIFAR-10 dataset is normalized, and the labels are one-hot encoded. The data is then split into training, validation, and test sets.
2. **Model Architecture**: A CNN model is built using several convolutional layers, followed by max-pooling, dropout, and dense layers. Batch normalization is applied after each convolutional layer to improve training stability.
3. **Learning Rate Schedulers**:
   - **Exponential Decay**: The learning rate decays exponentially over time.
   - **OneCycle LR**: The learning rate first increases and then decreases during training.
4. **Model Training**: Both models are trained using the Adam optimizer, with the learning rate scheduler callbacks applied during training.

### Results:
- **Exponential Decay LR**: The model trains with a steadily decreasing learning rate and it works better.
- **OneCycle LR**: The learning rate follows a cyclical pattern, which can help the model converge faster.

We compare the performance of both models on the validation set, and analyze their training curves to understand the impact of the learning rate schedules on model performance.
