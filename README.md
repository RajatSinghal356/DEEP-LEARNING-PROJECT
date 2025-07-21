# DEEP-LEARNING-PROJECT

*COMPANY*:  CODTECH IT SOLUTIONS

*NAME*:  RAJAT SINGHAL

*INTERN ID*:  CT04DG3154

*DOMAIN*:  DATA SCIENCE

*DURATION*:  4 WEEKS

*MENTOR*:  NEELA SANTOSH

##DESCRIPTION

In this deep learning project executed using Jupyter Notebook, a Convolutional Neural Network (CNN) was built and trained to classify images of cats and dogs. The project begins with the import of essential libraries from TensorFlow and Keras, including modules for constructing neural networks and preprocessing images, as well as matplotlib for visualization purposes. The dataset, compressed in a ZIP file, contains labeled images of cats and dogs, and it is extracted into a working directory using the `zipfile` module. Once extracted, the project sets up an image data pipeline using Keras’s `ImageDataGenerator` class. This class not only rescales the pixel values of the images from 0–255 to 0–1 but also allows for splitting the dataset into training and validation subsets, with 80% of the data used for training and the remaining 20% for validation. The images are resized to 150x150 pixels and fed into the network in batches of 32 using the `flow_from_directory` method, which automatically assigns binary labels based on subdirectory names.

The next step involves constructing the CNN model using the Sequential API from Keras. The architecture consists of three convolutional layers with increasing filter sizes (32, 64, and 128) and ReLU activations, each followed by max pooling layers to reduce the spatial dimensions of the data. After the convolutional base, the data is flattened and passed through a dense (fully connected) layer with 512 units and a ReLU activation function. A dropout layer with a rate of 0.5 is added to mitigate overfitting by randomly disabling neurons during training. The final output layer consists of a single neuron with a sigmoid activation function, which outputs a probability score indicating whether the image is a dog or a cat. The model is compiled with the binary cross-entropy loss function—appropriate for binary classification problems—alongside the Adam optimizer and accuracy as the performance metric.

Training the model is conducted over 10 epochs using the `fit()` function, with both training and validation data passed in to monitor the learning process. After training, two plots are generated using matplotlib to visualize how the training and validation accuracy, as well as the loss, evolve over time. These plots provide insight into the model’s learning behavior and whether overfitting or underfitting may be occurring. Finally, the model’s predictive capability is tested by taking a single sample image from the validation set. The image is displayed using matplotlib, and the model predicts whether it depicts a cat or a dog based on the output probability. The prediction is displayed along with the image, offering a real-world demonstration of the model's performance.

Overall, this project effectively showcases the implementation of a CNN for binary image classification using TensorFlow and Keras. It covers the entire workflow—from data extraction and preprocessing to model design, training, evaluation, and prediction—making it a comprehensive introduction to deep learning applied to computer vision tasks. The use of Jupyter Notebook allows for clear documentation and visualization, enhancing both the interpretability and reproducibility of the results.


