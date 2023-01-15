# Melanoma Skin Cancer detection using CNN for Image recognition 
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


## Dataset
The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images.
To overcome the issue of class imbalance, used a python package  Augmentor (https://augmentor.readthedocs.io/en/master/) to add more samples across all classes so that none of the classes have very few samples.

## Data Augmentation
Data augmentation is a technique used in image processing to artificially increase the size of a dataset by applying random modifications to the images. It is important in image processing because it can help to:

1. Increase the diversity of the dataset: By applying random modifications to the images, data augmentation can increase the diversity of the dataset, which can make the model more robust to variations in the input data.

2. Prevent overfitting: Overfitting occurs when a model is trained to fit the training data too closely and is not able to generalize well to new data. Data augmentation can help to prevent overfitting by providing the model with a larger and more diverse set of training data.

3. Improve the generalization of the model: By training the model on a diverse set of images, it can learn features that are more generalizable to new images. This can lead to better performance on the test set and on new images it hasn't seen before.

4. Handle class imbalance: In some cases, the dataset may have an imbalance in the number of images for each class. Data augmentation can be used to artificially balance the dataset by generating new samples for under-represented classes.

There are many ways to perform data augmentation, such as flipping, rotating, zooming, and adding noise to images. It's important to perform data augmentation in a way that preserves the original semantics of the image.

The following 3 Keras preprocessing layers are used for data augmentation during model training

tf.keras.layers.RandomFlip
tf.keras.layers.RandomRotation
tf.keras.layers.RandomZoom

## CNN Architecture Design
A Custom CNN model was used to classify different class of images including benign and non-benign skin lesions images.

The following layers are used in CNN

- Keras Rescalling Layer - To rescale an input in the [0, 255] range to be in the [0, 1] range.

In a convolutional neural network (CNN), images are typically rescaled before being passed through the network for a few reasons:

1. Computational efficiency: The convolutional layers in a CNN involve a large number of multiplications and additions, and having smaller input images can reduce the computational load on the network.

2. Regularization: Rescaling images can also be used as a regularization technique. It can help prevent overfitting by reducing the amount of information in the images, which can make it more difficult for the network to memorize the training data.

3. Standardization: Rescaling can also be used to standardize the images to a common scale, which can make it easier for the network to learn features that are relevant across different images.

4. Better model performance: Rescaling the images can also lead to better model performance by standardizing the input and making the features learned by the model more generalizable.

- Keras Preprocessing Layers for data augmentation
tf.keras.layers.RandomFlip
tf.keras.layers.RandomRotation
tf.keras.layers.RandomZoom

- Convolutional Layer - A convolutional layer applies a set of filters to the input data, where each filter is designed to detect a specific feature in the input. The filters are small matrices that are "slid" over the input data, element-wise multiplied with the input data and then summed up. This process is called a convolution, hence the name "convolutional layer". The output of this operation is called a feature map, which represents the presence of the features detected by the filters in the input data.  
- Pooling Layer - A pooling layer is used to reduce the spatial dimensions of the feature maps outputted by the convolutional layers. This is done by applying a pooling operation, such as max pooling or average pooling, to the feature maps. The pooling operation is typically applied to small sub-regions of the feature maps, called pooling windows, and the output of the operation is a single value for each pooling window. This process reduces the dimensionality of the feature maps, while also helping to introduce some degree of translational invariance to the features detected by the convolutional layers.
- Dropout Layer - A dropout layer is used to reduce overfitting by randomly dropping out, or setting to zero, a certain proportion of the values in the input data during training. The dropout rate is a hyperparameter that determines the proportion of values that are dropped out, typically set between 0.2 and 0.5. The idea behind dropout is that by dropping out different neurons during training, the network is forced to learn multiple independent representations of the input data, making it less likely to overfit to the training data.
- Flatten Layer - In a neural network, a flatten layer is used to convert the high-dimensional tensor output of previous layers into a one-dimensional tensor. This is typically done before passing the data through a fully connected layer, which expects one-dimensional inputs.
- Dense Layer - A dense layer, also known as a fully connected layer, is a layer in which every neuron receives input from every neuron in the previous layer. These layers are called dense because they are densely connected. In other words, each neuron in a dense layer is connected to every neuron in the preceding layer. The dense layers are typically used as the final layers in a neural network, where the output of the previous layers is processed and used to make predictions or classifications.
- Activation Function(ReLU) - In a neural network, an activation function is applied to the output of each neuron to introduce non-linearity into the network. ReLU (Rectified Linear Unit) is one of the commonly used activation functions.
It's defined as f(x)=max(0,x) , which means if the input to the neuron is positive then it will output the same value, otherwise, it will output zero. This activation function helps to alleviate the vanishing gradient problem, which occurs when the gradients become very small during backpropagation, and makes the training of deep networks easier. ReLU is computationally efficient and does not saturate for high input values, which makes it a popular choice for the hidden layers of neural networks.
- Activation Function(Softmax) - The softmax function is an activation function that is typically used in the output layer of a multi-class classification problem. It is used to convert the raw output of a neuron into a probability distribution over the classes.
The softmax function takes in a vector of real numbers and maps it to a probability distribution. It works by taking the exponential of each element in the input vector, and then normalizing the resulting vector so that the elements sum to 1. The output of the softmax function is a probability distribution over the classes, where the class with the highest probability is typically chosen as the prediction.
The softmax function is a generalization of the logistic function and is often used as the final activation function in neural networks that are used for multi-class classification.