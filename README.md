# FASHION-MNSIT-ASSIGNMENT
This project enabled me build a Convolutionary Neural Network(CNN) to classify images from the Fashion MNIST dataset.I started by installing the necessary libraries like tensorflow,matplotlib,numpy and pandas.The dataset became incuded by using tf.keras.dataset.fashionmnist.x_train and x_test was for image data while y_train and y_test was for labels or the integers which represent clothing categories.I normalized pixel values.I then reshaped the images to include a channel dimension.This converted the shape from (28,28) to (28,28,1) which is required for CNN input. 
## CNN Modelling Architecture
I constructed a six layer CNN using tf.keras.models.Sequential.
  -Conv2D was used to extract spatial features using convolution filters
  -MaxPooling2D downsampled feature maps to reduce computation
  -Flatten converted 2D features to 1D
  -Dense connected layers for classification
  -Softmax ouputed class probabilities
### Model Compilation
I compiled the model using various features.
   -Adam which is the adaptive learning rate optimizer
   -Sparse categorical crossentropy for the integer labels of rang 0 to 9
   -Accuracy is for monitoring classification performance
#### Model train & prediction
I trained the model using 5 epochs while using 10% of the training set for validation
I ran inferences on 2 test images and used argmax to get the predicted label which is the index of the highest probability
##### Visualizing predictions
I displayed the test images alongside the predicted class
