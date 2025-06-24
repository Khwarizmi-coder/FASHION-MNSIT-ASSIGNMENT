install_keras
install.packages("ggplot2")
library(keras)
library(ggplot2)

# Load Fashion MNIST
fashion_mnist <- dataset_fashion_mnist()
x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

# Normalize
x_train <- x_train / 255
x_test <- x_test / 255

# Reshape to add channel
x_train <- array_reshape(x_train, c(dim(x_train)[1], 28, 28, 1))
x_test <- array_reshape(x_test, c(dim(x_test)[1], 28, 28, 1))

# Build CNN model with 6 layers
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Training of the model
model %>% fit(x_train, y_train, epochs = 5, validation_split = 0.1)

# Predict on two test images
predictions <- model %>% predict(x_test[1:2, , , ])
predicted_classes <- apply(predictions, 1, which.max) - 1

# Plot images and predictions
for (i in 1:2) {
  img <- array_reshape(x_test[i, , , ], c(28, 28))
  ggplot() + 
    geom_raster(aes(x = 1:28, y = 1:28, fill = img)) + 
    scale_fill_gradient(low = "black", high = "white") + 
    ggtitle(paste("Predicted:", predicted_classes[i])) + 
    theme_void() + 
    coord_fixed()
}


