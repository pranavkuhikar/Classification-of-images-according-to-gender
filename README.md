# Classification-of-images-according-to-gender

Aim of this project is to classify images according to gender.
The image classification is done by open source computer vision library.
The dataset is obtained is from kaggle. Link: https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset
Once we have obtained the data, we need to segregate it into two folders i.e. female and male, containing images according to their respective gender.
Now, we need to create a loop to combine the images for the model to work.
It is important to note that the images could be of different size, and that would be a problem for the algorithm. The image_size is taken as 50 in this case.
It is time to create train data. and after that we can check the size of the data, which is 47009.
The images needs to be shuffled to avoid bais or class imbalance.
Further, it is essential to separate the features and labels from the train data. 

# Sampling of the data into train and test set.
Note that the dataset is normalised
X_train =X_train/255
X_test = X_test/255
There are 37607 images in the training data
and the test data contains 9402 images.

# Compile the model
when we are compiling the model, there are various hyper parameters involed and needs to be selected based on the performance after tuning.
model.add(Dense(128 , activation='relu'))
model.add(Dense(2 , activation='softmax'))
model.compile(optimizer = 'adam', loss= 'sparse_categorical_crossentropy' , metrics=['accuracy'])
model1 = model.fit(X_train , Y_train ,epochs=5 , validation_split= .2 ,batch_size  = 256)
Here the hyperparameters are epochs, validation_split, and batch_size. after various iterations and combinations the best values are chosen.

# building confusion matrix to evaluate the performance of the model.
The accuracy score was found to be 91.576 which is high.
# building classification report
  precision    recall  f1-score   support

           0       0.92      0.91      0.92      4701
           1       0.91      0.92      0.92      4701

    accuracy                           0.92      9402
   macro avg       0.92      0.92      0.92      9402
weighted avg       0.92      0.92      0.92      9402

We have a high precision score. Higher precision means that an algorithm returns more relevant results than irrelevant ones
Similarly, we have a high recall score. Models need high recall when you need output-sensitive predictions.
