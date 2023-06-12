# Road-Sign-Detection
A project undertaken to detect road signs in the Greater Toronto Area for an Autonomous Vehhicles. 

The project included the following stages: 

1. Data Pre-Processing: The images fed were standardized to 30x30 images to enure equality. Min-Max normalization was performed to ensure the data was scaled.
2. Train-Test Split: The datset was divided to ensure an 80-20 split in form of training data and test data.
3. Model Training: The model was trained using the Keras interfaces. A convolutional neural network was built, comprising of convolutional, pooling, flattening, fully connected, and dropout layers. The 'adam-prop' optimizer along with a 'log-loss' cost function is used to compile the model.
4. Testing: Using the test data, predictions are made.
5. Evaluation: Metrics such as precision, recall, accuracy, f1 score are used to evaluate the model. An accuracy of approximately 85%-89% was obtained across the various metrics.

Following are the libraries/frameworks required to run the code:

1. Tensorflow
2. Keras
3. Sci-Kit Learn
4. Numpy
5. Pandas
6. Matplotlib

All the libraries/frameworks were installed using the pip install 'library-name' command on CMD or the terminal of your IDE.

