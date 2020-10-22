# Binary-Classification-on-Structured-Data
We use PetFinder dataset and want to predict whether the pet was adopted, or not.

# The dataset

 We use the PetFinder dataset (https://www.kaggle.com/c/petfinder-adoption-prediction).Each row describes a pet, and each column describes an attribute. We use this information to predict if the pet will be adopted.
 
 # Data Preprocessing
The Keras preprocessing layers API allows you to build Keras-native input processing pipelines.   
1. Normalization - Feature-wise normalization of the data.
2. CategoryEncoding - Category encoding layer.
3. StringLookup - Maps strings from a vocabulary to integer indices.
4. IntegerLookup - Maps integers from a vocabulary to integer indices.
 # Numeric columns
 For each of the Numeric feature, you will use a Normalization() layer to make sure the mean of each feature is 0 and its standard deviation is 1.
 # Categorical columns
 In this dataset, Type is represented as a string (e.g. 'Dog', or 'Cat'). The strings are converted to integer indices. Then, the integer indices are converted to one-hot encoding.  
 
 The Numeric and Categorical Columns are concatenated into a single Layer and then they fed into to network.
 
 # Architecture of the Model
 
 1. All Features Layer: A Layer where both numerical and categorical columns are combined.
 2. Dense Layer: A Fully Connected Layer with 32-units and Relu activation function
 3. Dropout: With propability of 0.5.  
 4 Dense Layer: A Fully Connected Layer with 1-unit and Relu activation function.  
 
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Data/blob/main/connectivity_graph.png)

# Compile and Train the Model

The model is :  
 1: Compiled with BinaryCrossentropy as Loss function  and Adam as optimizer.    
 2. Trained for 10 epochs and the results:  `loss: 0.5358 - accuracy: 0.7139 - val_loss: 0.5067 - val_accuracy: 0.7546`.  
 3. Evaluated on Test Data and the results: `loss: 0.5143 - accuracy: 0.7439`.  
 
# Inference on new data
 
1. Save and reload the model.
2. Wrap scalars into a list so as to have a batch dimension.
3. Each feature is converted to tensor.

The Prediction is: `This particular pet had a 83.4 percent probability of getting adopted`.
