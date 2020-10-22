# Binary-Classification-on-Structured-Data
We use PetFinder dataset and want to predict whether the pet was adopted, or not.

# The dataset

 We use the PetFinder dataset(https://www.kaggle.com/c/petfinder-adoption-prediction).Each row describes a pet, and each column describes an attribute. We use this information to predict if the pet will be adopted.
 
 # Data Preprocessing
 
 Numeric columns: For each of the Numeric feature, you will use a Normalization() layer to make sure the mean of each feature is 0 and its standard deviation is 1.
 Categorical columns: In this dataset, Type is represented as a string (e.g. 'Dog', or 'Cat'). We cannot feed strings directly to a model. The preprocessing layer takes care of representing strings as a one-hot vector.
 
 # Create, compile, and train the model

![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Data/blob/main/connectivity_graph.png)
