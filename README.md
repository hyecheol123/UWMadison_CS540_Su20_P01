# UWMadison_CS540_Su20_P01

Repository for the programming assignment of UW-Madison CS540 Summer 2020 course 
(Introduction to Artificial Intelligence)


## Tasks

Classify hand-written digits by **Logistic Regression model** and the **Neural Network**.  
The dataset is given [here](https://pjreddie.com/projects/mnist-in-csv/).
We are only using training set of the given dataset.  


### Data Pre-process

For both parts, as we are requested to make model to classify the binary class,
we need to extract two labeled cases among 10 possible labels.  
In this assignment, I am requested to use ***digits 4 (label 0) and 7 (label 1)*** case.  
Moreover, we also need to ***change pixel intensity to [0, 1]***.

All the parsing and data-preprocessing parts are implemented in
[Dataset.java](https://github.com/hyecheol123/UWMadison_CS540_Su20_P01/blob/master/Dataset.java).

[Related Questions]  
- **Q1**: save Feature Vector of any one training image.   
          Code located in [LogisticRegression]().saveFeatureVector() method.


### Logistic Regression Model

For the logistic regression model, we are finding the weights and bias to properly classify the digit image.
All codes are located in [LogisticRegression.java]().  
I modified the algorithm to calculate the loss based on the updated weights (re-calculated activation),
as I think it is much more reasonable as we are interested on the current model's performance,
not the previous model's one.  

[Related Questions]  
- **Q2**: save model weights and bias of the logistic regression model.  
          Code located in [LogisticRegression]().saveModel() method.  
- **Q3**: save activations on the test set.  
          Code located in [LogisticRegression]().____() method.  
- **Q4**: Save predicted values on the test set.  
          Code located in [LogisticRegression]().____() method.  

