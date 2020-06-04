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
          Code located in [LogisticRegression.saveFeatureVector()](https://github.com/hyecheol123/UWMadison_CS540_Su20_P01/blob/master/LogisticRegression.java#L204) method.


### Logistic Regression Model

For the logistic regression model, we are finding the weights and bias to properly classify the digit image.
All codes are located in [LogisticRegression.java](https://github.com/hyecheol123/UWMadison_CS540_Su20_P01/blob/master/LogisticRegression.java).  
I modified the algorithm to calculate the loss based on the updated weights (re-calculated activation),
as I think it is much more reasonable as we are interested on the current model's performance,
not the previous model's one.  
Note that as [LogisticRegression.java](https://github.com/hyecheol123/UWMadison_CS540_Su20_P01/blob/master/LogisticRegression.java)
handles four questions and saves log, it requires five command line arguments (CLAs).
Five CLAs are the string filepath that specifies the location of the text files 
that answer of each question be stored (first four arguments) and the log file.
To get more detailed explanation about the filename,
please go to [LogisticRegression.main()](https://github.com/hyecheol123/UWMadison_CS540_Su20_P01/blob/master/LogisticRegression.java#L74)
method's javaDoc comments.

[Related Questions]  
- **Q2**: save model weights and bias of the logistic regression model.  
          Code located in [LogisticRegression.saveModel()](https://github.com/hyecheol123/UWMadison_CS540_Su20_P01/blob/master/LogisticRegression.java#L321) method.  
- **Q3**: save activations on the test set.  
          Code located in [LogisticRegression.saveTestActivation()](https://github.com/hyecheol123/UWMadison_CS540_Su20_P01/blob/master/LogisticRegression.java#L356) method.  
- **Q4**: Save predicted values on the test set.  
          Code located in [LogisticRegression.saveTestPrediction()](https://github.com/hyecheol123/UWMadison_CS540_Su20_P01/blob/master/LogisticRegression.java#L392) method.  


### Neural Network Model


## Development/Testing Environment
- Oracle Java 11.0.5 2019-10-15 LTS
- Windows 10 Education 1909, Build 18363.836
- IDE: IntelliJ IDEA 2020.1.2 (Ultimate Edition), Build #IU-201.7846.76