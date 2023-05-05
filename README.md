# Redwine Quality Prediction #

### Dataset in the repository is taken from UCI Machine Learning ###
- [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

### Description ###

Quality wine is important as world's wine consumption is over 1.1 billion gallons in 2021 already. It is hard to value a wine based on human quality assessment. We can use the features of a wine to accurately predict the quality score of a wine using algorithms. In this project we are going to build prediction model for the red wine based on 10 features and 4 strongest correlation features. We are trying to achive multi class classification to classify wine into 10 class ranging from score 1 to 10

### Data Investigation ###

- ***corrolation:*** In the dataset the residual sugar is not interested because its corrolation to our target quality is so low that it barely affect our model. 

### Main Ideas ###

- **Machine Learning:** is a collection of tools and techniques that transforms data into decisions by making ***Classifications*** or ***Regression***.

- **Classifications:** Classification is used to classify wine into different quality grade, in this case, this is a ***multiclass*** classification problem. 


### Models ###
- **LogisticRegression:** Linear Regression and Linear Models are great when we want to predict somthing that is ***continuous*** and Logistic Regression comes to be very good at classify discrete variables. However, Logistic Regression is not very good with multi class classification, even we made ***multi_class='multinomial'*** Logistic Regression model is performing poorly as we would expected. 
- **GaussianNB(Naive Bayes):** In the Naive Bayes model, we start with a prior probability, which represents the proportion of each class in the training set. For example, in a wine classification problem, the prior probability could represent the percentage of wines that are rated a certain score (such as 3 points or above).Next, we calculate the conditional probability of each feature given each class label, based on the training data. given the features the model assign the conditaion probabilities, and the conditional probabilities would represent the likelihood of observing each feature given a certain score.Using Bayes' theorem, we can then compute the posterior probability of each class label given the observed features of a new wine sample. This involves multiplying the prior probability of the class label with the conditional probabilities of the observed features given that class label, and then normalizing the probabilities to sum up to 1. The class label with the highest posterior probability is then selected as the predicted class label for the new wine sample.Overall, the Naive Bayes model is a simple yet effective probabilistic model that can be trained quickly on large datasets, and it can perform well even when the independence assumption between features does not hold perfectly.
- **RandomForestRegressor:** It is an ensemble learning method that builds a large number of decision trees and combines their predictions to improve the accuracy and robustness of the model.Each decision tree in the random forest is built using a random subset of the training data and a random subset of the features. This helps to reduce overfitting and improve the generalization of the model. During the training process, the algorithm selects the best split point for each node in the decision tree, based on the information gain or Gini impurity criterion.
- **SVC:** SVC works by finding the optimal hyperplane in a high-dimensional feature space that separates the different classes of data. It does this by maximizing the margin between the hyperplane and the closest data points of each class. The data points that are closest to the hyperplane are called support vectors, hence the name Support Vector Machines.
- **MLPClassifier(Neural Network):** An MLP consists of multiple layers of nodes or neurons that are interconnected in a feedforward manner. The input layer receives the input data, which is then propagated through one or more hidden layers before reaching the output layer. Each node in the network applies a non-linear activation function to its input and passes the result to the next layer.

### Model Evaluation ###

- **Cross Validation:** Because the only way to determine if a machine learning method has been ***Overfit*** to the Training Data is to try it on new data that it hasn't seen before. ***Cross Validation*** solves the problem of not knowing which points are the best for Testing by using them all in an iterative way. The first step is to randomly assign the data to different groups. When we divides the data into k group we would have to do k iterations, which ensures that each group is used for ***testing*** this is called ***k-Fold Cross Validation***. In this project, random forest model with strong corrolation features only model is seen to have accuracy of almost 100 percent that suggest that this model might be ***Overfit*** to the data set, that is why ***Cross Validation*** is introduced to ensure our model is working but not overfitted. ****3-fold cross validation is added into randomizedsearchCV when tuning hyperparameters on models****

- ***Precision:*** Precision = TP / (TP + FP). Precision measures how many correct postive result against all postive result. I another word Precision represents the accuracy of positive predictions made by the model, and it is a measure of how often the model correctly identifies positive examples among all the examples it has predicted as positive.

- ***Recall:*** Recall = TP / (TP + FN). Recall measures the proportion of correctly predicted positive examples out of all the actual positive examples in the dataset. In other words, recall represents the ability of the model to correctly identify positive examples out of all the actual positive examples in the dataset. It is a measure of how well the model can detect positive examples, and it is particularly useful in scenarios where the cost of false negatives is high, such as in medical diagnosis or security systems, where a false negative could lead to serious consequences.

- ***f1-score:*** F1-score = 2 * ((precision * recall) / (precision + recall)). Measure of the overall accuracy of the model, taking into account both false positives and false negatives.

### Experimental Setup ###
- ***Software :*** we hosted our jupyter notebook as an anaconda docker container on a virtual machine behind a reverse proxy so it can be accessed over web browser with a password, but essentially our source code can be run on any jupyter notebook instances.
- ***Hardware :*** The virtual machine is running on a Proxmox virtual environment configured to have 8 vCPUs (physical CPU is ryzen 7700X) and 8GiB of DDR5 RAM, to provide fast computation.

### Results ###
SVC model built upon only strong corrolation features performs best with accuracy of 96 percent while combined precision of 0.97, recall of 0.96 f1-score of 0.96. It had a high accuracy, high recall, high specificity. While random forest with strong corrolation features model perfomrs too well suspected it is overfited.
