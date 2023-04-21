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
- **GaussianNB(Naive Bayes):** 
- **RandomForestRegressor:** 
- **SVC:** 
- **MLPClassifier(Neural Network):** 

### Model Evaluation ###

- **Cross Validation:** Because the only way to determine if a machine learning method has been ***Overfit*** to the Training Data is to try it on new data that it hasn't seen before. ***Cross Validation*** solves the problem of not knowing which points are the best for Testing by using them all in an iterative way. The first step is to randomly assign the data to different groups. When we divides the data into k group we would have to do k iterations, which ensures that each group is used for ***testing*** this is called ***k-Fold Cross Validation***. In this project, random forest model with strong corrolation features only model is seen to have accuracy of almost 100 percent that suggest that this model might be ***Overfit*** to the data set, that is why ***Cross Validation*** is introduced to ensure our model is working but not overfitted. ****3-fold cross validation is added into randomizedsearchCV when tuning hyperparameters on models****

### Experimental Setup ###
- ***Software :*** we hosted our jupyter notebook as an anaconda docker container on a virtual machine behind a reverse proxy so it can be accessed over web browser with a password, but essentially our source code can be run on any jupyter notebook instances.
- ***Hardware :*** The virtual machine is running on a Proxmox virtual environment configured to have 8 vCPUs (physical CPU is ryzen 7700X) and 8GiB of DDR5 RAM, to provide fast computation.

### Results ###
SVC model built upon only strong corrolation features performs best with accuracy of 96 percent whiele combined precision of 0.97, recall of 0.96 f1-score of 0.96. It had a high accuracy, high recall, high specificity. While random forest with strong corrolation features model perfomrs too well suspected it is overfited.
