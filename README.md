# Redwine Quality Prediction #

### Dataset in the repository is taken from UCI Machine Learning ###
- [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

### Description ###

Quality wine is important as world's wine consumption is over 1.1 billion gallons in 2021 already. It is hard to value a wine based on human quality assessment. We can use the features of a wine to accurately predict the quality score of a wine using algorithms. In this project we are going to build prediction model for the red wine based on 10 features and 4 strongest correlation features. We are trying to achive multi class classification to classify wine into 10 class ranging from score 1 to 10

### Data Investigation ###

***corrolation:*** In the dataset the residual sugar is not interested because its corrolation to our target quality is so low that it barely affect our model. 

### Models ###
**LogisticRegression:** 
**GaussianNB(Naive Bayes):** 
**RandomForestRegressor:** 
**SVC:** 
**MLPClassifier(Neural Network):** 

### Model Evaluation ###


### Experimental Setup ###
***Software :*** we hosted our jupyter notebook as an anaconda docker container on a virtual machine behind a reverse proxy so it can be accessed over web browser with a password, but essentially our source code can be run on any jupyter notebook instances.
***Hardware :*** The virtual machine is running on a Proxmox virtual environment configured to have 8 vCPUs (physical CPU is ryzen 7700X) and 8GiB of DDR5 RAM, to provide fast computation.

### Results ###
SVC model built upon only strong corrolation features performs best with accuracy of 96 percent whiele combined precision of 0.97, recall of 0.96 f1-score of 0.96. It had a high accuracy, high recall, high specificity. While random forest with strong corrolation features model perfomrs too well suspected it is overfited.
