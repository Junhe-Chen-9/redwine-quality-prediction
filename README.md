# Redwine Quality Prediction #

### Dataset in the repository is taken from UCI Machine Learning ###
- [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

### Description ###

Quality wine is important as world's wine consumption is over 1.1 billion gallons in 2021 already. It is hard to value a wine based on human quality assessment. We can use the features of a wine to accurately predict the quality score of a wine using algorithms. In this project we are going to build prediction model for the red wine based on 10 features and 4 strongest correlation features. We are trying to achive multi class classification to classify wine into 10 class ranging from score 1 to 10

### Model Evaluation ###

- ***Confusion matrix:*** For binary classification, a confusion matrix is a 2x2 table that compares a model's output compared to the actual target values. Ideally a model's confusion matrix should contain high true positive and true negative rates.
- ***Accuracy:*** Accuracy is the percentage of types that are correctly classified by the model. It is the sum
of true positives and true negatives divided by the sum of positives and negatives.
- ***Precision:*** Precision is the percentage of tuples output as positive that are actually positive. It is the number of true positives divided by the sum of the true positives and false positives.
- ***Recall:*** Recall is the percentage of tuples output as negative that are actually negative. It is the number of true positives divided by the sum of the true positives and false negatives. In other words is it the number of true positives divided by the positive tuples. Another word for recall is sensitivity.
- ***Specificity:*** Specificity is the true negative rate. In other words it is the proportion of negative tuples that the model identified correctly.
- ***Elbow Method:*** The elbow method will provide better insight on what value to choose for K in K means cluster analysis

### Experimental Setup ###
***Software :*** we hosted our jupyter notebook as an anaconda docker container on a virtual machine behind a reverse proxy so it can be accessed over web browser with a password, but essentially our source code can be run on any jupyter notebook instances.
***Hardware :*** The virtual machine is running on a Proxmox virtual environment configured to have 8 vCPUs (physical CPU is ryzen 7700X) and 8GiB of DDR5 RAM, to provide fast computation.

### Results ###
SVC model built upon only strong corrolation features performs best with accuracy of 96 percent whiele combined precision of 0.97, recall of 0.96 f1-score of 0.96. It had a high accuracy, high recall, high specificity. While random forest with strong corrolation features model perfomrs too well suspected it is overfited.
