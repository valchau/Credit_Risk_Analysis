# Credit_Risk_Analysis
module 18 using supervised machine learning
## Overview of this Analysis:
Credit risk is very important to predict because today many people are using various sorts of credit to buy businesses, cars, trucks, invest in city infrastructure and more. All of us adults are affected by our credit scores. They help us buy homes and cars at low interest rates or we can't get loans at all. According to the instructions for this task, financially viable loans ('good loans') are much more common than those that are made without viable financial support. ('bad loans'). That is good, since otherwise banks would not make loans. 

For this analysis, I reviewed 6 different machine learning algorithm results using the given data in a csv formmated file to predict low or high risk status. Data scientists often use machine learning algorithms to make predictions. In the first couple of models I oversampled the data using randomoversampler and smote algorithms and then undersampled the data with the clustercentroid algorithm. For the fourth  model I used a combination approach to over and undersample the data using SMOTEENN. Finally, I compared two machine learning models that minimize bias, balancedrandomforestclassifier and easyensembleclassifier.


## Tools used:
I used Jupyter Notebooks and Python's DataFrames to load in the data. Then I used creat a model and then evaluate and train the models that they create, using Python's imbalanced-learn and scikit-learn libraries to build models and evalute them using resampling methods.

## Results of 6 algorithms to determine credit risk:
Below are the results of the 6 machine learning algorithms used.

### Naive Random Oversampling
Naive Random Oversampling results: Our balanced accuracy score is 57%, the precision is 99% and the recall is 70%.
<br>
<img src="https://github.com/valchau/Credit_Risk_Analysis/blob/main/NaiveRandomOverSampling.PNG" alt="NaiveRandomOverSampling" width="500" height="500" >
<br>


### SMOTE Oversampling
SMOTE oversampling results: the accuracy score is 64.7%, the precision is 99% and recall is 74% overall.
<br>
<img src="https://github.com/valchau/Credit_Risk_Analysis/blob/main/SMOTEOversampling.PNG" alt="SMOTEOversampling" width="500" height="500" >
<br>


### Cluster Centroid Undersampling
Undersampling results: balanced accuracy score is 64.7% overall, the precision is 99% and the recall is 55%.
<br>
<img src="https://github.com/valchau/Credit_Risk_Analysis/blob/main/UnderSamplingUsingClusterCentriods.PNG" alt="UnderSamplingUsingClusterCentriods" width="500" height="500" >
<br>

### Combination of Over and Under Sampling using SMOTEENN
Combination(over and undersampling) results: balanced accuracy score is 54.3% the precision is 99% and the recall is 58% overall
<br>
<img src="https://github.com/valchau/Credit_Risk_Analysis/blob/main/UnderAndOverSamplingUsingSMOTEENN.PNG" alt="UnderAndOverSamplingUsingSMOTEENN" width="500" height="500" >
<br>

### Ensemble using Balanced Random Forest
Balanced Random Forest Classifier results: the accuracy score is 77.2% the precision is 99% and the recall is 89%
<br>
<img src="https://github.com/valchau/Credit_Risk_Analysis/blob/main/BalancedForestEnsemble.PNG" alt="BalancedForestEnsemble" width="500" height="500" >
<br>

### Easy Ensemble AdaBoost
Easy Ensemble AdaBoost Classifier results: the accuracy score is 91.7% the precision is 99% and the recall is 94%.
<br>
<img src="https://github.com/valchau/Credit_Risk_Analysis/blob/main/EasyEnsemble.PNG" alt="EasyEnsemble" width="500" height="500" >
<br>

## Summary:
In the first four models the algorithms undersampled, oversampled and did a combination of both to determine which model is best at predicting which loans are the highest risk. 

The next two models resampled the data using ensemble classifiers to predict which which loans are high or low risk. Reviewing the results of first four models, the accuracy scores are not as high either as the ensemble classifiers.  The recall in the oversampling/undersampling/mixed models is low as well. Typically in prediction models, there should be a balance between recall and precision. Therefore, I would recommend the ensemble classifiers over the first four models. It appears that the Easy Ensemble had the best balance of all the models because of its high accuracy score and good balance of precision and recall scores, so that is the model I recommend.
