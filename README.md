

**Goal -  Leverage Machine Learning to predict Diabetes Risk**
- Link to Presentation Slides: https://docs.google.com/presentation/d/1eVz6_SZBXEXu6ttVq3CUnKso_gyeXuqG_abdFgvSwhw/edit?usp=sharing

**Data Sourcing**

Original data was sourced from BRFSS 2015 Codebook Report (cdc.gov)
The Behavioral Risk Factor Surveillance System (BRFSS) is a health-related telephone survey collected annually by the CDC. Each year, the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic health conditions, and the use of preventative services. It has been conducted every year since 1984. 
For this project Kaggle version of the data for 2015 was used - 


For this project Kaggle version of the data for 2015 was used - 
[Diabetes Health Indicators Dataset | Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv)

The target variable Diabetes_binary has 2 classes: 0 is for no diabetes, and 1 is for prediabetes or diabetes.
Two data sets were used -
a) Balanced Data Set was used for Neural Network Model
b) Unbalanced Data Set was used for Logistic Regression Model, subsequently balanced using Random Oversampler


**Google Drive/Spark/Colab**

Google Drive used to store data

from google.colab import drive
drive.mount('/content/drive')

Colab was used for the code
Spark was used to read the data from the google drive and create tables.
The tables were subsequently transformed to Pandas Dataframe.
Data was separated into labels and features.
Test and training data were split using train_test_split


**Logistic Regression Model**

Sample Size : 253,680

Classification Report with Unbalanced data 

              precision    recall  f1-score   support

         0.0       0.88      0.98      0.92     54551
         1.0       0.53      0.15      0.23      8869

    accuracy                           0.86     63420
   macro avg       0.70      0.56      0.58     63420
weighted avg       0.83      0.86      0.83     63420

Looking at the metrics for each class:

For class 0.0:

Precision: 0.88 Recall: 0.98 F1-score: 0.92 Support: 54551 For class 1.0:

Precision: 0.53 Recall: 0.15 F1-score: 0.23 Support: 8869

Overall Accuracy: 0.86

The macro average F1-score is 0.58, which represents the average F1-score across both classes. The weighted average F1-score is 0.83, which takes into account the class imbalance by weighting the F1-score by the number of samples in each class.

These metrics provide insights into the performance of your classification model. It appears that the model performs well in predicting class 0.0 (with high precision and recall), but it struggles with class 1.0, as indicated by the low precision, recall, and F1-score. The accuracy of 0.86 suggests that the model is correct in its predictions for 86% of the samples.

Next Random Oversampler was used to balance the dataset 

Classification Report after Random Oversampler used to balance the dataset

              precision    recall  f1-score   support

         0.0       0.95      0.73      0.82     54551
         1.0       0.31      0.77      0.44      8869

    accuracy                           0.73     63420
   macro avg       0.63      0.75      0.63     63420
weighted avg       0.86      0.73      0.77     63420


These metrics indicate that the model has a relatively high precision for class 0 (negative class) but a low precision for class 1 (positive class). It has a higher recall for class 1 compared to class 0, indicating that it is better at identifying the positive class. The F1-score for class 0 is relatively high, while the F1-score for class 1 is lower, suggesting that the model's performance is better for the negative class.

The overall accuracy of the model is 0.73, which means it correctly predicts the class for approximately 73% of the instances. The macro average metrics provide an average measure across both classes, while the weighted average metrics take into account the class imbalance in the dataset.

  
**Neural Networks Model & Decision Tree Classifier**

Neural network: a complex and flexible model inspired by the structure and function of the human brain. It consists of interconnected nodes or "neurons" organized in layers. Neural networks can learn complex patterns and relationships in data

Decision tree: Represented as a tree-like structure where each internal node represents a feature or attribute, each branch represents a decision based on that attribute, and each leaf node represents the outcome or class label.Widely used for classification and regression tasks and are particularly useful when dealing with categorical or discrete data.

Keras Tuner: a library that provides tools for hyperparameter tuning in Keras, which is a popular deep learning framework

Keras Tuner Search: functionality within Keras Tuner that helps in performing an automated search for the best set of hyperparameters for a given model

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
========================================================
 dense (Dense)               (None, 9)                 198       
                                                                 
 dense_1 (Dense)             (None, 3)                 30        
                                                                 
 dense_2 (Dense)             (None, 1)                 4         
                                                                 
========================================================
Total params: 232
Trainable params: 232
Non-trainable params: 0
_________________________________________________________________
None
553/553 - 1s - loss: 0.5036 - accuracy: 0.753 - 931ms/epoch - 2ms/step

Decision Tree Classifier: Decision Tree using Gini Index
Accuracy: 0.710


**Flask**






**Dashboard (HTML/CSS, JavaScript)**

