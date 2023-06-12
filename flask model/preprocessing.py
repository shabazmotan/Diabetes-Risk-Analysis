import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy
# Load dataset
url = "C:/Users/82104/Desktop/flask model/diabetes_cleaned.csv"
df = pandas.read_csv(url)

# filling missing values

df.drop(['Unnamed: 0'], axis=1, inplace=True)


#label Encoder
category_col = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI',
                'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                'HvyAlcoholConsump','AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']
labelEncoder = preprocessing.LabelEncoder()

# creating a map of all the numerical values of each categorical labels.
mapping_dict = {}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
 
    le_name_mapping = dict(zip(labelEncoder.classes_,
                               labelEncoder.transform(labelEncoder.classes_)))
 
    mapping_dict[col] = le_name_mapping
print(mapping_dict)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
 
X = df.values[:, 1:]
y = df.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)
dt_clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)
y_pred_gini = dt_clf_gini.predict(X_test)

print ("Desicion Tree using Gini Index\nAccuracy is ", accuracy_score(y_test,y_pred_gini)*100 )

import pickle
pickle.dump(dt_clf_gini, open("model.pkl","wb"))