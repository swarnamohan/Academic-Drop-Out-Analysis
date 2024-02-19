import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pickle

# Read Dataset
acaddropout = pd.read_csv('acaddropout.csv')


# Split dataset into features and labels

acaddropout1 = acaddropout.drop('Target', axis =1)
labels = acaddropout.Target

# Create dataset without units
acaddropout2 = acaddropout.drop(['Target','Curricular units 1st sem credited','Curricular units 1st sem enrolled',
                                 'Curricular units 1st sem evaluations','Curricular units 1st sem approved',
                                 'Curricular units 1st sem grade','Curricular units 1st sem without evaluations',
                                 'Curricular units 2nd sem credited','Curricular units 2nd sem enrolled',
                                 'Curricular units 2nd sem evaluations','Curricular units 2nd sem approved',
                                 'Curricular units 2nd sem grade','Curricular units 2nd sem without evaluations'], axis =1)

#print(acaddropout.head())
# One Hot Encoding for string variables

features = pd.get_dummies(acaddropout1, columns=["Marital status", "Application Mode", "Application order", "Course", 
                                                    "attendance_time", "Previous Qualification", "Nationality", "Mother Qualification","Father Qualification",
                                                    "Mother Occupation","Father Occupation", "Gender"])

features2 = pd.get_dummies(acaddropout2, columns=["Marital status", "Application Mode", "Application order", "Course", 
                                                    "attendance_time", "Previous Qualification", "Nationality", "Mother Qualification","Father Qualification",
                                                    "Mother Occupation","Father Occupation", "Gender"])

print(features.head()) 
print(features2.head())

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state = 42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(features2, labels, test_size=0.30, random_state = 42)

# Using Grid Search to find the best parameters
param_grid1 = { 
    'n_estimators': [200],
    'max_features': [153],
    'max_depth' : [9,11],
    'criterion' :['gini'],
    'min_samples_split':[2]
    }

param_grid2 = { 
    'n_estimators': [100],
    'max_features': [141],
    'max_depth' : [13,15],
    'criterion' :['gini'],
    'min_samples_split':[2]
    }


# Training RF Model 1 with K-Fold of 5 
rf_models = GridSearchCV(RandomForestClassifier(random_state = 42), param_grid=param_grid1, cv=5, verbose=1)
rf_models.fit(X_train, y_train)
print(rf_models.best_params_)

# Training RF Model 2 with K-Fold of 5 
rf_models2 = GridSearchCV(RandomForestClassifier(random_state = 42), param_grid=param_grid2, cv=5, verbose=1)
rf_models2.fit(X_train2, y_train2)
print(rf_models2.best_params_)

# Get the predictions
predictions = rf_models.predict(X_test)
predictions2 = rf_models2.predict(X_test2)
#print(predictions)

# Print the Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(predictions, y_test))

# Print Feature Importance
feature_importance = pd.DataFrame(data={"features":X_test.columns, "importance":rf_models.best_estimator_.feature_importances_*100})
print("Feature Importance")
print(feature_importance.sort_values('importance', ascending=False).head(10))

# Print the 2nd Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(predictions2, y_test2))

# Print 2nd Model Feature Importance
feature_importance2 = pd.DataFrame(data={"features":X_test2.columns, "importance":rf_models2.best_estimator_.feature_importances_*100})
print("Feature Importance")
print(feature_importance2.sort_values('importance', ascending=False).head(10))

print("")

# Save Model 2 to SAS file (predict dropout rate for incoming students
#filename = 'model2.sav'
#pickle.dump(rf_models2, open(filename, 'wb'))
