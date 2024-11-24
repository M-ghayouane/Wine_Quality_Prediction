import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import mean_squared_error, accuracy_score 
data = pd.read_csv("winequality-red.csv") 
data.head(10) 
 
# Check for missing values 
missing_values = data.isnull().sum() 
 
# Print the number of missing values in each column 
print(missing_values) 
# Separate features and target 
X = data.drop('quality', axis=1) 
y = data['quality'] 
 
# Split data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.2, random_state=42) 
 
# Initialize the SVM classifier 
model = SVC(kernel='linear') 
# Train the model on the training data 
model.fit(X_train, y_train) 
# Print the model parameters 
 
 
# Initialize the KNN classifier 
knn_model = KNeighborsClassifier(n_neighbors=3) 
 
# Train the model on the training data 
knn_model.fit(X_train, y_train) 
 
# Evaluate the SVM model on the test data 
svm_predictions = model.predict(X_test) 
 
# Calculate the RMSE for the SVM model 
svm_rmse = np.sqrt(mean_squared_error(y_test, svm_predictions)) 
# Calculate the accuracy for the SVM model 
svm_accuracy = accuracy_score(y_test, svm_predictions) 
# Evaluate the KNN model on the test data 
knn_predictions = knn_model.predict(X_test) 
# Calculate the RMSE for the KNN model 
Knn_rmse= np.sqrt(mean_squared_error(y_test,knn_predictions )) 
# Calculate the accuracy for the KNN model 
knn_accuracy = accuracy_score(y_test, knn_predictions) 
 
# Print the results 
print("SVM RMSE:", svm_rmse) 
print("KNN RMSE:", Knn_rmse) 
 
print("SVM Accuracy:", svm_accuracy) 
print("KNN Accuracy:", knn_accuracy) 
 
# Visualize the results for the SVM model 
import matplotlib.pyplot as plt 
 
plt.figure(figsize=(8, 6)) 
plt.scatter(y_test, svm_predictions, color='blue', label='SVM Predictions') 
plt.plot([0, 10], [0, 10], color='red', linestyle='--', label='Ideal Line') 
plt.xlabel('Actual Quality') 
plt.ylabel('Predicted Quality') 
plt.title('SVM Predictions vs. Actual Quality') 
plt.legend() 
plt.show() 
 
# Visualize the results for the KNN model 
plt.figure(figsize=(8, 6)) 
plt.scatter(y_test, knn_predictions, color='green', label='KNN Predictions') 
plt.plot([0, 10], [0, 10], color='red', linestyle='--', label='Ideal Line') 
plt.xlabel('Actual Quality') 
plt.ylabel('Predicted Quality') 
plt.title('KNN Predictions vs. Actual Quality') 
plt.legend() 
plt.show()