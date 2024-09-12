#importation of libraries needed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#iris dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # measurements
y = iris.target  # species

#this creates a table (DataFrame) with the measurements and species
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = y

#to see the first few rows of the data
print(df.head())

#splits the data, using 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#Decision Tree model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

#Predict with KNN
y_pred_knn = knn.predict(X_test)

#Predict with Decision Tree
y_pred_dtree = dtree.predict(X_test)

#to see how well the KNN model performed
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# to see how well the Decision Tree model performed
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dtree))
print(classification_report(y_test, y_pred_dtree))

#to visualize the performance of the KNN model
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for KNN - Spyros Gewrgiou')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#to visualize the performance of the Decision Tree model
cm_dtree = confusion_matrix(y_test, y_pred_dtree)
sns.heatmap(cm_dtree, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Decision Tree - Spyros Gewrgiou')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()




