import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('patient_data.csv')
print(df.head())

# Features and Target
X = df[['Age','Blood_Pressure','Cholesterol','Glucose','BMI']]
y = df['Disease']

# Encode Yes/No to 1/0
y = y.map({'Yes':1, 'No':0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Building
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.1f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
importance = model.coef_[0]
plt.figure()
plt.bar(X.columns, importance, color='orange')
plt.title('Feature Importance - Disease Prediction')
plt.xticks(rotation=30)
plt.savefig('feature_importance.png')
plt.show()

# Prediction Distribution
plt.figure()
df['Disease'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['green','red'])
plt.title('Disease Distribution')
plt.savefig('disease_pie.png')
plt.show()

print("Top Feature:", X.columns[importance.argmax()])
