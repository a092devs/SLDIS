import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_excel('iris_stress.xlsx')
data['stress_level'] = data['stress_level'] - 1

X = data[['iris_diameter', 'pupil_diameter', 'iris_to_pupil_ratio']]
y = data['stress_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

joblib.dump(clf, 'iris_stress_model.pkl')