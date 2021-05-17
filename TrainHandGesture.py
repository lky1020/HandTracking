import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

exportCsvPath = "HandGestureDataSet/" + "number.csv"
exportPicklePath = "HandGestureDataSet/" + "number.pkl"

df = pd.read_csv(exportCsvPath)

# Features
x = df.drop('class', axis=1)

# Target Value
y = df['class']

# Prepare data for test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

# Set up pipeline for train
pipelines ={
    'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

# Create Train Model
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(x_train, y_train)
    fit_models[algo] = model

# Check Accuracy
for algo, model in fit_models.items():
    yhat = model.predict(x_test)
    print(algo, accuracy_score(y_test, yhat))

# Create Pickle File
with open(exportPicklePath, 'wb') as f:
    pickle.dump(fit_models['rf'], f)