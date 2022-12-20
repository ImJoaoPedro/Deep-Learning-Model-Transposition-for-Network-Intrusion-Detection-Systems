import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

dataframe_file = 'flows/dataframe.csv'
outcomes_file = 'flows/outcomes.csv'
model_file = 'models/'

df = pd.read_csv(dataframe_file)
outcomes = pd.read_csv(outcomes_file)

print('dataframe shape is', df.shape)

x_train, x_test, y_train, y_test = train_test_split(df, outcomes, stratify=outcomes)

x_test = np.array(x_test)
y_test = np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

model = keras.models.load_model(model_file)

print(model.summary())

predictions = model.predict(x_test)
predictions = np.round(predictions).astype(int)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, digits=4))
