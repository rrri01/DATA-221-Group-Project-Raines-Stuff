# github repo link for my own reference: https://github.com/rrri01/DATA-221-Group-Project-Raines-Stuff
# group repository link for my own reference: https://github.com/chloeptrsn/DATA-221-Project.git

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import numpy as np

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

california_house_prices = pd.read_csv('housing.csv', delimiter=',')

california_house_prices = california_house_prices.replace({"NEAR BAY": 0, "<1H OCEAN": 1, "INLAND": 2, "NEAR OCEAN": 3, "ISLAND": 4})


# creates feature matrix X of all columns except "median_house_value" and create label vector y as "median_house_value"
feature_matrix = california_house_prices.drop("median_house_value", axis=1)
target_prices = california_house_prices["median_house_value"]

# splits the data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(feature_matrix, target_prices, test_size=0.3, random_state=42)

