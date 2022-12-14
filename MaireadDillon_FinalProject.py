"""Creating a neural network to predictor diabetes on a variety of predictor variables."""

# Import libraries
import pandas as pd
import numpy as np

# Read csv
diabetes = pd.read_csv(
    "/Users/maireaddillon/Documents/Duke/BIOS-823/Final Project/diabetes.csv"
)


# Create normalization function
def normalize(df):
    """Create a function to normalize the predictor variables. Note that this function assumes that the data has 8 predictor variables and 1 outcome variable."""
    result = df.copy()
    for col in df.columns[0:8]:
        max_value = df[col].max()
        min_value = df[col].min()
        result[col] = (df[col] - min_value) / (max_value - min_value)
    return result


# Save normalized diabetes dataframe
diabetes_normalized = normalize(diabetes)

# Create a function to calculate the sigmoid for the neural network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Create a function to make predictions
def make_prediction(input_vector, weights, bias):
    layer1 = np.dot(input_vector, weights) + bias
    layer2 = sigmoid(layer1)
    return layer2


weights_ex = (np.random.randint(0, 10, size=(9)) / 100) + 0.001


print(make_prediction(diabetes_normalized.iloc[0], weights_ex, np.array([0.0])))

# Create neural network
def neural_net(df, bias):
    for row in df:
        input_vector = row.columns[0:8]
        weights1 = (np.random.randint(0, 10, size=(len(input_vector))) / 100) + 0.001
        prediction = make_prediction(input_vector, weights1, bias)
    return layer2
