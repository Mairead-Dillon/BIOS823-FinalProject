"""Creating a neural network to predictor diabetes on a variety of predictor variables."""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read csv
diabetes = pd.read_csv(
    "/Users/maireaddillon/Documents/Duke/BIOS-823/BIOS823-FinalProject/diabetes.csv"
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

# Split variables into predictors (X) and outcome (y)
X = diabetes_normalized[
    [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
]
y = diabetes_normalized["Outcome"]

# Split the dataframe into a training set and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=123, test_size=0.2
)

# Convert data to numpy arrays to use in the neural network
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# self.weights = (np.random.randint(0, 10, size=(len(input_vector))) / 100) + 0.001


class Neural_Network:
    def __init__(self, num_nodes_in, learning_rate):
        self.weights = np.array(
            (np.random.randint(0, 10, size=(num_nodes_in)) / 100) + 0.001
        )
        self.bias = np.random.randn()
        self.num_nodes_in = num_nodes_in
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        """Function to calculate the sigmoid of the neural network."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Function to calculate the derivative of the sigmoid for calculating the MSE."""
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def predict(self, input_vector):
        """Function to predict the outcomes using the initial weights and bias."""
        layer1 = np.dot(input_vector, self.weights) + self.bias
        layer2 = self.sigmoid(layer1)
        for i in layer2:
            if i > 0.5:
                prediction = 1
            else:
                prediction = 0
        return prediction

    def compute_gradients(self, input_vector, target):
        """Function to compute vector gradients."""
        layer1 = np.dot(input_vector, self.weights) + self.bias
        layer2 = self.sigmoid(layer1)
        prediction = layer2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self.sigmoid_derivative(layer1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights

        return derror_dbias, derror_dweights

    def update_parameters(self, derror_dbias, derror_dweights):
        """Function to update the bias and weights."""
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    def train(self, input_vectors, targets, iterations):
        """Function to train the neural network."""
        cumulative_errors = []
        for current_iteration in range(iterations):

            # Pick a random data instance
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self.compute_gradients(input_vector, target)

            self.update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error of all instances
            if current_iteration % 100 == 0:
                cumulative_error = 0

                # To measure the error loop through all the instances
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)
        return cumulative_errors


# print(neural_net.train(train.columns[0:2], train.columns[8], 10000))

input_vectors = np.array(
    [[3, 1.5], [2, 1], [4, 1.5], [3, 4], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1]]
)

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])

neural_net = Neural_Network(8, 0.5)

training_error = neural_net.train(X_train, y_train, 10000)

print(neural_net.predict(X_train))

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("Cumulative_Error.png")
