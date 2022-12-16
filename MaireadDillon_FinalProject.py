"""Creating a neural network to predict whether or not a patient has diabetes using a variety of predictor variables."""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Read csv
diabetes = pd.read_csv(
    "/Users/maireaddillon/Documents/Duke/BIOS-823/BIOS823-FinalProject/diabetes.csv"
)

# Split variables into predictors (X) and outcome (y)
predictor_vars = diabetes[
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
y = diabetes["Outcome"]

# Create normalization function
def normalize(df):
    """Create a function to normalize the predictor variables."""
    result = df.copy()
    for col in df.columns:
        max_value = df[col].max()
        min_value = df[col].min()
        result[col] = (df[col] - min_value) / (max_value - min_value)
    return result


# Save normalized diabetes predictors dataframe
X = normalize(predictor_vars)

# Split the dataframe into a training set and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=123, test_size=0.2
)

# Convert data to numpy arrays to use in the neural network
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Create Neural Network class
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
        """Function to predict the outcomes using the  weights and bias."""
        layer1 = np.dot(input_vector, self.weights) + self.bias
        layer2 = self.sigmoid(layer1)
        prediction = layer2
        return prediction

    def compute_gradients(self, input_vector, target):
        """Function to compute vector gradients."""
        layer1 = np.dot(input_vector, self.weights) + self.bias
        layer2 = self.sigmoid(layer1)
        prediction = layer2

        # Take the derivative of the error with respect to the prediction
        derror_dprediction = 2 * (prediction - target)
        # Take the derivative of the prediction with respect to layer 1
        dprediction_dlayer1 = self.sigmoid_derivative(layer1)
        # Take the derivative of layer 1 with respect to the bias
        dlayer1_dbias = 1
        # Take the derivative of layer 1 with respect to the initial weights
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        # Take the derivative of the error with respect to the bias
        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        # Take the derivative of the error with respect to the weights
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights

        return derror_dbias, derror_dweights

    def update_parameters(self, derror_dbias, derror_dweights):
        """Function to update the bias and weights using backpropagation."""
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


# Run neural network for diabetes data
neural_net = Neural_Network(8, 0.1)

# Find training error
training_error = neural_net.train(X_train, y_train, 10000)

# Predict using test data
predicted = neural_net.predict(X_test)

# Create empty array to input predictions
predictions = []

# Use probabilities to predict outcome as 0s and 1s
for i in predicted:
    if i >= 0.5:
        predictions.append(1)
    else:
        predictions.append(0)

# Find and print confusion matrix
confusion_mat = confusion_matrix(y_test, predictions)
print(confusion_mat)

# Find and print accuracy
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

# Plot training error
plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.title("Training Error for Diabetes Data")
plt.savefig("Cumulative_Error.png")

# Import new dataset
cancer = pd.read_csv(
    "/Users/maireaddillon/Documents/Duke/BIOS-823/BIOS823-FinalProject/breast-cancer.csv"
)

# Drop the ID column
cancer1 = cancer.drop(["id"], axis=1)

# Create predictors data frame using mean values for different measurements
X1 = cancer1[
    [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
    ]
]

# Change outcome to binary and create outcome vector
y1 = pd.get_dummies(cancer1["diagnosis"])

# Remove benign column from y1 so that M=1 and B=0
y1 = y1.drop(["B"], axis=1)

# Normalize
X1_normalized = normalize(X1)

# Split cancer data into training and testing set
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1_normalized, y1, random_state=123, test_size=0.2
)

# Convert data to numpy arrays to use in the neural network
X_train1 = X_train1.to_numpy()
X_test1 = X_test1.to_numpy()
y_train1 = y_train1.to_numpy()
y_test1 = y_test1.to_numpy()

# Run neural network for cancer data
neural_net1 = Neural_Network(10, 0.1)

# Find training error
training_error1 = neural_net1.train(X_train1, y_train1, 10000)

# Predict on test dataset
predicted1 = neural_net1.predict(X_test1)

# Create empty array to input prediction values into
predictions1 = []

# Use probabilities to predict outcome as 0s and 1s
for i in predicted1:
    if i >= 0.5:
        predictions1.append(1)
    else:
        predictions1.append(0)

# Find and print confusion matrix
confusion_mat1 = confusion_matrix(y_test1, predictions1)
print(confusion_mat1)

# Find and print accuracy
accuracy1 = accuracy_score(y_test1, predictions1)
print(accuracy1)

# Plot and save graph of training error
plt.plot(training_error1)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.title("Training Error for Breast Cancer Data")
plt.savefig("Cumulative_Error1.png")
