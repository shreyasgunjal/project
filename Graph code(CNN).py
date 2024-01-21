import os
import scipy.io
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten

# Define paths to calm and stress folders
calm_folder = 'E:/EEG_signal 100% Code/EEG_signal 100% Code/Database/Calm/'
stress_folder = 'E:/EEG_signal 100% Code/EEG_signal 100% Code/Database/Stress/'

# Load calm files
calm_X = []
for i, filename in enumerate(os.listdir(calm_folder)):
    if filename.endswith(".mat") and i < 16:
        file_path = os.path.join(calm_folder, filename)
        data = scipy.io.loadmat(file_path)
        X = data['data']
        X = np.transpose(X)  # Transpose to get data in shape (time_steps, features)
        calm_X.append(X)
calm_X = np.concatenate(calm_X, axis=0)
calm_y = np.zeros(calm_X.shape[0])  # Label as 0 (calm)

# Load stress files
stress_X = []
for i, filename in enumerate(os.listdir(stress_folder)):
    if filename.endswith(".mat") and i < 16:
        file_path = os.path.join(stress_folder, filename)
        data = scipy.io.loadmat(file_path)
        X = data['data']
        X = np.transpose(X)  # Transpose to get data in shape (time_steps, features)
        stress_X.append(X)
stress_X = np.concatenate(stress_X, axis=0)
stress_y = np.ones(stress_X.shape[0])  # Label as 1 (stress)

# Concatenate and shuffle data
X = np.concatenate([calm_X, stress_X], axis=0)
y = np.concatenate([calm_y, stress_y], axis=0)
indices = np.random.permutation(X.shape[0])
X = X[indices]
y = y[indices]

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets
split = int(0.8 * X.shape[0])
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# Reshape data for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile and fit model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1, batch_size=32)

# Evaluate model on test data
loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Save the trained model
# model.save('my_model.h5')

import scipy.io
import numpy as np
from tensorflow.keras.models import load_model

# Define the function to apply preprocessing steps to the data
def apply_preprocessing(data):
    # Define the preprocessing steps to apply to the data
    # For example:
    # data = preprocess_data(data)
    # data = apply_filter(data)
    # data = apply_feature_extraction(data)
    return data

# Load the saved model
# model = load_model('my_model.h5')

# Prepare the new input data
calm_file = 'E:/EEG_signal 100% Code/EEG_signal 100% Code/Database/Stress/s02_60.mat'
new_data = scipy.io.loadmat(calm_file)

new_data = new_data['data']
new_data = new_data.reshape(-1, 40)  # reshape the new data to have shape (num_samples, num_features)

# Apply preprocessing steps to the new data
new_data = apply_preprocessing(new_data)

# Scale the new data using the same scaler used for training
new_data = scaler.transform(new_data)

num_samples = new_data.shape[0]
num_features = new_data.shape[1]

new_data = new_data.reshape((num_samples, num_features, 1))  # reshape back to original shape (num_samples, num_features, 1) for input to the model

# Make predictions on the new data
y_pred = model.predict(new_data)

# Convert the predicted probabilities to class labels
y_pred_labels = (y_pred > 0.5).astype('int32')

# Print whether the input graph has high or low stress
if y_pred_labels[0] == 0:
    print('The input graph has Low stress.')
else:
    print('The input graph has High stress.')

