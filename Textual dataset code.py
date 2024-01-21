import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the EEG data into a pandas dataframe
df = pd.read_csv('E:/EEG_signal 100% Code/mental-state.csv')

# Split the data into features (X) and labels (y)
X = df.drop(['Label'], axis=1).values
y = df['Label'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert the labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Reshape the input data for the CNN
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)

# Define the model architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))

#model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Confusion matrix

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
from sklearn.metrics import confusion_matrix

y_test = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)
print(cm)

#print confusion matrix

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(test_accuracy)
plt.title(all_sample_title, size = 15);
plt.show()


# Calculate precision, recall, and F1-score
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1score = f1_score(y_test, y_pred, average='macro')

# Print precision, recall, and F1-score
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1score)

# model.save('model_name.h5')
# #Reshape the input data to have the same shape as X_train and X_test
# x = X_test[400].reshape(-1, X_train.shape[1], 1)

# #Make a prediction for the input data
# prediction = model.predict(x)

# #Get the index of the predicted label
# predicted_label = np.argmax(prediction)

# #Print the predicted label
# print(predicted_label)

# if predicted_label==0:
#     print("Normal Stress")
    
# elif predicted_label==1:
#     print("Medium Stress")
    
# elif predicted_label==2:
#     print("High Stress")

# import csv

# with open('mental-state.csv', 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     row = next(reader)
#     print(row)
# # Extract the features from row 4
features = df.iloc[0].values.reshape(1, -1, 1)

# Predict the output using the trained model
output = model.predict(features)

# Print the predicted label
predicted_label = np.argmax(output, axis=1)[0]
print('Predicted label:', predicted_label)

if predicted_label==0:
     print("Normal Stress")
    
elif predicted_label==1:
     print("Medium Stress")
    
elif predicted_label==2:
     print("High Stress")