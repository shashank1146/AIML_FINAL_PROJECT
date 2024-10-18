from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
import os

# Label mapping
label_map = {label:num for num, label in enumerate(actions)}

# Initialize sequences and labels
sequences, labels = [], []
expected_shape = (63,)  # Each frame should have 63 elements
sequence_length = 30    # Each sequence should have 30 frames

# Load the data
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Load each frame
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)

            # Check if res is empty or has an unexpected shape
            if res.size == 0 or res.shape != expected_shape:
                print(f"Skipping frame {frame_num} in sequence {sequence} for action {action}. Original shape: {res.shape} or empty")
                continue  # Skip this frame if it is empty or not of the expected shape

            window.append(res)

        # Ensure all sequences have exactly 30 frames by padding with zeros if necessary
        while len(window) < sequence_length:
            window.append(np.zeros(expected_shape))  # Pad with zero frames

        # Only add complete sequences to the main list
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

# Convert sequences and labels into NumPy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Verify the shape of X
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Set up logging directory
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Display model summary
model.summary()

# Save the model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
