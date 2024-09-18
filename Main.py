import pandas as pd
import numpy as np
# Load the CSV files
data = pd.read_csv("File Path")

# Display the first few rows
print(data.head())

# Check the shape of the dataset
print(f"Dataset shape: {data.shape}")

# Check the distribution of classes
print("Unique classes and their counts:")
print(data['label'].value_counts())

# Determine the number of unique classes
num_classes = data['label'].nunique()
print(f"Number of unique classes: {num_classes}")


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Separate features and labels
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Normalize the pixel values to be between 0 and 1
X = X / 255.0

# Reshape the data to match the input shape of CNN



X = X.reshape(-1, 28, 28, 1)

# One-hot encode the labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)








# Assuming 24 gesture classes instead of 25
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(24, activation='softmax')  # Updated to 24 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Re-run the training
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_val, y_val))

# Evaluate the model
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {val_acc}")


import matplotlib.pyplot as plt

# Predict on a sample image from the validation set
sample_image = X_val[0].reshape(1, 28, 28, 1)
prediction = model.predict(sample_image)
predicted_label = lb.inverse_transform(prediction)[0]

# Display the image with the predicted label
plt.imshow(X_val[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted Gesture: {predicted_label}")
plt.show()

#Output with hand Gesture Recognition

plt.imshow(X_val[5].reshape(28, 28), cmap='gray')

#Output with hand Gesture Recognition

 
