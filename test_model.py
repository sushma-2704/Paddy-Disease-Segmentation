import tensorflow as tf
import numpy as np

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "train_images",
    image_size=(224, 224),
    batch_size=1,
    validation_split=0.2,
    subset="validation",
    seed=42,
    shuffle=False
)

class_names = validation_dataset.class_names

for images, labels in validation_dataset.take(1):
    image = images
    actual_index = int(labels[0])

model = tf.keras.models.load_model("paddy_disease_model.keras")
predictions = model.predict(image)
predicted_index = np.argmax(predictions[0])

print("Actual class:", class_names[actual_index])
print("Predicted class:", class_names[predicted_index])
print("Confidence:", round(float(predictions[0][predicted_index]) * 100, 2), "%")
