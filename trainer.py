import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- 1. Eigene Bilder vorbereiten ---
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    './data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    './data',  # ← gleiche Quelle wie train_gen!
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.one_hot(0, 3)  # label "car" → Index 0



# --- 3. Daten kombinieren ---
# Wir nehmen einfach beide Generatoren/Datasets und trainieren nacheinander
# (oder man merged sie in tf.data.Dataset)

# --- 4. Modell definieren (komplett eigenes CNN) ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')  # 3 Klassen: car, chicken, cow
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 5. Training ---
from tqdm import tqdm
import numpy as np

epochs = 5
steps_per_epoch = len(train_gen)

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    progbar = tqdm(total=steps_per_epoch, desc="Training", unit="it")

    for step in range(steps_per_epoch):
        x_batch, y_batch = next(train_gen)  # ← HIER ist der Fix!
        loss, acc = model.train_on_batch(x_batch, y_batch)
        progbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")
        progbar.update(1)

    progbar.close()

    # Optional: Validierung nach jeder Epoche
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"Validation — Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")



# Speichern im HDF5-Format
model.save("chicken_car_cow_model.h5")

# Oder als TensorFlow SavedModel (Ordnerstruktur)
model.save("chicken_car_cow_model")

