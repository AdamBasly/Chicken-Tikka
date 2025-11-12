import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# --- 1. Modell laden ---
model = tf.keras.models.load_model("chicken_car_cow_model.h5")

# --- 2. Bild vorbereiten ---
img_path = "./test/image.png"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# --- 3. Vorhersage ---
pred = model.predict(x)
classes = ["car", "chicken", "cow"]
print("Vorhersage:", classes[np.argmax(pred)])
