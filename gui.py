import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load Model
model = load_model('models/cat_dog_model.h5')

# Create Window
root = tk.Tk()
root.title("Cat and Dog Identification")
root.geometry("500x500")

# Label
label = tk.Label(root, text="Upload an Image", font=("Arial", 18))
label.pack(pady=20)

# Image Display
img_label = tk.Label(root)
img_label.pack()

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=20)

# Prediction Function
def upload_image():
    file_path = filedialog.askopenfilename()

    if file_path:
        # Open Image
        img = Image.open(file_path)
        img = img.resize((200, 200))

        # Show Image
        photo = ImageTk.PhotoImage(img)
        img_label.config(image=photo)
        img_label.image = photo

        # Prepare Image for Prediction
        test_img = image.load_img(file_path, target_size=(150,150))
        img_array = image.img_to_array(test_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        prediction = model.predict(img_array)

        # Result
        if prediction[0][0] > 0.5:
            result_label.config(text="Dog Detected 🐶")
        else:
            result_label.config(text="Cat Detected 🐱")

# Upload Button
upload_btn = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_btn.pack(pady=20)

# Run GUI
root.mainloop()