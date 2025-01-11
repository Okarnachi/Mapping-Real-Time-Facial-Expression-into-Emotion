import os
import random
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Define the path to the test folder
test_folder_path = 'data/test'

# Define the emotion categories based on the folder names
emotion_categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to select random images from each emotion folder
def get_random_images_from_folders(folder_path, categories, total_images=16):
    selected_images = []
    num_images_per_category = total_images // len(categories)  # Divide total images among categories
    
    for category in categories:
        category_path = os.path.join(folder_path, category)
        all_images = os.listdir(category_path)
        
        # Select random images, limiting to what's available
        if len(all_images) > 0:
            selected_images_for_category = random.sample(all_images, min(num_images_per_category, len(all_images)))
            for img_name in selected_images_for_category:
                img_path = os.path.join(category_path, img_name)
                selected_images.append((img_path, category))  # Store image path and label
    
    return selected_images

# Function to display images in the Tkinter window
def display_images_in_tkinter(images, window, grid_size=(4, 4)):
    image_size = (150, 150)  # Resize images to fit the display
    images_per_row, rows = grid_size
    
    # Loop over each image, resize, and place it in the Tkinter window
    for idx, (img_path, category) in enumerate(images):
        # Open and resize the image
        img = Image.open(img_path)
        img = img.resize(image_size)  # Resize the image
        img_tk = ImageTk.PhotoImage(img)
        
        # Determine position on the grid
        row = idx // images_per_row
        col = idx % images_per_row
        
        # Create a label for the image
        image_label = Label(window, image=img_tk)
        image_label.image = img_tk  # Keep reference to avoid garbage collection
        image_label.grid(row=row * 2, column=col)  # Place image on grid
        
        # Create a label for the category (emotion)
        label = Label(window, text=category, font=("Arial", 12))
        label.grid(row=row * 2 + 1, column=col)  # Place label under image

# Main function
if __name__ == '__main__':
    # Initialize the Tkinter window
    root = tk.Tk()
    root.title("Random Emotion Images")
    
    # Select random images from each folder
    selected_images = get_random_images_from_folders(test_folder_path, emotion_categories, total_images=16)
    
    # Display the images in the Tkinter window
    display_images_in_tkinter(selected_images, root)
    
    # Run the Tkinter loop
    root.mainloop()
