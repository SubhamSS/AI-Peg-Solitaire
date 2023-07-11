import cv2
import os

# Define the path to the directory containing the images
image_dir = "C:/Personal_Data/VT SEM2/Human Robot Interaction/New Folder_2/New Folder/images/"

# Define the frame rate (number of frames per second) of the output video
# frame_rate = 2
#
# # Get the list of image filenames in the directory
# image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
#
# # Sort the image filenames alphabetically
# image_files.sort()
#
# # Get the first image to determine the size of the output video
# img = cv2.imread(image_files[0])
# height, width, channels = img.shape
#
# # Create a VideoWriter object to write the output video
# video_writer = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (width, height))
#
# # Loop through the image files and add them to the output video
# for image_file in image_files:
#     img = cv2.imread(image_file)
#     video_writer.write(img)
#
# # Release the VideoWriter object
# video_writer.release()
from PIL import Image

image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Sort the image filenames alphabetically
image_files.sort()

# Create a list of image objects from the files
images = [Image.open(os.path.join(image_dir, f)) for f in image_files]

# Save the list of images as an animated GIF
images[0].save('output.gif', save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)