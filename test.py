from PIL import Image
from PIL import ImageDraw
 
# Open an Image
img = Image.open('green_airplane/test/0000.png')
 
# Call draw Method to add 2D graphics in an image
I1 = ImageDraw.Draw(img)
 
# Add Text to an image
I1.text((28, 36), "nice Car", fill=(0, 0, 0))
 
# Save the edited image
img.save("car2.png")