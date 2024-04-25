import cv2
import numpy as np

# define mouse callback function to draw circle
def draw_rectangle(event, x, y, flags, param):
   global ix, iy, drawing, img
   if event == cv2.EVENT_LBUTTONDOWN:
      drawing = True
      ix = x
      iy = y
   elif event == cv2.EVENT_LBUTTONUP:
      drawing = False
      cv2.rectangle(img, (ix, iy),(x, y),(0, 255, 255),-1)

# Create a black image
img = np.zeros((512,700,3), np.uint8)

# Create a window and bind the function to window
cv2.namedWindow("Rectangle Window")

# Connect the mouse button to our callback function
cv2.setMouseCallback("Rectangle Window", draw_rectangle)

# display the window
while True:
   cv2.imshow("Rectangle Window", img)
   if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()