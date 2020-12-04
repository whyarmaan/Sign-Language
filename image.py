# Capture the mouse click events in Python and OpenCV
'''
-> draw shape on any image 
-> reset shape on selection
-> crop the selection
run the code : python capture_events.py --image image_example.jpg
'''


# import the necessary packages
import argparse
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
ref_point = []
cropping = False

def shape_selection(event, x, y, flags, param):
  # grab references to the global variables
  global ref_point, cropping

  # if the left mouse button was clicked, record the starting
  # (x, y) coordinates and indicate that cropping is being
  # performed
  if event == cv2.EVENT_LBUTTONDOWN:
    ref_point = [(x, y)]
    cropping = True

  # check to see if the left mouse button was released
  elif event == cv2.EVENT_LBUTTONUP:
    # record the ending (x, y) coordinates and indicate that
    # the cropping operation is finished
    ref_point.append((x, y))
    cropping = False

    # draw a rectangle around the region of interest
    cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
    cv2.imshow("image", image)

capture = cv2.VideoCapture(0)
# keep looping until the 'q' key is pressed
while True:
  # load the image, clone it, and setup the mouse callback function
  ret, frame = capture.read()
  cv2.setMouseCallback("image", shape_selection)
  # display the image and wait for a keypress
  cv2.imshow("image", frame)
  key = cv2.waitKey(1) & 0xFF

  # if the 'c' key is pressed, break from the loop
  if key == ord("c"):
    break
  
  # if there are two reference points, then crop the region of interest
  # from teh image and display it
  if len(ref_point) == 2:
    crop_img = frame[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
    cv2.imshow("crop_img", crop_img)
    cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()
