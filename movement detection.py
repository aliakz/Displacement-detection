#importing libraries 
import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture('example.mp4')  # Replace with the path to your video file or 0 for webcam

# Read the first frame
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    # Read the current frame
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the current frame and the previous frame
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply a threshold to highlight the regions with significant differences 
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 300:  # Adjust the area threshold as needed
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + 2*w, y+2*h), (0, 255, 0), 3)

    # Display the result
    cv2.imshow("Movement Detection", frame2)

    # Update the previous frame
    gray1 = gray2.copy()

    # Exit on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()