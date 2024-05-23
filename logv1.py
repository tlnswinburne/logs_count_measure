import cv2
import urllib.request
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from ultralytics import YOLO

def draw_line(event, x, y, flags, param):
    global ref_points, frame, scale, root
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        ref_points.append((x, y))
        cv2.line(frame, ref_points[0], ref_points[1], (0, 0, 255), 2)
        if not scale:  # Ask for the reference length only if scale is not set
            root.update()  # Update the main window
            actual_length = simpledialog.askfloat("Input", "Enter the actual length of the reference line in units:", parent=root)
            if actual_length:  # Check if the user entered a value
                pixel_length = np.sqrt((ref_points[1][0] - ref_points[0][0]) ** 2 + (ref_points[1][1] - ref_points[0][1]) ** 2)
                scale = pixel_length / actual_length

# Initialize variables
ref_points = []
scale = None
# IP address of the camera
url='http://192.168.68.110/'
# extension for quality of capture
##'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''

# Create a root window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Load the YOLO model
model = YOLO('best.pt')
cv2.namedWindow('Live Detection')
cv2.setMouseCallback('Live Detection', draw_line)

while True:
    # Initialize the webcam
    img_resp = urllib.request.urlopen(url+'cam-hi.jpg')
    imgnp = np.array(bytearray(img_resp.read()),dtype=np.uint8)
    frame = cv2.imdecode(imgnp,-1)

    if frame is None:
        print("could not read frame")
        break
    

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Predict using the YOLO model
    results = model.predict(rgb_frame)
    
    # Draw detections from results
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0].item()), int(box.xyxy[0][1].item()), int(box.xyxy[0][2].item()), int(box.xyxy[0][3].item())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 2)
            if scale:
                width_cm = (x2 - x1) / scale
                height_cm = (y2 - y1) / scale
                measurement = f"{width_cm:.2f} cm x {height_cm:.2f} cm"
                cv2.putText(frame, measurement, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 2)

    # Show the frame
    cv2.imshow('Live Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
root.destroy()  # Close the GUI