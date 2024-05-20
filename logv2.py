import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, Label, Button, Entry, StringVar, Frame
from ultralytics import YOLO
from PIL import Image, ImageTk

# Define known width of a standard truck in cm
KNOWN_TRUCK_WIDTH = 250

def calculate_scale_from_truck(truck_box):
    pixel_width = truck_box[2] - truck_box[0]
    # Convert meter to cm by multiplying by 100
    return (pixel_width / KNOWN_TRUCK_WIDTH)

def draw_text(img, text, pos, color=(255, 255, 255), font_scale=0.5):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

def draw_reference_line(event, x, y, flags, param):
    global ref_points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        ref_points.append((x, y))
        cv2.line(frame, ref_points[0], ref_points[1], (0, 0, 255), 2)
        ask_for_length()

def ask_for_length():
    actual_length = float(entry_length.get())
    if actual_length:
        pixel_length = np.sqrt((ref_points[1][0] - ref_points[0][0])**2 + (ref_points[1][1] - ref_points[0][1])**2)
        scale_var.set(f"{pixel_length / actual_length} cm/px")
        draw_text(frame, f"Ref line: {pixel_length / actual_length:.2f} cm", (ref_points[0][0], ref_points[0][1] - 10), (0, 0, 255))

def use_truck_as_reference():
    if truck_detected:
        scale_var.set(f"{calculate_scale_from_truck(truck_box)} cm/px")

def clear_reference_measurement():
    scale_var.set('')
    logs_list.delete(*logs_list.get_children())
    log_count_var.set("Logs detected: 0")
    print("Reference measurement and logs data cleared.")

def update_frame():
    global frame
    ret, frame = cap.read()
    if ret:
        process_detections()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.imgtk = imgtk
        image_label.configure(image=imgtk)
    root.after(10, update_frame)



def process_detections():
    global frame, truck_detected, truck_box
    truck_detected = False
    truck_box = []
    # Predict using YOLO models
    truck_results = truck_model.predict(frame)
    log_results = log_model.predict(frame)
    logs_list.delete(*logs_list.get_children())  # Clear previous entries in the table

    # Process truck detections
    truck_positions = []
    for box in truck_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        truck_width_cm = (x2 - x1) / calculate_scale_from_truck([x1, y1, x2, y2])
        draw_text(frame, f"Truck: {truck_width_cm:.2f} cm", (x1, y1 - 10), (255, 0, 0))
        truck_positions.append((x1, y1, x2, y2))

    
    # Process log detections
    count = 0
    for box in log_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for tx1, ty1, tx2, ty2 in truck_positions:
            if abs(x1 - tx1) < 100 and abs(y1 - ty1) < 100:
                truck_detected = True
                truck_box = [tx1, ty1, tx2, ty2]
        if scale_var.get():
            scale = float(scale_var.get().split()[0])
            width_cm = (x2 - x1) / scale
            height_cm = (y2 - y1) / scale
            logs_list.insert("", "end", values=(f"Log {count + 1}", f"{width_cm:.2f} cm", f"{height_cm:.2f} cm"))
            
            draw_text(frame, f"{width_cm:.2f} x {height_cm:.2f} cm", (x1, y2 + 10), (0, 255, 0))
        count += 1
    log_count_var.set(f"Logs detected: {count}")

# Initialize models and webcam
log_model = YOLO('best.pt')
truck_model = YOLO('best_truck.pt')
cap = cv2.VideoCapture(0)

# Tkinter GUI setup
root = tk.Tk()
root.title("Log and Truck Detection System")
root.geometry("1280x720")

# Frames for layout
left_frame = Frame(root, width=640, height=360)
left_frame.grid(row=0, column=0, padx=10, pady=10)
right_frame = Frame(root)
right_frame.grid(row=0, column=1, padx=10, pady=10)

# Camera display label
image_label = Label(left_frame)
image_label.pack()

# Scale and length entry
scale_var = StringVar()
entry_length = Entry(right_frame, textvariable=scale_var)
entry_length.pack()

button_set_length = Button(right_frame, text="Set Reference Length", command=ask_for_length)
button_set_length.pack()

button_use_truck = Button(right_frame, text="Use Truck as Reference", command=use_truck_as_reference)
button_use_truck.pack()

button_clear = Button(right_frame, text="Clear Measurement", command=clear_reference_measurement)
button_clear.pack()

# Log count label
log_count_var = StringVar()
log_count_var.set("Logs detected: 0")
log_count_label = Label(right_frame, textvariable=log_count_var)
log_count_label.pack()

# Table for logs
logs_list = ttk.Treeview(right_frame, columns=("Log", "Width (cm)", "Height (cm)"), show="headings")
logs_list.heading("Log", text="Log")
logs_list.heading("Width (cm)", text="Width (cm)")
logs_list.heading("Height (cm)", text="Height (cm)")
logs_list.column("Log", width=100)  # Adjust column width
logs_list.column("Width (cm)", width=100)  # Adjust column width
logs_list.column("Height (cm)", width=100)  # Adjust column width
logs_list.pack(expand=True, fill='both')

# Start updating frames
root.after(0, update_frame)

# Start GUI
root.mainloop()
cap.release()
cv2.destroyAllWindows()
