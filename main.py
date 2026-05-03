import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict
    results = model(frame, verbose=False)

    # Get classification result
    probs = results[0].probs

    if probs is not None:
        class_id = probs.top1
        confidence = probs.top1conf.item()
        class_name = model.names[class_id]

        label = f"{class_name} ({confidence:.2f})"

        # Draw label
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    # Show frame
    cv2.imshow("ISL Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()