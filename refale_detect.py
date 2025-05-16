import cv2
import os
import numpy as np

# Folder jisme images hain
folder = 'images'

# Image file names
image_files = ['refalejet1.jpeg', 'refalejet2.jpg', 'refalejet3.jpg']

# List to hold combined image rows
final_rows = []

for file in image_files:
    path = os.path.join(folder, file)
    img = cv2.imread(path)

    if img is None:
        print(f"❌ {file} nahi mila. Folder aur naam check karo.")
        continue

    print(f"✅ {file} load ho gaya!")

    # Resize image (chhoti size)
    img_resized = cv2.resize(img, (240, 160))

    # Grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Contour detection
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{file}: {len(contours)} contours found")  # Debug info

    # Draw bounding boxes for contours
    img_contours = img_resized.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Threshold kam kiya gaya hai
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Area display
            cv2.putText(img_contours, f'Area: {int(area)}', (x, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # X, Y coordinates display
            cv2.putText(img_contours, f'({x},{y})', (x, y + h + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    # Stack views side by side
    row = np.hstack((
        img_resized,
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
        img_contours
    ))

    final_rows.append(row)

# Combine all image rows vertically
if final_rows:
    final_display = np.vstack(final_rows)
    cv2.imshow("Rafale Jet - Original | Grayscale | Blurred | Edges | Contours", final_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❗ Koi image load nahi hui.")
