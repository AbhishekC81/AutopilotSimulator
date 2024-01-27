from tensorflow.keras.models import load_model
import cv2
import math
import numpy as np

# Restore the trained model

model = load_model("models/model.keras",safe_mode=False)

# Load the steering wheel image
img = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = img.shape

# Read data.txt
xs = []
ys = []

with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        ys.append(float(line.split()[1]) * math.pi / 180)

# Get the number of images
num_images = len(xs)

# Get the last 20% frames
i = math.ceil(num_images * 0.8)
print("Starting frame of video:" + str(i))


smooth_angle=0.0
while cv2.waitKey(10) != ord('q'):
    full_image = cv2.imread("driving_dataset/" + str(i) + ".jpg")
    print("Reading Image: ","driving_dataset/" + str(i) + ".jpg")
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0  # Resize and normalize the image
    degrees = model.predict(np.expand_dims(image, axis=0))[0][0] * 180.0 / math.pi

    print("Steering angle: {:.2f} (pred)\t{:.2f} (actual)".format(degrees, ys[i] * 180 / math.pi))

    # Display the full image
    cv2.imshow("frame", full_image)

    # Make smooth angle transitions by turning the steering wheel
    smooth_angle += 0.2 * pow(abs((degrees - smooth_angle)), 2.0 / 3.0) * (degrees - smooth_angle) / abs(degrees - smooth_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smooth_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

    i += 1

cv2.destroyAllWindows()
