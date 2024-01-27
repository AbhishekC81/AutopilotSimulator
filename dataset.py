import cv2
import math

# points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

xs = []
ys = []

# read data.txt
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        # steering wheel angle in radians
        ys.append(float(line.split()[1]) * math.pi / 180)


# get number of images
num_images = len(xs)


train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)


def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        image_path = train_xs[(train_batch_pointer + i) % num_train_images]
        image = cv2.imread(image_path)
        image = cv2.resize(image[-150:], (200, 66)) / 255.0
        x_out.append(image)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        image_path = val_xs[(val_batch_pointer + i) % num_val_images]
        image = cv2.imread(image_path)
        image = cv2.resize(image[-150:], (200, 66)) / 255.0
        x_out.append(image)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out

