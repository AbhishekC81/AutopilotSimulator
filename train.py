import os
import dataset
import model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np

LOGDIR = './logs'
CHECKPOINTDIR = './checkpoints'


# Build the model
model = model.buildModel()


# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

# Set up TensorBoard
tensorboard_callback = TensorBoard(log_dir=LOGDIR, histogram_freq=1, write_graph=True, write_images=True)

# Set up ModelCheckpoint to save the model after every epoch
checkpoint_callback = ModelCheckpoint(
    os.path.join(CHECKPOINTDIR, "model_checkpoint.h5"),
    save_best_only=False,  # Save only the best model based on validation loss
    save_weights_only=False,
    verbose=1
)

# Train the model
epochs = 30
batch_size = 100

for epoch in range(epochs):
    for i in range(int(dataset.num_images / batch_size)):
        xs, ys = dataset.LoadTrainBatch(batch_size)
        xs = np.array(xs)
        ys = np.array(ys)
        history = model.fit(xs, ys, epochs=1, batch_size=batch_size, validation_split=0.1, callbacks=[tensorboard_callback, checkpoint_callback])

        if i % 100 == 0:
            print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, history.history['loss'][0]))
            predictions = model.predict(xs)
            print("Predictions:", predictions)

model.save('models/model.keras')
print("Model saved to models/model.keras")
