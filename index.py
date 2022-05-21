from dataset import train_ds, test_ds
from model import model
from preprocess import preprocess

# Preprocess data
# uncomment only if images are added/removed from the original dataset
preprocess()

# Display model summary
model.summary()

# Train the model
epochs = 10
history = model.fit(train_ds, validation_data=test_ds, epochs=epochs)
