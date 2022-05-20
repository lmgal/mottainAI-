from dataset import train_ds, test_ds
from model import model

# Parameters
epochs = 3

model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs
)
