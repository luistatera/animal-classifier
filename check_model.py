import tensorflow as tf
import numpy as np

# Load the model
model_path = "models/prj_best_resnet_model.keras"
model = tf.keras.models.load_model(model_path)

# Print model summary
print("\nModel Summary:")
model.summary()

# Print model layers
print("\nModel Layers:")
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} - {layer.__class__.__name__}")

# Try to get weights from the last layer
try:
    last_layer = model.layers[-1]
    weights = last_layer.get_weights()
    print(f"\nLast layer weights shape: {[w.shape for w in weights]}")
except Exception as e:
    print(f"Error getting weights: {e}") 