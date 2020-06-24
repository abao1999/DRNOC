# https://www.tensorflow.org/tutorials/quickstart/beginner
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# sequential model, very basic
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# model returns vector of logits or "log-odds" scores for each class
predictions = model(x_train[:1]).numpy()
print(predictions)

# use softmax function to convert logits to probabilities for each class
probs = tf.nn.softmax(predictions).numpy()
print(probs)

# use sparse categorical cross entropy loss to take vector of logits and a True index and return scalar loss for each example
# this loss is equal to negative log probability of the true class; it is zero if model is unsure
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

print("Evaluation: ")
model.evaluate(x_test, y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

print("First few probabilities: ")
print(probability_model(x_test[:5]))
