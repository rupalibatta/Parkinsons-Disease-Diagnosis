#training params
learning_rate = 0.0009
training_steps = 1000
batch_size = 200 #this is the k-fold cross validation
display_step = 100

num_classes = 1 #num of total classes (2 possible outcomes)

#network params
n_hidden = 512 #num of neurons

#use tf.data API to shuffle and batch the data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(6590).batch(batch_size).prefetch(1)

#store layers weight and bias

#random value gen to initialize weights
random_normal = tf.random_normal_initializer

weights = {
    'h': tf.Variable(tf.random.normal([num_features, n_hidden])),
    'out': tf.Variable(tf.random.normal([n_hidden, num_classes]))
}

biases = {
    'b': tf.Variable(tf.zeros([n_hidden])),
    'out': tf.Variable(tf.zeros([num_classes]))
}

#create model
def neural_net(inputData):
  #hidden fully connected layer w/ 512 neurons
  hidden_layer = tf.add(tf.matmul(inputData, weights ['h']), biases['b'])
  #apply sigmoid to hidden_layer output for non-linearity
  hidden_layer = tf.nn.sigmoid(hidden_layer)

  #output fully connected layer with a neuron for each class
  out_layer = tf.matmul(hidden_layer, weights['out'])+biases['out']
  #apply softmax to normalize the logits to a probability distribution
  return tf.nn.softmax(out_layer)

#define the loss function
def cross_entropy(y_pred, y_true):
  #one-hot format [0, 1] where first is -1 and second is 1
  y_true = tf.one_hot(y_true, depth=num_classes)
  #clip prediction values to avoid log(0) error
  y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
  #compute cross-entropy -> H(p,q) = p(x)*log(q(x))
  return keras.losses.binary_crossentropy(y_true, y_pred)

#set up the SGD optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate)

def run_optimization(x, y):
  #wrap computation inside a GradientTape for differentiation
  with tf.GradientTape() as g:
    pred = neural_net(x)
    loss = cross_entropy(pred, y)
  #variables to update (trainable)
  trainable_variables = list(weights.values()) + list(biases.values())
  #computer gradients
  gradients = g.gradient(loss, trainable_variables)
  #update w and b following gradients
  optimizer.apply_gradients(zip(gradients, trainable_variables))

#accuracy metric
def accuracy(y_pred, y_true):
  #predicted class is the index of highest score in prediction vector
  correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
  return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis = -1)

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
  #run the optimization to update W and b values
  run_optimization(batch_x, batch_y)
  if step % display_step == 0:
    pred = neural_net(batch_x)
    loss = cross_entropy(pred, batch_y)
    acc = accuracy(pred, batch_y)
    #print("Training the epoch: %i, Loss %f, Accuracy: %f" % (step, loss, acc))

#test model on validation set
pred = neural_net(x_test)
print("ANN Test Accuracy: %f" % accuracy(pred, y_test))
