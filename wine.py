import tensorflow as tf

import numpy as np

tf.enable_eager_execution()

VERBOSE = 2

# specify path to training data and testing data

train_dataset_location = "wine_train.csv"
test_dataset_location = "wine.csv"

# csv format: feature_0, feature_1, .., feature_{n-1}, label
# n features followed by one label
# features are real, label is int. Label values: 0..NUM_CATEGORIES-1

def read_csv(filename, shuffle=False):
    tmp_matrix = np.loadtxt(open(filename), delimiter=",")
    if shuffle:
        np.random.shuffle(tmp_matrix)
    X = tmp_matrix[:,0:-1]
    y = tmp_matrix[:,-1]
    y = np.array(y).astype("float32")
    return(X, y)

########## some of the parameters that can be adjusted #############

my_batch_size = 20
my_epochs1 = 300
my_epochs2 = 600
my_epochs3 = 900
my_optimizer1 = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.8, beta2=0.998)
my_optimizer2 = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.99, beta2=0.999)
my_optimizer3 = tf.train.AdamOptimizer(learning_rate=0.0008, beta1=0.99, beta2=0.999)
my_shuffle_each_epoch = True

# activation in layers
# number of hidden-layer-nodes
# add/remove layers

# Model must be sequential. This should not be changed.
# Last layer must be linear. This should not be changed.
########################################################

# define the model

my_model = tf.keras.Sequential()

#############layer1
my_model.add(tf.keras.layers.Dense(7, kernel_initializer='uniform', activation='linear'))

#############Layer2
my_model.add(tf.keras.layers.Dense(6, kernel_initializer='uniform', activation='linear'))

#############Layer3
my_model.add(tf.keras.layers.Dense(3, kernel_initializer='uniform', activation='linear'))


def my_losses(y_true, y_pred):
  return tf.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

# last layer is linear, with number of nodes determined by output encoding
# in onehot encoding it is the number of categories
# add to the model once NUM_CATEGORIES is known
# my_model.add(tf.keras.layers.Dense(NUM_CATEGORIES))

# read the training dataset, determine NUM_CATEGORIES and add last layer
(X_train, y_train) = read_csv(train_dataset_location, shuffle=True)
NUM_CATEGORIES = y_train.max() + 1

X_train=tf.keras.utils.normalize(X_train, axis=1, order=2)
y_train=tf.keras.utils.to_categorical(y_train)

my_model.add(tf.keras.layers.Dense(NUM_CATEGORIES))

# compile the model

my_model.compile(optimizer=my_optimizer1, loss=my_losses,
                 metrics=['accuracy'])

# train the model

my_model.fit(x=X_train, y=y_train, batch_size=my_batch_size, epochs=my_epochs1,
             verbose=VERBOSE, shuffle=my_shuffle_each_epoch)

# compile the model

my_model.compile(optimizer=my_optimizer2, loss=my_losses,
                 metrics=['accuracy'])

# train the model

my_model.fit(x=X_train, y=y_train, batch_size=my_batch_size, epochs=my_epochs2,
             verbose=VERBOSE, shuffle=my_shuffle_each_epoch)

# compile the model

my_model.compile(optimizer=my_optimizer3, loss=my_losses,
                 metrics=['accuracy'])

# train the model

my_model.fit(x=X_train, y=y_train, batch_size=my_batch_size, epochs=my_epochs3,
             verbose=VERBOSE, shuffle=my_shuffle_each_epoch)

## evaluate the model

print("evaluation on training data",
      my_model.evaluate(x=X_train, y=y_train, batch_size=my_batch_size))

# theere should not be any change below this point
# read the testing dataset and print its accuracy
(X_test, y_test) = read_csv(test_dataset_location)
X_test=tf.keras.utils.normalize(X_test, axis=1, order=2)
y_test=tf.keras.utils.to_categorical(y_test)

print("evaluation on test data",
      my_model.evaluate(x=X_test, y=y_test, batch_size=my_batch_size))

