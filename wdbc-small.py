import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

VERBOSE = 2

# specify path to training data and testing data

train_dataset_location = "wdbc_train_small.csv"
test_dataset_location = "wdbc.csv"


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
my_batch_size = 550 # Batch size changed
my_epochs = 750 #
my_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
my_shuffle_each_epoch = True

# activation in layers
# number of hidden-layer-nodes
# add/remove layers

# Model must be sequential. This should not be changed.
# Last layer must be linear. This should not be changed.
########################################################

# define the model

my_model = tf.keras.Sequential()

#############layer 1
my_model.add(tf.keras.layers.Dense(80,kernel_initializer='uniform'))
my_model.add(tf.keras.layers.BatchNormalization()) ## to normalize nonlinear data
my_model.add(tf.keras.layers.Activation("linear"))


##Adding dropout to regularize
my_model.add(tf.keras.layers.Dropout(0.50))


#############layer 2
my_model.add(tf.keras.layers.Dense(50, kernel_initializer='uniform'))
my_model.add(tf.keras.layers.BatchNormalization()) ## to normalize nonlinear data
my_model.add(tf.keras.layers.Activation("linear"))

##Adding dropout to regularize
my_model.add(tf.keras.layers.Dropout(0.40))

############layer 3
my_model.add(tf.keras.layers.Dense(40,kernel_initializer='uniform'))
my_model.add(tf.keras.layers.BatchNormalization()) ## to normalize nonlinear data
my_model.add(tf.keras.layers.Activation("linear"))


##Adding dropout to regularize
my_model.add(tf.keras.layers.Dropout(0.20))

##########layer 4
my_model.add(tf.keras.layers.Dense(30, kernel_initializer='uniform', activation="linear"))
my_model.add(tf.keras.layers.BatchNormalization()) ## to normalize nonlinear data
my_model.add(tf.keras.layers.Activation("linear"))

##Adding dropout to regularize
my_model.add(tf.keras.layers.Dropout(0.10))

##########layer 5
my_model.add(tf.keras.layers.Dense(10, kernel_initializer='uniform', activation="linear"))
my_model.add(tf.keras.layers.BatchNormalization()) ## to normalize nonlinear data
my_model.add(tf.keras.layers.Activation("linear"))


#dense layers with regularization
my_model.add(
    tf.keras.layers.Dense(
        5,
        activation="linear",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        bias_regularizer=tf.keras.regularizers.l2(0.01)
        )
    )
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

# Defining loss to include logits. reduced Nan for loss
def my_model_loss(y_true, y_pred):
    return tf.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

# compile the model
my_model.compile(optimizer=my_optimizer, loss=my_model_loss,
             metrics=['accuracy'])

#compile the model
#my_model.compile(optimizer=my_optimizer, loss='categorical_crossentropy',
#                 metrics=['accuracy'])

# train the model
my_model.fit(x=X_train, y=y_train, batch_size=my_batch_size, epochs=my_epochs,
             verbose=VERBOSE, shuffle=my_shuffle_each_epoch)

## evaluate the model

print("evaluation on training data",
      my_model.evaluate(x=X_train, y=y_train, batch_size=my_batch_size))

# theere should not be any change below this point
# read the testing dataset and print its accuracy
(X_test, y_test) = read_csv(test_dataset_location)
print("evaluation on test data",
      my_model.evaluate(x=X_test, y=y_test, batch_size=my_batch_size))

