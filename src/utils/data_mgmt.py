import tensorflow as tf

#Validation size is a paramter on our config.yaml. This is important so we can actually tune it in that file. This is the usefulness of yaml

def get_data(validation_datasize):
    #This function was copy and pasted from jupyter norebook. This is why jupyter notebook is a great start before scripting
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    # create a validation data set from the full training data 
    # Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
    X_valid, X_train = X_train_full[:validation_datasize] / 255., X_train_full[validation_datasize:] / 255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    # scale the test set as well
    X_test = X_test / 255.

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test) #Return an array