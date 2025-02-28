import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import activations

#gets dataset
#if training=True, returns training date
#else returns test data
def get_dataset(training=True):
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    if training == True:
        return (train_images, train_labels)
    else:
        return (test_images, test_labels)
    
#prints statistics of training data
#prints number of images, dimension of images, and frequency of each label      
def print_stats(train_images, train_labels):
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    print(train_images.shape[0])
    print(train_images.shape[1] , 'x', train_images.shape[2], sep="")

    numbers = []
    for i in range(0,10):
        numbers.append(0)

    for i in range(len(train_labels)):
        numbers[train_labels[i]] =  numbers[train_labels[i]] + 1
    
    for i in range(len(numbers)):
        print(i, '. ' , class_names[i],' - ', numbers[i], sep="")

#builds neural network model with Flatten and Dense layers   
#compiles and returns model
def build_model():
    model = keras.Sequential(
        [
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation=activations.relu),
            layers.Dense(64, activation=activations.relu),
            layers.Dense(10),
        ]
    )

    opt = keras.optimizers.SGD(learning_rate=0.001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
 
    model.compile(loss=loss_fn, optimizer=opt, metrics =['accuracy'])
    return model

#trains model
def train_model(model, train_images, train_labels, T):
    model.fit(x=train_images, y=train_labels, epochs=T)

#evaluates loss and accuracy of model with test data
def evaluate_model(model, test_images, test_labels, show_loss=True):
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=False)

    format_loss = "{:.4f}".format(loss)
    format_accuracy = "{:.2f}".format(accuracy * 100) 

    if show_loss == True:
        print('Loss: ', format_loss, sep="")
    
    print('Accuracy: ', format_accuracy, '%', sep="")

#percent predictions of the top three labels of a test image
# 
def predict_label(model, test_images, index):
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    
    #model.add(keras.layers.Softmax())

    #store the results of predict, and the values and try sorting
    predictions = model.predict(test_images)[index]

    sortedP = np.sort(predictions)
    sortedP = sortedP[::-1]

    #print(predictions)
    #print(sortedP)

    index = []
    for i in range(0,3):
        for j in range(len(predictions)):
            if predictions[j] == sortedP[i]:
                index.append(j)

    #print(index)

    for i in range(0,3):
        percent = "{:.2f}".format(sortedP[i] * 100) 
        print(class_names[index[i]], ': ', percent, '%', sep="")

def main():
    (train_images, train_labels) = get_dataset()

    (test_images, test_labels) = get_dataset(False)

    model = build_model()
 
    train_model(model, train_images, train_labels, 10)

    evaluate_model(model, test_images, test_labels)

    predict_label(model, test_images, 1)

if __name__=="__main__":
    main()
