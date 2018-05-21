from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.callbacks import EarlyStopping 
import csv
 
def cnn(momentum, nb_epochs, learningRate, batchSize):
    X  = np.genfromtxt(open('tx.csv'), delimiter=',', max_rows=495)
    y_train = X[:,-1:] 
    X_train = X[:,:-1] 
  
   


    Y  = np.genfromtxt(open('rx.csv'), delimiter=',', max_rows=500) 
    y_test = Y[:,-1:]
    X_test = Y[:,:-1]
   
    Z = np.genfromtxt(open('tx.csv'), delimiter=',', max_rows=495)
    ZY_val = Z[:,-1:]
    ZX_val = Z[:,:-1]
    
    ZY_val_test = Z[:,-1:]
    ZX_val_test = Z[:,:-1]
   

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    ZX_val = ZX_val.reshape(ZX_val.shape[0], 1, 28, 28).astype('float32')
    #ZX_val_test = ZX_val_test.reshape(ZX_val_test.shape[0], 1, 28, 28).astype('float32')
 
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')  
    ZX_val = ZX_val.astype('float32')
    #ZX_val_test = ZX_val_test.astype('float32')
 
    X_train = X_train / 1000.
    X_test = X_test / 1000.
    ZX_val = ZX_val / 1000.
    #ZX_val_test = ZX_val_test / 1000.
   
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    ZY_val = np_utils.to_categorical(ZY_val)
    
   
    print('Train Samples: ', X_train.shape[0])
    print('Validation Samples: ', ZX_val.shape[0])
    print('Test Samples: ', X_test.shape[0])
    
   
    num_classes = 495
    print('num_classes = ', num_classes)
  
    def baseline_model():
        model = Sequential()
       
        #layer - 1
        model.add(Conv2D(128, (3, 3), input_shape=(1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
       
       
        model.add(Flatten())
       
        model.add(Dense(512, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
       
        sgd = optimizers.SGD(lr=learningRate, momentum=momentum)

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
       
        return model
   
    def plot_model_history(model_history):
        fig, axs = plt.subplots(1,2,figsize=(15,5))
        # summarize history for accuracy
        axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
        axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
        axs[0].legend(['train', 'val'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
        axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
        axs[1].legend(['train', 'val'], loc='best')
        #plt.show()
        plt.savefig('rxtx.png')
        
    def accuracy(test_x, test_y, model):
        result = model.predict(test_x)
        predicted_class = np.argmax(result, axis=1)
        true_class = np.argmax(test_y, axis=1)
        num_correct = np.sum(predicted_class == true_class)
        accuracy = float(num_correct)/result.shape[0]
        return (accuracy * 100)
    
   
    model = baseline_model()
   
    # define early stopping callback
#    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
#    callbacks_list = [earlystop]
#    model_info = model.fit(X_train, y_train, validation_data=(ZX_val, ZY_val), epochs=nb_epochs, batch_size=batchSize, verbose=1, callbacks=callbacks_list)
 
    model_info = model.fit(X_train, y_train, validation_data=(ZX_val, ZY_val), epochs=nb_epochs, batch_size=batchSize, verbose=1)
      
    #plot_model_history(model_info)
    
    # Final evaluation of the model
    scores = model.evaluate(ZX_val, ZY_val, verbose=1)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))


    ynew = model.predict_classes(X_test)
 
    print ynew[0]
   
    probabilities = model.predict(X_test) 
    
    for j in range(0,499):
     for p, i in sorted([(p, i) for i, p in enumerate(probabilities[0])], reverse=True):
      if ( i == ynew[j]):  
        print(i,p) 
     
        


   
 
    i = 0
    with open('rx.csv','r') as csvinput:
        with open('prediciton.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
            all = []
            for row in reader:
                row.append(ynew[i])
                i=i+1
                all.append(row)
    
            writer.writerows(all)
    

    
    


#
# Mini-Batch run
#
        
m = [0.9]
lr = [0.07]
bs = [100]
nb_epochs=40
 
for i in m:
    for j in lr:
        for k in bs:
            cnn(i, nb_epochs, j, k)

