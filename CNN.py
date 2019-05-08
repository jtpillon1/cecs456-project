def build(num_classes, input_size, learnrate, EPOCHS, pad):
     
    K.clear_session()
    model = Sequential()

    #Convolution Layer
    model.add(Conv2D(32, (3, 3), padding = pad, input_shape = (56,56,3)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    
    #A Second Convolutional Layer
    model.add(Conv2D(64, (3, 3), padding = pad))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    
    #Third Convolutional Layer
    model.add(Conv2D(128, (3, 3), padding = pad))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())

    #Fully Connected Layer
    model.add(Flatten())
    #Xavier initialization of the weights 
    model.add(Dense(units = 512, kernel_initializer = 'glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = num_classes))
    model.add(Activation('softmax'))
    
    #Compiling the CNN
    print('[INFO] compiling model...')
    opt = Adam(lr = learnrate, decay = learnrate / EPOCHS)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
    print(model.summary())
    return model

    
def fit_model(dataset, learnrate, EPOCHS, batch, height = 56, width = 56):

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(dataset + '/train', target_size = (height, width), batch_size = batch, class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory(dataset + '/test', target_size = (height, width), batch_size = batch, class_mode = 'categorical')
    
    #count(dataset + '/train')
    #num_classes = len(classes)
    #model = build(num_classes, learnrate, EPOCHS, input_size = (height, width))
    print('[INFO] training network')
    history = model.fit_generator(training_set, steps_per_epoch = number_training, epochs = EPOCHS, validation_data = test_set, validation_steps = number_tested)
    
    print("[INFO] serializing network...")

    #H = model.save(args["model"])
    
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig('AccuracyResult.png')
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig('LossResult.png')
    plt.close()
    