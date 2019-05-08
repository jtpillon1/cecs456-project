def initvgg(train_dir, batch):
    K.clear_session()
    #Initialize the VGG model without the Fully Connected Layers, 
    #thus we can train the model again on our dataset
    global vgg16_model
    vgg16_model = VGG16(include_top = False, weights = 'imagenet',
                   input_shape = (224, 224, 3))
    
    height = 224
    width = 224

    #train_datagen = ImageDataGenerator(rescale = 1./255,
                                       #shear_range = 0.2,
                                       #zoom_range = 0.2,
                                       #horizontal_flip = True)
    
    #train_datagen =  ImageDataGenerator(preprocessing_function = preprocess_input, 
     #                               rotation_range = 90, horizontal_flip = True,
    #                              vertical_flip = True)
    
    #train_generator = train_datagen.flow_from_directory(train_dir, target_size = (height, width), batch_size = batch)
    

def build_transfer_model(vgg16_model, dropout, fc_layers, num_classes):
    for layer in vgg16_model.layers:
        layer.trainable = False

    x = vgg16_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    transfer_model = Model(inputs=vgg16_model.input, outputs=predictions)

    return transfer_model

def fit_transfer_model(dataset, EPOCHS, batch, FC_layers = [512], dropout = 0.5, learnrate = 0.00001):

    #count(dataset + '/train')
    EPOCHS = EPOCHS
#    transfer_model = build_transfer_model(vgg16_model, 
#                                      dropout=dropout, 
#                                      fc_layers=FC_layers, 
#                                      num_classes=len(classes))

    adam = Adam(lr=learnrate)
    transfer_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

    #filepath="./checkpoints/" + "ResNet50" + "_model_weights.h5"
    #checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
    #callbacks_list = [checkpoint]
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

    training_set = train_datagen.flow_from_directory(dataset + '/train', target_size = (224, 224), batch_size = batch, class_mode = 'categorical')
    
    test_datagen = ImageDataGenerator(rescale = 1./255)

    test_set = test_datagen.flow_from_directory(dataset + '/test', target_size = (224, 224), batch_size = batch, class_mode = 'categorical')

    
    print('[INFO] training network')
    history = transfer_model.fit_generator(training_set, epochs = EPOCHS, workers=8, steps_per_epoch = number_training // batch, validation_data = test_set, validation_steps = number_tested // batch, shuffle=True)
    
    print("[INFO] serializing network...")
    #global H
    #H = model.save(args["model"])
    
    
def evaluate_transfer_model():    
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
    
