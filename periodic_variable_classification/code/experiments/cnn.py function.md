# cnn.py function

    NOTE: while running the function the current working directory should be ../name(one of the arguments)/code/ 
          and the dmdt processed data should be in ../name(one of the arguments)/data/data_folder(one of the arguments)/
          containing X_2d.npy which is a 3D matrix containing dmdts with dimensions as (#dmdts, height of dmdt, width of dmdt),
          X_features.npy which is a 2D matrix with dimensions (#dmdts, #features) and y.npy containing dmdt labels 
          corresponding to X_2D.npy with dimension (#dmdts,)      
    
    Arguments:
      
    name: denotes the parent directory for which cnn is to be trained eg: ensemble, cnn_with, cnn_without, gdr21, periodic, 
          trans, ptf_classifier
    
    cnn_layers: denotes the list of layers making a CNN. Refer: https://lasagne.readthedocs.io/en/latest/modules/layers.html
                for different layers which can be used
        eg:         
           [
                (InputLayer, {'name':'input', 'shape': (None, X_train.shape[1], X_train.shape[2], X_train.shape[3])}),

                (Conv2DLayer, {'name':'conv2d1', 'num_filters': 64, 'filter_size': (5, 5), 'pad': 0, 'nonlinearity':rectify}),
                (MaxPool2DLayer, {'name':'maxpool1','pool_size': (2, 2)}),
                (DropoutLayer, {'name':'dropout1','p':0.1}),

                #(Conv2DLayer, {'name':'conv2d2','num_filters': 128, 'filter_size': (5, 5), 'pad': 2, 'nonlinearity':rectify}),
                #(MaxPool2DLayer, {'pool_size': (2, 2)}),
                #(DropoutLayer, {'p':0.3}),         

                #(Conv2DLayer, {'name':'conv2d3','num_filters': 256, 'filter_size': (5, 5), 'pad': 2, 'nonlinearity':rectify}),
                #(MaxPool2DLayer, {'pool_size': (2, 2)}),
                #(DropoutLayer, {'p':0.5}),       

                #(DenseLayer, {'name':'dense1','num_units': 512}),
                #(DropoutLayer, {'name':'dropout2','p':0.5}),
                #(DenseLayer, {'name':'dense2','num_units': 512}),

                (DenseLayer, {'name':'output','num_units': len(list(set(y))), 'nonlinearity': softmax}),
          ]
        
    classes: a list denoting the class numbers to be used for training the CNN eg: for ensemble CNN, 
             classes = [1,2,3,4,5,6,7,9,10,11,13,18]
    
    data_folder: refer to the 'name' argument details
    
    epochs, update_func, learning_rate, objective_l2, train_split_eval_size, verbose: denotes NeuralNet parameters
    
    output_folder: denotes the name of the directory in which the results of trained CNN will be saved, most of the time it 
                   will be similar to name argument
    
    oversampling, undersampling: equal to 1 for oversampling or undersampling respectively the training data, else 0
    
    oversampling_ratio: Refer ratio argument in 
                        http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html
    
    undersampling_ratio: Refer ratio argument in 
                       http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.under_sampling.RandomUnderSampler.html
     
