
# coding: utf-8

# In[ ]:


def cnn(name, cnn_layers, classes, epochs = 500, learning_rate = 0.0002, verbose = 1, seed = 0, test_size = 0.2, 
        data_folder = "all", oversampling = 0, undersampling = 0, oversampling_ratio, undersampling_ratio, 
        update_func = lasagne.updates.adam, objective_l2=0.0025, train_split_eval_size=0.05, output_folder):
    
    # NOTE: while running the function the current working directory should be ../name(one of the arguments)/code/ 
    #       and the dmdt processed data should be in ../name(one of the arguments)/data/data_folder(one of the arguments)/
    #       containing X_2d.npy which is a 3D matrix containing dmdts with dimensions as (#dmdts, height of dmdt, width of dmdt),
    #       X_features.npy which is a 2D matrix with dimensions (#dmdts, #features) and y.npy containing dmdt labels 
    #       corresponding to X_2D.npy with dimension (#dmdts,)      
    
    # Arguments:
        
    # name: denotes the parent directory for which cnn is to be trained eg: ensemble, cnn_with, cnn_without, gdr21, periodic, 
    #       trans, ptf_classifier
    
    # cnn_layers: denotes the list of layers making a CNN. Refer: https://lasagne.readthedocs.io/en/latest/modules/layers.html
    #             for different layers which can be used
    #     eg:         
    #        [
    #             (InputLayer, {'name':'input', 'shape': (None, X_train.shape[1], X_train.shape[2], X_train.shape[3])}),

    #             (Conv2DLayer, {'name':'conv2d1', 'num_filters': 64, 'filter_size': (5, 5), 'pad': 0, 'nonlinearity':rectify}),
    #             (MaxPool2DLayer, {'name':'maxpool1','pool_size': (2, 2)}),
    #             (DropoutLayer, {'name':'dropout1','p':0.1}),

    #             #(Conv2DLayer, {'name':'conv2d2','num_filters': 128, 'filter_size': (5, 5), 'pad': 2, 'nonlinearity':rectify}),
    #             #(MaxPool2DLayer, {'pool_size': (2, 2)}),
    #             #(DropoutLayer, {'p':0.3}),         

    #             #(Conv2DLayer, {'name':'conv2d3','num_filters': 256, 'filter_size': (5, 5), 'pad': 2, 'nonlinearity':rectify}),
    #             #(MaxPool2DLayer, {'pool_size': (2, 2)}),
    #             #(DropoutLayer, {'p':0.5}),       

    #             #(DenseLayer, {'name':'dense1','num_units': 512}),
    #             #(DropoutLayer, {'name':'dropout2','p':0.5}),
    #             #(DenseLayer, {'name':'dense2','num_units': 512}),

    #             (DenseLayer, {'name':'output','num_units': len(list(set(y))), 'nonlinearity': softmax}),
    #       ]
        
    # classes: a list denoting the class numbers to be used for training the CNN eg: for ensemble CNN, 
    #          classes = [1,2,3,4,5,6,7,9,10,11,13,18]
    
    # data_folder: refer to the 'name' argument details
    
    # epochs, update_func, learning_rate, objective_l2, train_split_eval_size, verbose: denotes NeuralNet parameters
    
    # output_folder: denotes the name of the directory in which the results of trained CNN will be saved, most of the time it 
    #                will be similar to name argument
    
    # oversampling, undersampling: equal to 1 for oversampling or undersampling respectively the training data, else 0
    
    # oversampling_ratio: Refer ratio argument in 
    #                     http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html
    
    # undersampling_ratio: Refer ratio argument in 
    #                      http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.under_sampling.RandomUnderSampler.html
     
        
        
    import matplotlib
    matplotlib.use('Agg')

    import os
    import copy
    import generate
    import numpy
    import theano
    import theano.gpuarray
    import pygpu
    from pygpu import gpuarray
    #gpuarray.use("gpu"+str(0))
    #import theano.sandbox.cuda
    #theano.sandbox.cuda.use("gpu"+str(0))
    #theano.gpuarray.use("gpu" + str(0))
    theano.gpuarray.use("cuda" + str(0))
    import lasagne
    from nolearn.lasagne import NeuralNet, objective, TrainSplit, visualize
    from lasagne.nonlinearities import softmax, rectify
    from lasagne.layers import InputLayer
    from lasagne.layers import Conv2DLayer
    from lasagne.layers import MaxPool2DLayer
    from lasagne.layers import DropoutLayer
    from lasagne.layers import DenseLayer
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from util import plot_confusion, plot_misclassifications
    import numpy as np
    from six.moves import cPickle
    import pickle
    from imblearn.over_sampling import SMOTE 
    from imblearn.under_sampling import TomekLinks, RandomUnderSampler


    # parameters
    #epochs = 500
    #learning_rate = 0.0002
    #verbose = 1
    #seed = 0
    #classes = [5,6]
    #what are classes

    #test_size = 0.2

    # get data and encode labels
    #X_2d, X_features, y, indices = generate.get_data("all", classes=classes, shuffle=True, seed=seed)
    X_2d, X_features, y, indices = generate.get_data(data_folder, classes=classes, shuffle=True, seed=seed)
    ##sm=SMOTE(random_state=seed)
    ##(f,g,h)=X_2d.shape
    ##X_2d,y=sm.fit_sample(X_2d.reshape(f,g*h),y)
    ##X_2d=X_2d.reshape((X_2d.shape[0],g,h))
    #print(scenario)
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(y)
    y = labelencoder.transform(y).astype(numpy.int32)
    print("Total number of instances: " + str(len(y)))

    # split data (train/test)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_2d, y, indices, test_size=test_size, random_state=seed)
    if oversampling==1:
        #sm=SMOTE(random_state=seed)
        sm=SMOTE(random_state=seed, ratio=oversampling_ratio) #ratio={2:1000,4:1000,5:1000})
        (f,g,h)=X_train.shape
        X_train,y_train=sm.fit_sample(X_train.reshape(f,g*h),y_train)
        X_train=X_train.reshape((X_train.shape[0],g,h))
    if undersampling==1:
        rus=RandomUnderSampler(random_state=seed, ratio=undersampling_ratio) #ratio={0:1000,1:1000,3:1000})
        (ff,gg,hh)=X_train.shape
        X_train,y_train=rus.fit_sample(X_train.reshape(ff,gg*hh),y_train)
        X_train=X_train.reshape((X_train.shape[0],gg,hh))

    #sm=SMOTE(random_state=seed)
    #X_train,y_train=sm.fit_sample(X_train,y_train)
    X_test_plot = copy.deepcopy(X_test)
    # why reshaping ?
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    print("Number of training instances: %i" % len(y_train))
    print("Number of test instances: %i" % len(y_test))

    layers = cnn_layers
#         [
#             (InputLayer, {'name':'input', 'shape': (None, X_train.shape[1], X_train.shape[2], X_train.shape[3])}),

#             (Conv2DLayer, {'name':'conv2d1', 'num_filters': 64, 'filter_size': (5, 5), 'pad': 0, 'nonlinearity':rectify}),
#             (MaxPool2DLayer, {'name':'maxpool1','pool_size': (2, 2)}),
#             (DropoutLayer, {'name':'dropout1','p':0.1}),

#             #(Conv2DLayer, {'name':'conv2d2','num_filters': 128, 'filter_size': (5, 5), 'pad': 2, 'nonlinearity':rectify}),
#             #(MaxPool2DLayer, {'pool_size': (2, 2)}),
#             #(DropoutLayer, {'p':0.3}),         

#             #(Conv2DLayer, {'name':'conv2d3','num_filters': 256, 'filter_size': (5, 5), 'pad': 2, 'nonlinearity':rectify}),
#             #(MaxPool2DLayer, {'pool_size': (2, 2)}),
#             #(DropoutLayer, {'p':0.5}),       

#             #(DenseLayer, {'name':'dense1','num_units': 512}),
#             #(DropoutLayer, {'name':'dropout2','p':0.5}),
#             #(DenseLayer, {'name':'dense2','num_units': 512}),

#             (DenseLayer, {'name':'output','num_units': len(list(set(y))), 'nonlinearity': softmax}),
#         ]

    net = NeuralNet(
        layers=layers,
        #layers=[('input', InputLayer),
        #        ('conv2d1', Conv2DLayer),
        #        ('maxpool1', MaxPool2DLayer),
        #        ('dropout1', DropoutLayer),
        #        ('conv2d2', Conv2DLayer),
        #	    ('conv2d3', Conv2DLayer),
        #        ('dense1', DenseLayer),
        #        ('dropout2', DropoutLayer),
        #	    ('dense2', DenseLayer),
        #        ('output', DenseLayer),
        #        ],
        max_epochs=epochs,
        update=update_func,
        update_learning_rate=learning_rate,
        objective_l2=objective_l2,
        train_split=TrainSplit(eval_size=train_split_eval_size),
        verbose=verbose,
    )

    net.fit(X_train, y_train)
    preds = net.predict(X_test)
    preds_proba = net.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy: %f" % acc)

    y_test = labelencoder.inverse_transform(y_test)
    preds = labelencoder.inverse_transform(preds)

    # plot misclassifications
    plot_misclassifications(y_test, preds, X_test_plot, indices_test, output_folder+"/misclassifications")

    # save output
    # os.mkdir("cnn_cd")
    numpy.save(output_folder+"/X_test",X_test)
    numpy.save(output_folder+"/y_test",y_test)
    numpy.save(output_folder+"/preds_proba",preds_proba)
    numpy.save(output_folder+"/preds",preds)
    numpy.savetxt(output_folder+"/y_test_cnn.csv", y_test, delimiter=",", fmt='%.4f')
    numpy.savetxt(output_folder+"/preds_cnn.csv", preds, delimiter=",", fmt='%.4f')
    numpy.savetxt(output_folder+"/preds_proba_cnn.csv", preds_proba, delimiter=",", fmt='%.4f')
    plot_confusion(y_test, preds, output_folder+"/confusion_cnn_hpercent.png")
    plt1=visualize.plot_conv_weights(net.layers_['conv2d1'])
    plt1.savefig(output_folder+"/filters1.png")
    plt1.close()
    plt2=visualize.plot_conv_weights(net.layers_['conv2d2'])
    plt2.savefig(output_folder+"/filters2.png")
    plt3=visualize.plot_conv_weights(net.layers_['conv2d3'])
    plt3.savefig(output_folder+"/filters3.png")
    plt3.close()

    f=open(output_folder+'/obj.save_cd','wb')
    cPickle.dump(net,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    #f=open('obj.save','rb')
    #net=cPickle.load(f)
    #f.close()

    #print(net)

    print("F1 Score: "+str(f1_score(y_test.reshape(y_test.shape[0]),preds.reshape(preds.shape[0]),average=None)))
    print("Matthews correlation coefficient (MCC): "+str(matthews_corrcoef(y_test.reshape(y_test.shape[0]),preds.reshape(preds.shape[0]))))

