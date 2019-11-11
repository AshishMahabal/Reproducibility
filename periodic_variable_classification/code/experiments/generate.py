import os,numpy

#from sklearn import preprocessing

def _get_data_path():
# have corrected this code on the copy on nirgun
    return os.path.join(os.getcwd().split("code")[0],"data")

def get_data(scenario, **kwargs):
    
    assert scenario in ["all"]
  # print(scenario)

    X_2d = numpy.load(os.path.join(_get_data_path(), str(scenario), "X_2d.npy"))
    X_features = numpy.load(os.path.join(_get_data_path(), str(scenario), "X_features.npy"))
    y = numpy.load(os.path.join(_get_data_path(), str(scenario), "y.npy"))
    indices = numpy.array(range(len(y)))

    if 'classes' in kwargs.keys():
	#only picking out the data corresponding the classes provided
        mask = numpy.in1d(y, numpy.array(kwargs['classes']))
        y = y[mask]
        X_2d = X_2d[mask]
        X_features = X_features[mask]
        indices = indices[mask]

    # shuffle data
    if 'shuffle' in kwargs.keys():
        seed = 0
        if 'seed' in kwargs.keys():
            seed = kwargs['seed']
        numpy.random.seed(seed)
        perm = numpy.random.permutation(len(y))
        y = y[perm]
        X_2d = X_2d[perm]
        X_features = X_features[perm]
        indices = indices[perm]
    
    return X_2d, X_features, y, indices
    
