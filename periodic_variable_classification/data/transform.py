import os
import numpy
import pandas
from os import listdir
from os.path import isfile, join

import helpers
import statistics

# @Ashish: for debugging
#NO_PARALLEL = True
#MAX_INSTANCES = 100

# in case everythings works -> parallel processing
NO_PARALLEL = False
MAX_INSTANCES = None

def _get_data_path():

    #return os.path.join(os.getcwd().split('dmdt')[0], "dmdt/data")
    return os.getcwd()

def compute_features(data):

    # @ASHISH: FIXME, add feature extraction code here that takes 'data' as input and returns features (numpy array)

    data = numpy.array(data.tolist()).astype(numpy.float64)

    sdata = statistics._time_sort(data)
    meanmag = numpy.mean(sdata[1])
    minmag = numpy.min(sdata[1])
    maxmag = numpy.max(sdata[1])
    amplitude = statistics.amplitude(sdata)
    beyond1std = statistics.beyond1std(sdata)
    flux_percentile_ratio_mid20 = statistics.flux_percentile_ratio_mid20(sdata)
    flux_percentile_ratio_mid35 = statistics.flux_percentile_ratio_mid35(sdata)
    flux_percentile_ratio_mid50 = statistics.flux_percentile_ratio_mid50(sdata)
    flux_percentile_ratio_mid65 = statistics.flux_percentile_ratio_mid65(sdata)
    flux_percentile_ratio_mid80 = statistics.flux_percentile_ratio_mid80(sdata)
    linear_trend = statistics.linear_trend (sdata)
    max_slope = statistics.max_slope(sdata)
    median_absolute_deviation = statistics.median_absolute_deviation (sdata)
    median_buffer_range_percentage = statistics.median_buffer_range_percentage(sdata)
    pair_slope_trend = statistics.pair_slope_trend(sdata)
    percent_difference_flux_percentile = statistics.percent_difference_flux_percentile(sdata)
    skew = statistics.skew (sdata)
    small_kurtosis = statistics.small_kurtosis(sdata)
    std = statistics.std(sdata)
    stetson_j = statistics.stetson_j(sdata)
    stetson_k = statistics.stetson_k(sdata)
#    aov = statistics.aov(sdata)
#    percent_amplitude = statistics.percent_amplitude(sdata)
#    qso = statistics.qso(sdata)
#    ls = statistics.ls(sdata)

    return [\
    meanmag,\
    minmag,\
    maxmag,\
    amplitude,\
    beyond1std,\
    flux_percentile_ratio_mid20,\
    flux_percentile_ratio_mid35,\
    flux_percentile_ratio_mid50,\
    flux_percentile_ratio_mid65,\
    flux_percentile_ratio_mid80,\
    linear_trend,\
    max_slope,\
    median_absolute_deviation,\
    median_buffer_range_percentage,\
    pair_slope_trend,\
    percent_difference_flux_percentile,\
    skew,\
    small_kurtosis,\
    std,\
    stetson_j,\
    stetson_k,\
#   aov,\
#   percent_amplitude,\
#   qso,\
#   ls,\
    ]

def compute_2d_representation(data):

    (smjd,smag) = zip(*sorted(zip(data['mjd'],data['mag'])))
    dmdt2dar = helpers.dmdtim(smjd, smag, len(helpers.dmints), len(helpers.dtints))

    return dmdt2dar 

def _extract_images_features(args):

    try:

        class_id, classnum, path, data_aug = args

        # load data
        datafile = os.path.join(path, "indiv_lc", "class" + str(classnum), str(class_id) + '.csv')
#here!          datafile = os.path.join(os.getcwd(),'ptf',str(class_id)+'.txt')
        # array with entries of form
        #
        # id, smjd, smag, err(?)
        #('1109065028988', 53466.263201, 15.027325, 0.057909)
        # ('1109065028988', 53466.270722, 14.987169, 0.05767)
        # ('1109065028988', 53466.278261, 14.920083, 0.057312)
        # ('1109065028988', 53466.285811, 14.855363, 0.056981)
        if data_aug=="sample_1":
            data = numpy.genfromtxt(datafile, dtype=[('lcid','S13'),('mjd','f8'),('mag','f8'),('magerr','f8')], comments="#",delimiter=",")
            data['mag'] += numpy.multiply(numpy.random.normal(loc=0.0,scale=1.0,size=len(data['mag'])),data['magerr']/3)
        #data['mag'] += numpy.array(map(lambda x: numpy.random.normal(loc=0.0,scale=x/3),data['magerr']))
            images = compute_2d_representation(data)
            print ("1 augmented dmdt for %i class done"%classnum)
            return(images,None,classnum)



        if data_aug=="double":
            data = numpy.genfromtxt(datafile, dtype=[('lcid','S13'),('mjd','f8'),('mag','f8'),('magerr','f8')], comments="#",delimiter=",")
        # extract 2d representation
            images = compute_2d_representation(data)
        # extract features
            #features = compute_features(data)
            return ([images,images], [features,features], [classnum,classnum])

        elif data_aug=="vanilla":
            data = numpy.genfromtxt(datafile, dtype=[('lcid','S13'),('mjd','f8'),('mag','f8'),('magerr','f8')], comments="#",delimiter=",")
        # extract 2d representation
            images = compute_2d_representation(data)
        # extract features
            print ('one done    ')
            features = compute_features(data)
            #return(images,None,classnum)
            return(images,features,classnum,class_id)



    except:
        return (None, None, classnum)

def _get_all_lc(version=0, n_jobs=4,data_aug = "vanilla",classes=None,class_len=None):

    samples = [10000/i for i in class_len ]
    randoms = [10000%i for i in class_len ]
    ids = []
    counts = [0]*17
    for i in range(len(randoms)):
        ids.append(numpy.sort(numpy.random.choice(class_len[i], randoms[i],replace=False)))


    path = os.path.join(_get_data_path(), "v%i" % version)

    all_class_ids = numpy.loadtxt(os.path.join(path, 'classes.csv'), delimiter=" ", skiprows=1, dtype=numpy.int64)
#here!  all_class_ids = numpy.loadtxt(os.path.join(os.getcwd(), 'classes_ptf.txt'), delimiter=" ", skiprows=0, dtype=numpy.int64)
    all_class_counts = numpy.bincount(all_class_ids[:,1])

    params = []
    if data_aug=="sample_1":
        for i in range(len(all_class_ids)):
            print("Preparing instance %i of %i ..." % (i, all_class_ids.shape[0]))
            class_id, classnum = all_class_ids[i][0], all_class_ids[i][1]
            if classnum in classes:
                if samples[classes.index(classnum)]!=0:
                    for j in range(samples[classes.index(classnum)]-1):
                        params.append([class_id, classnum, path,data_aug])  
                if counts[classnum-1] in ids[classes.index(classnum)]:
                    params.append([class_id, classnum, path,data_aug])
                counts[classnum-1]+=1

    else:
        for i in range(len(all_class_ids)):
            print("Preparing instance %i of %i ..." % (i, all_class_ids.shape[0]))
            class_id, classnum = all_class_ids[i][0], all_class_ids[i][1]
            params.append([class_id, classnum, path,data_aug])
            
    print("Starting feature/2d extraction ...")

    results = helpers.perform_task_in_parallel(_extract_images_features, params, 
                                               n_jobs=n_jobs, 
                                               no_parallel=NO_PARALLEL, 
                                               max_instances=MAX_INSTANCES)


    X_2d, X_features, y, lcid = [], [], [], []

    if data_aug=="vanilla" or "sample_1":
        for i in range(len(results)):

            if results[i][0] is not None:
                X_2d.append(results[i][0])
                X_features.append(results[i][1])
                y.append(results[i][2])
                lcid.append(results[i][3])
            else:
                print("Could not parse features/representation for %s ..." % str(results[i][2]))
    elif data_aug=="double":
        for i in range(len(results)):

            if results[i][0] is not None:
                X_2d.append(results[i][0][0])
                X_features.append(results[i][1][0])
                y.append(results[i][2][0])

                X_2d.append(results[i][0][0][1])
                X_features.append(results[i][1][1])
                y.append(results[i][2][1])
            else:
                print("Could not parse features/representation for %s ..." % str(results[i][2]))

    X_2d = numpy.array(X_2d).astype(numpy.float32)
    X_features = numpy.array(X_features).astype(numpy.float32)
    y = numpy.array(y).astype(numpy.int32)
    lcid = numpy.array(lcid).astype(numpy.int64)

    return numpy.array(X_2d), numpy.array(X_features), numpy.array(y), numpy.array(lcid)    

def parse_dmdt_all(version=0,data_aug = "vanilla",classes=range(1,18),class_len=None):

    X_2d, X_features, y, lcid = _get_all_lc(version=version,data_aug = data_aug,classes=classes,class_len=class_len)

    d = os.path.join(_get_data_path(), "all")
    if not os.path.exists(d):
        os.makedirs(d)
    if data_aug=="sample_1":
        numpy.save(os.path.join(d, "X_2d_sample_2.npy"), X_2d)
        numpy.save(os.path.join(d, "X_features_sample_2.npy"), X_features)
        numpy.save(os.path.join(d, "y_sample_2.npy"), y)

    else:
        numpy.save(os.path.join(d, "X_2d.npy"), X_2d)
        numpy.save(os.path.join(d, "X_features.npy"), X_features)
        numpy.save(os.path.join(d, "y.npy"), y)
        numpy.save(os.path.join(d, "lcid.npy"), lcid)
##        numpy.save(os.path.join(d, "X_2d_ptf.npy"), X_2d)
##        numpy.save(os.path.join(d, "X_features_ptf.npy"), X_features)
##        numpy.save(os.path.join(d, "y_ptf.npy"), y)


    
if __name__ == "__main__":
    class_len = [30593, 4658, 279, 2420, 5433, 500, 72, 1514, 62, 124, 242, 7, 512, 142, 25, 85, 153]
    classes = [1,2,4,5,6,8,13]
    classes_ = {}
    class_len = [class_len[i-1] for i in classes]



    numpy.random.seed(0)
    #data = numpy.genfromtxt('v0/indiv_lc/class15/1009049027666.csv',comments="#",delimiter=",")
    #data = numpy.genfromtxt('v0/indiv_lc/class15/1009049027666.csv', dtype=[('lcid','S13'),('mjd','f8'),('mag','f8'),('magerr','f8')],comments="#",delimiter=",")
    #print compute_features(data)
    data_aug = input("Enter the type of data augmentation that you want ... Warning - previously generated data will be overwritten")
    parse_dmdt_all(data_aug=data_aug,classes=classes,class_len=class_len)
