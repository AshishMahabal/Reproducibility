import numpy as np
import struct
import gzip
import multiprocessing
import matplotlib.pyplot as plt

dmints = [-8,-5,-3,-2.5,-2,-1.5,-1,-0.5,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.5,1,1.5,2,2.5,3,5,8]
dtints = [0,1.0/145,2.0/145,3.0/145,4.0/145,1.0/25,2.0/25,3.0/25,1.5,2.5,3.5,4.5,5.5,7,10,20,30,60,90,120,240,600,960,2000,4000]


def readlc(lc):
    """Read a lightcurve (currently in a specific format)
    
    Returns id,mjd,mag
    """
#    data = np.genfromtxt(lc,dtype=[('lcid','S13'),('mag','f8'),('magerr','f8'),
#                                   ('ra','f8'),('dec','f8'),('mjd','f8'),('duplicity','i8')],
#                         comments="#",delimiter=",")
    data = np.genfromtxt(lc,dtype=[('lcid','S13'),('mjd','f8'),('mag','f8'),('magerr','f8')],
                         comments="#",delimiter=",")
    (smjd,smag) = zip(*sorted(zip(data['mjd'],data['mag'])))
    return data['lcid'][0],list(smjd),list(smag)

def lcplot(id,mjd,mag):
    """Plot a light-curve
    
    Todo: plot points, not curves. 
    Plot error-bars.
    Check for upper limits?
    """
    plt.plot(mjd,mag)
    plt.axis=(np.min(mjd),np.max(mjd),np.max(mag),np.min(mag))
    plt.title(id)
    plt.xlabel("Mean Julian Date")
    plt.ylabel("Magnitude")
    #plt.legend(loc = 'upper left')
    plt.gca().invert_yaxis()

def dmdtim(mjd,mag,ldmints,ldtints):
    # ldmints is the LENGTH of dmints
    # ldtints is the LENGTH of dtints
    """Push each dmdt point to its appropriate bin
    
    The function could be made faster using some tricks.
    Also, the normalization needs to be thought of better.
    Right now we divide by number of points in the pairs.
    One obvious change to do is to divide by points in lc instead, 
    or even log of points or some such since twice the points are not two times better,
    especially for bif differences.
    
    The main question here is if two objects of the same class, one with a richer structure
    and another with a sparser set, pull algorithms apart.
    """
    dmdt=np.zeros(shape=(ldmints,ldtints))
    maxval = 255
    maxpts = len(mjd)*(len(mjd)-1)/2
    dmjd = []
    dmag = []

    # generate differences (w.r.t. time and mags)
    for i in range(len(mjd)):
        for j in range(i+1,len(mjd)):
            dmjd.append(mjd[j]-mjd[i])
            dmag.append(mag[j]-mag[i])
    
    # sort w.r.t. to first component (dmjd)
    (sdmjd,sdmag) = list(zip(*sorted(list(zip(dmjd,dmag)))))

    # sdmjd
    #(0.00096199999825330451, 0.00096299999859184027, 0.00097200000163866207, 0.0019249999968451448, 0.0019350000002305023, 0.0021459999989019707, 0.0021469999992405064, 0.002148999999917578, 0.0025590000004740432, 0.0025789999999688007)
    # sdmag
    #(-0.034694000000000003, 0.0077630000000006305, -0.023438000000000514, -0.026930999999999372, -0.015674999999999883, -0.012509999999998911, -0.019592000000001164, 0.02876299999999965, 0.03605400000000003, 0.039031999999998845)    

##    minmjdbin = 0
##    for i in range(len(sdmjd)):
##        mjdbin = minmjdbin
##        for k in range(minmjdbin,ldtints):
##            if sdmjd[i] > dtints[k]:
##                mjdbin = k
##        
##        minmjdbin = mjdbin
##        magbin = 0
##        for k in range(ldmints):
##            if sdmag[i] > dmints[k]:
##                magbin = k
##                
##        dmdt[magbin,mjdbin] += 1

    h=plt.hist2d(sdmjd,sdmag,bins=[dtints,dmints])
    dmdt=h[0]
    dmdt=np.transpose(dmdt)
    
    return (maxval*dmdt/maxpts+0.99999).astype(int)

def write_bin(ar2d):
    """Return binary formatted numbers from the 2d array
    
    These have to be prefixed by dims once for all images
    """
    far2d = ar2d.flatten()
    myfmt = '>' + 'B'*len(far2d)
    bin = struct.pack(myfmt,*far2d)
    return bin

def perform_task_in_parallel(task, params_parallel, n_jobs=2, no_parallel=False, max_instances=10):
    
    if not no_parallel:

        pool = multiprocessing.Pool(n_jobs)
        results = pool.map(task, params_parallel)

        pool.close()
        pool.join()
            
        return results
    else:
        results = []
        for i in range(min(max_instances,len(params_parallel))):
            results.append(task(params_parallel[i]))
        return results

