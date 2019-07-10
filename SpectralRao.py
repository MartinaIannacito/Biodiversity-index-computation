#standard import
import sys

#math import
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import scipy
from scipy import spatial
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import itertools

#geo import
from osgeo import ogr
from osgeo import gdal
import rasterio
import gc
# use OGR specific exceptions
ogr.UseExceptions()


###progress bar function from Greenstick https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/34325723
""" Call in a loop to create terminal progress bar
     @params:
         iteration   - Required  : current iteration (Int)
         total       - Required  : total iterations (Int)
         prefix      - Optional  : prefix string (Str)
         suffix      - Optional  : suffix string (Str)
         decimals    - Optional  : positive number of decimals in percent complete (Int)
         length      - Optional  : character length of bar (Int)
         fill        - Optional  : bar fill character (Str)
    """
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def spectralrao (input=0, distance_m="euclidean", missing = -5, p=None, window = 3, mode="classic", Lambda=0, shannon=False, base= np.exp(1), Type = "Hill", rescale=False, NaTolerance=0.0, simplify=3, NcCores=1, ClusterType="MPI", debugging=False):
    mfactor = 1
    #### input check
    if isinstance(input, str):
        tif_ds = gdal.Open(input,gdal.GA_ReadOnly)
        rasterm = np.array(tif_ds.GetRasterBand(1).ReadAsArray())
    elif isinstance(input,list):
        rasterm = np.asanyarray(input[0])
    elif isinstance(input,np.matrix):
        rastrem = np.asarray(input)
    elif isinstance(input,np.ndarray):
        rasterm = input
    else:
        raise ValueError('The given input object is not valid')
    ### tol check
    if NaTolerance > 1:
        raise ValueError("na.tolerance must be in the [0-1] interval. Exiting...")
    if mode == "classic":
        isfloat = False # If data are float numbers, transform them in integer
        if len( rasterm[ rasterm != rasterm.astype(int)]) > 0:
            print("Converting input data in an integer matrix...")
            isfloat = True
            mfactor = 100**simplify
            rasterm = np.array(rasterm*mfactor, dtype = "int")
            print(rasterm.dtype)
    if mode=="classic" and shannon:
        print("Matrix check OK: \nRao and Shannon output matrices will be returned")
    elif mode=="classic" and not shannon:
        print("Matrix check OK: \nRao output matrix will be returned")
    elif mode =="multidimension" and shannon:
        raise ValueError("Matrix check failed: \nMultidimension and Shannon not compatible, set shannon=FALSE")
    elif mode=="multidimension" and not shannon:
        print("Matrix check OK: \nA matrix with multimension RaoQ will be returned")
    else:
        raise ValueError("Matrix check failed: \nNot a valid input | method | distance, please check all these options")   
    isfloat = False
    if NcCores > 1:
        if mode == "multidimension":
            print("Multi-core is not supported for multidimensional Rao, proceeding with 1 core...")
            NcCores = 1
        else:
            print("##################### Starting parallel calculation #######################")
    #
    ## Derive operational moving window
    #
    if window %2 == 1:
        w = int(( window - 1 ) / 2)
    else:
        raise ValueError("Moving window size must be an odd number.")
    #
    ## If mode is classic Rao
    #
    if mode == "classic" :
    #
    # If classic RaoQ is parallelized
    #
        if NcCores > 1 :
    #
    ## Preparation of output matrices
    #
            raoqe = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1])#.astype(np.float32)
            print(raoqe.dtype)
            raoqe = BorderRaoP(raoqe, rasterm, missing, w, distance_m, debugging, isfloat, mfactor, NcCores)
            raoqe = RaoP(raoqe, rasterm, missing, w, distance_m, debugging, isfloat, mfactor, NcCores)
        else:
            raoqe = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
            raoqe = BorderRaoS(raoqe, rasterm, missing, w, distance_m, debugging, isfloat, mfactor)
            raoqe = RaoS(raoqe, rasterm, missing, w, distance_m, debugging, isfloat, mfactor)

    elif mode == "multidimension":
        if debugging:
            print("#check: Into multidimensional clause.")
        ############
    if shannon:
        outS = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
        print("\nStarting Shannon-Wiener index calculation:\n")
        if NcCores == 1:    
            outS = BorderShannonS(outS, rasterm, missing, w, base, debugging, Type)
            outS = ShannonS(outS, rasterm, missing, w, base, debugging, Type)
        else:
            outS = BorderShannonP(outS, rasterm, missing, w, base, debugging, Type, NcCores)
            outS = ShannonP(out = outS, rasterm = rasterm, missing = missing, w = w, base = base, debugging = debugging, Type = Type, NcCores = NcCores)#)#, isfloat = False, mfactor = 0, 
        print(("\nCalculation of Shannon's index is completenot \n")) # End outS
    #
    ## Return multiple outputs
    #
    if debugging:
        print( "#check: return function.")
    if shannon and  mode == "classic":
        outl = dict()
        outl["RaoQ"] = raoqe
        outl["Shannon"] = outS 
    elif shannon and mode == "multidimension":
        outl = dict()
        outl["Multidimensional_RaoQ"] = raoqe
        outl["Shannon"] = outS
    elif not shannon and  mode == "classic":
        outl = dict()
        outl["RaoQ"] = raoqe
    elif not shannon and mode=="multidimension":
        outl = dict()
        outl["Multidimensional_RaoQ"] = raoqe
    return(outl)


def BorderRaoP(out, rasterm, missing, w, distance_m, debugging, isfloat, mfactor, NcCores):
    print("\nStarting Rao's index calculation on the border:\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: Rao single.")
    for  rw in range(0, w):
        for cl in range(0,w):
            d = dict()
            d["UL"] = np.unique(rasterm[0:(window-w+rw), 0:(window-w+cl)], return_counts=True)#UpLeft
            d["UR"] = np.unique(rasterm[0:(window-w+rw), (c+w-window-cl):c], return_counts=True)#UpRight
            d["DL"] = np.unique(rasterm[(r-window+w-rw):r, 0:(window-w+cl)], return_counts=True)#DownLeft
            d["DR"] = np.unique(rasterm[(r-window+w-rw):r, (c+w-window-cl):c], return_counts=True)#DownRight
            for key in d.keys():
                if len(np.where(d[key][0] == missing)[0]) != 0:
                    d[key]=list(d[key])
                    d[key][1] = np.delete( d[key][1], np.where(d[key][0] == missing))#frequences
                    d[key][0] = np.delete( d[key][0], np.where(d[key][0] == missing))#values
                if debugging:
                    print("Working on coords ",rw,",",cl,". classes len: ",len(d[key]),". window size=",window)
                if len(d[key][0]) > 1:
                    p = d[key][1]/np.sum(d[key][1])
                    p1 = np.zeros((len(d[key][0]),len(d[key][0])))
                    comb = np.array([x[0]*x[1] for x in list(itertools.combinations(p, 2))])
                    p1[ np.triu_indices(len(d[key][1]), k=1)] = comb
                    p1[ np.tril_indices(len(d[key][1]), k=-1)] = comb
                    d1 = spatial.distance.cdist(np.diag(d[key][0]), np.diag(d[key][0]), distance_m)
                    if isfloat:
                        d[key] = np.sum(np.multiply(p1,d1)) / mfactor
                    else:
                        d[key] = np.sum(np.multiply(p1,d1))                    
                elif len(d[key][0]) == 1:
                    d[key] = 0
                else:
                    d[key] = missing
                if key is "UL":
                    out[rw, cl] = d[key]
                elif key is "UR":
                    out[rw,c-cl-1] = d[key]
                elif key is "DL":
                    out[r-rw-1, cl] = d[key]
                else:
                    out[r-rw-1,c-cl-1] = d[key]
            del(d)
            gc.collect()
    print("\nComplete Rao's index calculation on the corners.\n")
    def UpRao(cl, c= c, rasterm = rasterm, missing = missing, window= window, w = w, distance_m = distance_m, debugging =  debugging, isfloat = isfloat, mfactor = mfactor):
        for  rw in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[0:(window-w+rw), cl-w:(w+cl+1)], return_counts=True)#UpLeftd["D"] = np.unique(rasterm[(r-window+w-rw):r, cl-w:(w+cl+1)], return_counts=True)#DownLeft
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) > 1:
                d1 = spatial.distance.cdist(np.diag(tw_labels), np.diag(tw_labels), distance_m)
                p = tw_values/np.sum(tw_values)
                p1 = np.zeros((len(tw_values),len(tw_values)))
                comb = np.array([x[0]*x[1] for x in list(itertools.combinations(p, 2))])
                p1[ np.triu_indices(len(tw_values), k=1)] = comb
                p1[ np.tril_indices(len(tw_values), k=-1)] = comb
                if isfloat:
                    return(np.sum(np.multiply(p1,d1))/mfactor)
                else:
                    return(np.sum(np.multiply(p1,d1)))
            elif len(tw_values) == 1:
                return(0)
            else:
                return(missing)
    def DownRao(cl, c= c, rasterm = rasterm, missing = missing, window= window, w = w, distance_m = distance_m, debugging =  debugging, isfloat = isfloat, mfactor = mfactor):
        for  rw in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[(r-window+w-rw):r, cl-w:(w+cl+1)], return_counts=True)#DownLeft#UpLeftd["D"] = 
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) > 1:
                d1 = spatial.distance.cdist(np.diag(tw_labels), np.diag(tw_labels), distance_m)
                p = tw_values/np.sum(tw_values)
                p1 = np.zeros((len(tw_values),len(tw_values)))
                comb = np.array([x[0]*x[1] for x in list(itertools.combinations(p, 2))])
                p1[ np.triu_indices(len(tw_values), k=1)] = comb
                p1[ np.tril_indices(len(tw_values), k=-1)] = comb
                if isfloat:
                    return(np.sum(np.multiply(p1,d1))/mfactor)
                else:
                    return(np.sum(np.multiply(p1,d1)))
            elif len(tw_values) == 1:
                return(0)
            else:
                return(missing)
    for  rw in range(0, w):  
        out[rw, w:(c-w)] = np.asarray(Parallel(n_jobs = NcCores)(delayed(UpRao)(cl) for cl in range(w,c-w)))
        out[r-rw-1, w:(c-w)] = np.asarray(Parallel(n_jobs = NcCores)(delayed(DownRao)(cl) for cl in range(w,c-w)))
        printProgressBar(rw + 1, w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check

    def LeftRao(rw, r = r, rasterm = rasterm, missing = missing, window= window, w = w, distance_m = distance_m, debugging =  debugging, isfloat = isfloat, mfactor = mfactor):
        for cl in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[ rw-w:(w+rw+1), 0:(window-w+cl)], return_counts=True)#Left
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) > 1:
                d1 = spatial.distance.cdist(np.diag(tw_labels), np.diag(tw_labels), distance_m)
                p = tw_values/np.sum(tw_values)
                p1 = np.zeros((len(tw_values),len(tw_values)))
                comb = np.array([x[0]*x[1] for x in list(itertools.combinations(p, 2))])
                p1[ np.triu_indices(len(tw_values), k=1)] = comb
                p1[ np.tril_indices(len(tw_values), k=-1)] = comb
                if isfloat:
                    return(np.sum(np.multiply(p1,d1))/mfactor)
                else:
                    return(np.sum(np.multiply(p1,d1)))
            elif len(tw_values) == 1:
                return(0)
            else:
                return(missing)
    def RightRao(rw, r = r, rasterm = rasterm, missing = missing, window= window, w = w, distance_m = distance_m, debugging =  debugging, isfloat = isfloat, mfactor = mfactor):
        for  cl in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[ rw-w:(w+rw+1), (c+w-window-cl):c], return_counts=True)#Right
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) > 1:
                d1 = spatial.distance.cdist(np.diag(tw_labels), np.diag(tw_labels), distance_m)
                p = tw_values/np.sum(tw_values)
                p1 = np.zeros((len(tw_values),len(tw_values)))
                comb = np.array([x[0]*x[1] for x in list(itertools.combinations(p, 2))])
                p1[ np.triu_indices(len(tw_values), k=1)] = comb
                p1[ np.tril_indices(len(tw_values), k=-1)] = comb
                if isfloat:
                    return(np.sum(np.multiply(p1,d1))/mfactor)
                else:
                    return(np.sum(np.multiply(p1,d1)))
            elif len(tw_values) == 1:
                return(0)
            else:
                return(missing)
    for cl in range(0,w):
        out[w:(r-w), cl] = np.asarray(Parallel(n_jobs = NcCores)(delayed(LeftRao)(rw) for rw in range(w,r-w)))
        out[w:(r-w), c-cl-1] = np.asarray(Parallel(n_jobs = NcCores)(delayed(RightRao)(rw) for rw in range(w,r-w)))
        printProgressBar(cl + 1, w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    print("\nComplete Rao's index calculation on the borders.\n") 
    return (out) 
    


def RaoP(out, rasterm, missing, w, distance_m, debugging, isfloat, mfactor, NcCores):
    print("\nStarting Rao's index calculation:\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    def raop(rw):
        def raout (cl, rw = rw, rasterm = rasterm, missing = missing, w = w, distance_m = distance_m, debugging = debugging, isfloat = isfloat, mfactor = mfactor):
            tw_labels, tw_values = np.unique(rasterm[(rw-w):(rw+w+1),(cl-w):(cl+w+1)], return_counts=True)
            if len(np.where(tw_labels == missing)) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes length: ",len(tw_values),". window size=",window)
            if len(tw_values) > 1:
                d1 = spatial.distance.cdist(np.diag(tw_labels), np.diag(tw_labels), distance_m)
                p = tw_values/np.sum(tw_values)
                p1 = np.zeros((len(tw_values),len(tw_values)))
                comb = np.array([x[0]*x[1] for x in list(itertools.combinations(p, 2))])
                p1[ np.triu_indices(len(tw_values), k=1)] = comb
                p1[ np.tril_indices(len(tw_values), k=-1)] = comb
                if isfloat:
                    return (np.sum(np.multiply(p1,d1))/mfactor)
                else:
                    return ((np.sum(np.multiply(p1,d1))))
            elif len(tw_values) == 1:
                return (((0)))
            else:
                return ((missing))
        printProgressBar(rw - w + 1, r - 2*w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
        Raout = Parallel(n_jobs = NcCores)(delayed(raout)(cl) for cl in range(w,c-w))
        return (Raout)
    out[w:(r-w), w:(c-w)] = (np.asarray(Parallel(n_jobs = NcCores)(delayed(raop)(rw) for rw in range(w,(r-w)))).reshape(r-2*w,c-2*w))#.astype(np.float32)
    print("\n\nCalculation of Rao's index complete.\n")
    return(out)

def BorderRaoS(out, rasterm, missing, w, distance_m, debugging, isfloat, mfactor):
    print("\nStarting Rao's index calculation on the border:\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: Rao single.")
    for  rw in range(0, w):
        for cl in range(0,w):
            d = dict()
            d["UL"] = np.unique(rasterm[0:(window-w+rw), 0:(window-w+cl)], return_counts=True)#UpLeft
            d["UR"] = np.unique(rasterm[0:(window-w+rw), (c+w-window-cl):c], return_counts=True)#UpRight
            d["DL"] = np.unique(rasterm[(r-window+w-rw):r, 0:(window-w+cl)], return_counts=True)#DownLeft
            d["DR"] = np.unique(rasterm[(r-window+w-rw):r, (c+w-window-cl):c], return_counts=True)#DownRight
            for key in d.keys():
                if len(np.where(d[key][0] == missing)[0]) != 0:
                    d[key]=list(d[key])
                    d[key][1] = np.delete( d[key][1], np.where(d[key][0] == missing))#frequences
                    d[key][0] = np.delete( d[key][0], np.where(d[key][0] == missing))#values
                if debugging:
                    print("Working on coords ",rw,",",cl,". classes len: ",len(d[key]),". window size=",window)
                if len(d[key][0]) > 1:
                    p = d[key][1]/np.sum(d[key][1])
                    p1 = np.zeros((len(d[key][0]),len(d[key][0])))
                    comb = np.array([x[0]*x[1] for x in list(itertools.combinations(p, 2))])
                    p1[ np.triu_indices(len(d[key][1]), k=1)] = comb
                    p1[ np.tril_indices(len(d[key][1]), k=-1)] = comb
                    d1 = spatial.distance.cdist(np.diag(d[key][0]), np.diag(d[key][0]), distance_m)
                    if isfloat:
                        d[key] = np.sum(np.multiply(p1,d1)) / mfactor
                    else:
                        d[key] = np.sum(np.multiply(p1,d1))                    
                elif len(d[key][0]) == 1:
                    d[key] = 0
                else:
                    d[key] = missing
                if key is "UL":
                    out[rw, cl] = d[key]
                elif key is "UR":
                    out[rw,c-cl-1] = d[key]
                elif key is "DL":
                    out[r-rw-1, cl] = d[key]
                else:
                    out[r-rw-1,c-cl-1] = d[key]
            del(d)
            gc.collect()
    print("\nComplete Rao's index calculation on the corners.\n") 
    for cl in range(w,c-w):
        for  rw in range(0, w):
            d = dict()
            d["U"] = np.unique(rasterm[0:(window-w+rw), cl-w:(w+cl+1)], return_counts=True)#UpLeft
            d["D"] = np.unique(rasterm[(r-window+w-rw):r, cl-w:(w+cl+1)], return_counts=True)#DownLeft
            for key in d.keys():
                if len(np.where(d[key][0] == missing)[0]) != 0:
                    d[key]=list(d[key])
                    d[key][1] = np.delete( d[key][1], np.where(d[key][0] == missing))#frequences
                    d[key][0] = np.delete( d[key][0], np.where(d[key][0] == missing))#values
                if debugging:
                    print("Working on coords ",rw,",",cl,". classes len: ",len(d[key]),". window size=",window)
                if len(d[key][0]) > 1:
                    p = d[key][1]/np.sum(d[key][1])
                    p1 = np.zeros((len(d[key][0]),len(d[key][0])))
                    comb = np.array([x[0]*x[1] for x in list(itertools.combinations(p, 2))])
                    p1[ np.triu_indices(len(d[key][1]), k=1)] = comb
                    p1[ np.tril_indices(len(d[key][1]), k=-1)] = comb
                    d1 = spatial.distance.cdist(np.diag(d[key][0]), np.diag(d[key][0]), distance_m)
                    if isfloat:
                        d[key] = np.sum(np.multiply(p1,d1)) / mfactor
                    else:
                        d[key] = np.sum(np.multiply(p1,d1))                    
                elif len(d[key][0]) == 1:
                    d[key] = 0
                else:
                    d[key] = missing
                if key is "U":
                    out[rw, cl] = d[key]
                else:
                    out[r-rw-1, cl] = d[key]
            del(d)
            gc.collect()
        printProgressBar(cl + 1, c - w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    for rw in range(w,r-w):
        for cl in range(0, w):
            d = dict()
            d["L"] = np.unique(rasterm[ rw-w:(w+rw+1), 0:(window-w+cl)], return_counts=True)#Left
            d["R"] = np.unique(rasterm[ rw-w:(w+rw+1), (c+w-window-cl):c], return_counts=True)#Right
            for key in d.keys():
                if len(np.where(d[key][0] == missing)[0]) != 0:
                    d[key]=list(d[key])
                    d[key][1] = np.delete( d[key][1], np.where(d[key][0] == missing))#frequences
                    d[key][0] = np.delete( d[key][0], np.where(d[key][0] == missing))#values
                if debugging:
                    print("Working on coords ",rw,",",cl,". classes len: ",len(d[key]),". window size=",window)
                if len(d[key][0]) > 1:
                    p = d[key][1]/np.sum(d[key][1])
                    p1 = np.zeros((len(d[key][0]),len(d[key][0])))
                    comb = np.array([x[0]*x[1] for x in list(itertools.combinations(p, 2))])
                    p1[ np.triu_indices(len(d[key][1]), k=1)] = comb
                    p1[ np.tril_indices(len(d[key][1]), k=-1)] = comb
                    d1 = spatial.distance.cdist(np.diag(d[key][0]), np.diag(d[key][0]), distance_m)
                    if isfloat:
                        d[key] = np.sum(np.multiply(p1,d1)) / mfactor
                    else:
                        d[key] = np.sum(np.multiply(p1,d1))                    
                elif len(d[key][0]) == 1:
                    d[key] = 0
                else:
                    d[key] = missing
                if key is "L":
                    out[rw, cl] = d[key]
                else:
                    out[rw, c-cl-1] = d[key]
            del(d)
            gc.collect()
        printProgressBar(rw + 1, r - w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    print("\nComplete Rao's index calculation on the borders.\n") 
    return (out)



def RaoS(out, rasterm, missing, w, distance_m, debugging, isfloat, mfactor):
    print("\nStarting Rao's index calculation:\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: Rao single.")
    for rw in range(w,r-w):
        for cl in range(w,c-w):
            tw_labels, tw_values = np.unique(rasterm[(rw-w):(rw+w+1),(cl-w):(cl+w+1)], return_counts=True)
            if len(np.where(tw_labels == missing)) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes length: ",len(tw_values),". window size=",window)
            if len(tw_values) > 1:
                d1 = spatial.distance.cdist(np.diag(tw_labels), np.diag(tw_labels), distance_m)
                p = tw_values/np.sum(tw_values)
                p1 = np.zeros((len(tw_values),len(tw_values)))
                comb = np.array([x[0]*x[1] for x in list(itertools.combinations(p, 2))])
                p1[ np.triu_indices(len(tw_values), k=1)] = comb
                p1[ np.tril_indices(len(tw_values), k=-1)] = comb
                if isfloat:
                    out[rw,cl] = np.sum(np.multiply(p1,d1)) / mfactor
                else:
                    out[rw,cl] = np.sum(np.multiply(p1,d1))
            elif len(tw_values) == 1:
                out[rw,cl] = 0
        printProgressBar(rw-w + 1, r-2*w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    print("\nCalculation of Rao's index complete.\n")
    return(out)

def BorderShannonS (out, rasterm, missing, w, base, debugging, Type):
    print("\nStarting Shannon-Wiener index calculation on the border:\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: Shannon single.")
    for  rw in range(0, w):
        for cl in range(0,w):
            d = dict()
            d["UL"] = np.unique(rasterm[0:(window-w+rw), 0:(window-w+cl)], return_counts=True)#UpLeft
            d["UR"] = np.unique(rasterm[0:(window-w+rw), (c+w-window-cl):c], return_counts=True)#UpRight
            d["DL"] = np.unique(rasterm[(r-window+w-rw):r, 0:(window-w+cl)], return_counts=True)#DownLeft
            d["DR"] = np.unique(rasterm[(r-window+w-rw):r, (c+w-window-cl):c], return_counts=True)#DownRight
            for key in d.keys():
                if len(np.where(d[key][0] == missing)[0]) != 0:
                    d[key] = np.delete( d[key][1], np.where(d[key][0] == missing))
                    #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
                else:
                    d[key] = d[key][1]
                if debugging:
                    print("Working on coords ",rw,",",cl,". classes len: ",len(d[key]),". window size=",window)
                if len(d[key]) != 0:
                    p = d[key]/np.sum(d[key])
                    if Type == "Rényi":
                        p_log = np.log(p)/ np.log(base)
                        if np.dot(p,p_log) == 0:
                            d[key] = 0
                        else:
                            d[key] = (-1)* np.dot(p,p_log)
                    else:
                        p_log = np.log(p)
                        d[key] = np.exp( (-1 )* np.dot(p,p_log))
                else:
                    d[key] = missing
                if key is "UL":
                    out[rw, cl] = d[key]
                elif key is "UR":
                    out[rw,c-cl-1] = d[key]
                elif key is "DL":
                    out[r-rw-1, cl] = d[key]
                else:
                    out[r-rw-1,c-cl-1] = d[key]
            del(d)
            gc.collect()
    print("\nComplete Shannon-Wiener index calculation on the corners.\n") 
    for cl in range(w,c-w):
        for  rw in range(0, w):
            d = dict()
            d["U"] = np.unique(rasterm[0:(window-w+rw), cl-w:(w+cl+1)], return_counts=True)#UpLeft
            d["D"] = np.unique(rasterm[(r-window+w-rw):r, cl-w:(w+cl+1)], return_counts=True)#DownLeft
            for key in d.keys():
                if len(np.where(d[key][0] == missing)[0]) != 0:
                    d[key] = np.delete( d[key][1], np.where(d[key][0] == missing))
                    #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
                else:
                    d[key] = d[key][1]
                if debugging:
                    print("Working on coords ",rw,",",cl,". classes len: ",len(d[key]),". window size=",window)
                if len(d[key]) != 0:
                    p = d[key]/np.sum(d[key])
                    if Type == "Rényi":
                        p_log = np.log(p)/ np.log(base)
                        if np.dot(p,p_log) == 0:
                            d[key] = 0
                        else:
                            d[key] = (-1)* np.dot(p,p_log)
                    else:
                        p_log = np.log(p)
                        d[key] = np.exp( (-1 )* np.dot(p,p_log))
                else:
                    d[key] = missing
                if key is "U":
                    out[rw, cl] = d[key]
                else:
                    out[r-rw-1, cl] = d[key]
            del(d)
            gc.collect()
        printProgressBar(cl + 1, c - w + 1, prefix = 'Progress:', suffix = 'Complete', length= 50)#check
    for rw in range(w,r-w):
        for cl in range(0, w):
            d = dict()
            d["L"] = np.unique(rasterm[ rw-w:(w+rw+1), 0:(window-w+cl)], return_counts=True)#Left
            d["R"] = np.unique(rasterm[ rw-w:(w+rw+1), (c+w-window-cl):c], return_counts=True)#Right
            for key in d.keys():
                if len(np.where(d[key][0] == missing)[0]) != 0:
                    d[key] = np.delete( d[key][1], np.where(d[key][0] == missing))
                    #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
                else:
                    d[key] = d[key][1]
                if debugging:
                    print("Working on coords ",rw,",",cl,". classes len: ",len(d[key]),". window size=",window)
                if len(d[key]) != 0:
                    p = d[key]/np.sum(d[key])
                    if Type == "Rényi":
                        p_log = np.log(p)/ np.log(base)
                        if np.dot(p,p_log) == 0:
                            d[key] = 0
                        else:
                            d[key] = (-1)* np.dot(p,p_log)
                    else:
                        p_log = np.log(p)
                        d[key] = np.exp( (-1 )* np.dot(p,p_log))
                else:
                    d[key] = missing
                if key is "L":
                    out[rw, cl] = d[key]
                else:
                    out[rw, c-cl-1] = d[key]
            del(d)
            gc.collect()
        printProgressBar(rw + 1, r - w + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    print("\nComplete Shannon-Wiener index calculation on the borders.\n") 
    return (out)

def ShannonS (out, rasterm, missing, w, base, debugging, Type):
    print("\nStarting Shannon-Wiener index calculation:\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: Shannon single.")
    #
    ## Loop over all the pixels
    #
    for rw in range(w, r - w):
        for cl in range(w, c - w):
            tw_labels, tw_values = np.unique(rasterm[(rw-w):(rw+w+1),(cl-w):(cl+w+1)], return_counts=True)
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) != 0:
                p = tw_values/np.sum(tw_values)
                if Type == "Rényi":
                    p_log = np.log(p)/ np.log(base)
                    if np.dot(p,p_log) == 0:
                        out[rw, cl] = 0
                    else:
                        out[rw, cl] = (-1)* np.dot(p,p_log)
                else:
                    p_log = np.log(p)
                    out[rw, cl] = np.exp( (-1 )* np.dot(p,p_log))
        printProgressBar(rw - w + 1, r - w + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    return(out)

def BorderShannonP (out, rasterm, missing, w, base, debugging, Type, NcCores):
    print("\nStarting Shannon-Wiener index calculation on the border:\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: Shannon single.")
    for  rw in range(0, w):
        for cl in range(0,w):
            d = dict()
            d["UL"] = np.unique(rasterm[0:(window-w+rw), 0:(window-w+cl)], return_counts=True)#UpLeft
            d["UR"] = np.unique(rasterm[0:(window-w+rw), (c+w-window-cl):c], return_counts=True)#UpRight
            d["DL"] = np.unique(rasterm[(r-window+w-rw):r, 0:(window-w+cl)], return_counts=True)#DownLeft
            d["DR"] = np.unique(rasterm[(r-window+w-rw):r, (c+w-window-cl):c], return_counts=True)#DownRight
            for key in d.keys():
                if len(np.where(d[key][0] == missing)[0]) != 0:
                    d[key] = np.delete( d[key][1], np.where(d[key][0] == missing))
                    #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
                else:
                    d[key] = d[key][1]
                if debugging:
                    print("Working on coords ",rw,",",cl,". classes len: ",len(d[key]),". window size=",window)
                if len(d[key]) != 0:
                    p = d[key]/np.sum(d[key])
                    if Type == "Rényi":
                        p_log = np.log(p)/ np.log(base)
                        if np.dot(p,p_log) == 0:
                            d[key] = 0
                        else:
                            d[key] = (-1)* np.dot(p,p_log)
                    else:
                        p_log = np.log(p)
                        d[key] = np.exp( (-1 )* np.dot(p,p_log))
                else:
                    d[key] = missing
                if key is "UL":
                    out[rw, cl] = d[key]
                elif key is "UR":
                    out[rw,c-cl-1] = d[key]
                elif key is "DL":
                    out[r-rw-1, cl] = d[key]
                else:
                    out[r-rw-1,c-cl-1] = d[key]
            del(d)
            gc.collect()
    print("\nComplete Shannon-Wiener index calculation on the corners.\n") 
    def UpShannon(cl, c= c, rasterm = rasterm, missing = missing, window= window, w = w, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
        for  rw in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[0:(window-w+rw), cl-w:(w+cl+1)], return_counts=True)#UpLeftd["D"] = np.unique(rasterm[(r-window+w-rw):r, cl-w:(w+cl+1)], return_counts=True)#DownLeft
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) != 0:
                p = tw_values/np.sum(tw_values)
                if Type == "Rényi":
                    p_log = np.log(p)/ np.log(base)
                    if np.dot(p,p_log) == 0:
                        return (0)
                    else:
                        return((-1)* np.dot(p,p_log))
                else:
                    p_log = np.log(p)
                    return (np.exp( (-1 )* np.dot(p,p_log)))
            else:
                return (missing)
    def DownShannon(cl, c= c, rasterm = rasterm, missing = missing, window= window, w = w, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
        for  rw in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[(r-window+w-rw):r, cl-w:(w+cl+1)], return_counts=True)#DownLeft#UpLeftd["D"] = 
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) != 0:
                p = tw_values/np.sum(tw_values)
                if Type == "Rényi":
                    p_log = np.log(p)/ np.log(base)
                    if np.dot(p,p_log) == 0:
                        return (0)
                    else:
                        return((-1)* np.dot(p,p_log))
                else:
                    p_log = np.log(p)
                    return (np.exp( (-1 )* np.dot(p,p_log)))
            else:
                return (missing)
    for  rw in range(0, w):  
        out[rw, w:(c-w)] = np.asarray(Parallel(n_jobs = NcCores)(delayed(UpShannon)(cl) for cl in range(w,c-w)))
        out[r-rw-1, w:(c-w)] = np.asarray(Parallel(n_jobs = NcCores)(delayed(DownShannon)(cl) for cl in range(w,c-w)))
        printProgressBar(rw + 1, w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check

    def LeftShannon(rw, r = r, rasterm = rasterm, missing = missing, window= window, w = w, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
        for cl in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[ rw-w:(w+rw+1), 0:(window-w+cl)], return_counts=True)#Left
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) != 0:
                p = tw_values/np.sum(tw_values)
                if Type == "Rényi":
                    p_log = np.log(p)/ np.log(base)
                    if np.dot(p,p_log) == 0:
                        return (0)
                    else:
                        return((-1)* np.dot(p,p_log))
                else:
                    p_log = np.log(p)
                    return (np.exp( (-1 )* np.dot(p,p_log)))
            else:
                return (missing)
    def RightShannon(rw, r = r, rasterm = rasterm, missing = missing, window= window, w = w, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
        for  cl in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[ rw-w:(w+rw+1), (c+w-window-cl):c], return_counts=True)#Right
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) != 0:
                p = tw_values/np.sum(tw_values)
                if Type == "Rényi":
                    p_log = np.log(p)/ np.log(base)
                    if np.dot(p,p_log) == 0:
                        return (0)
                    else:
                        return((-1)* np.dot(p,p_log))
                else:
                    p_log = np.log(p)
                    return (np.exp( (-1 )* np.dot(p,p_log)))
            else:
                return (missing)
    for cl in range(0,w):
        out[w:(r-w), cl] = np.asarray(Parallel(n_jobs = NcCores)(delayed(LeftShannon)(rw) for rw in range(w,r-w)))
        out[w:(r-w), c-cl-1] = np.asarray(Parallel(n_jobs = NcCores)(delayed(RightShannon)(rw) for rw in range(w,r-w)))
        printProgressBar(cl + 1, w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    print("\nComplete Shannon-Wiener index calculation on the borders.\n") 
    return(out)

def ShannonP(out, rasterm, missing, w, base, debugging, Type, NcCores):
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: Shannon single.")
    def ShannonOP (rw):
        def ShannonOut (cl, rw = rw, rasterm = rasterm,  missing = missing, w = w, base = base, debugging = debugging, Type = Type):
            tw_labels, tw_values = np.unique(rasterm[(rw-w):(rw+w+1),(cl-w):(cl+w+1)], return_counts=True)
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                        #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Shannon - parallelized\nWorking on coords ",rw,",",cl,". classes len: ",len(tw_labels),". window size=",window)
            if len(tw_values) != 0:
                p = tw_values/np.sum(tw_values)
                if Type == "Rényi":
                    p_log = np.log(p) / np.log(base)
                    if (np.dot(p,p_log)) == 0:
                        return(0)
                    else:
                        return((-1) * (np.dot(p,p_log)))
                else:
                    p_log= np.log(p)
                    return(np.exp((-1) * np.dot(p,p_log)))
            else:
                return(missing)
        printProgressBar(rw - w + 1, r - 2*w , prefix = 'Progress:', suffix = 'Complete', length = 50)#check
        Shannon_Out = Parallel(n_jobs = NcCores)(delayed(ShannonOut)(cl) for cl in range(w,(c- w)))
        return (Shannon_Out)
    out[w:(r-w), w:(c-w)]=np.asarray(Parallel(n_jobs = NcCores)(delayed(ShannonOP)(rw) for rw in range(w,(r-w)))).reshape(r-2*w,c-2*w)
    print("\n\n Parallel calculation of Shannon's index complete.\n")
    return(out)




#test
m1 = np.full((3,3),5)
m2 = np.full((3,3),8)
m3 = np.full((3,3),6)
m4 = np.full((3,3),9)
m = np.vstack((np.hstack((m1,m2)),np.hstack((m3,m4))))*0.1
print(m)
rao = spectralrao(input=m, missing= -1, window=3, shannon= False ,debugging= False, NcCores=1)


   
