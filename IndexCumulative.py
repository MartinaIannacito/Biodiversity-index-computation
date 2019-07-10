#standard import
import os
import sys
#math import
import numpy as np
import math
#np.set_printoptions(threshold=sys.maxsize)
import scipy
from scipy import spatial
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import itertools
import gc
import numba as nb
import h5py
#geo import
from osgeo import ogr
from osgeo import gdal
import rasterio
# use OGR specific exceptions
ogr.UseExceptions()
MyVariable = None
#print(np.intp)
import platform
#print(platform.architecture())

####progress bar function from Greenstick https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/34325723

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledlen = int(len * iteration // total)
    bar = fill * filledlen + '-' * (len - filledlen)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def IndexComputation(input, missing, Type, window = 3, mode="single", BergerParker=False, alpha=1, base = np.exp(1), integer =False, NaTolerance=0.0, simplify=3, NcCores=1, debugging=False): 
    mfactor = 1
    Shannon = False
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
    if base < np.finfo(float).eps:
        raise ValueError("base value must be in the (0,+\u221E) interval. Exiting...")
    if not  (isinstance(input,int) or isinstance(input,np.ndarray)):
        raise ValueError("alpha must be a number or a vector. Exiting...")
    if mode=="single":
        if isinstance(alpha,list) or isinstance(alpha, np.ndarray):
            raise ValueError("In mode \"single\" alpha must be a single number. Exiting...")
        if alpha < 0:
            raise ValueError("The alpha value must be a non-negative number. Exiting...")
        if (alpha is 1 or np.abs(alpha-1) < np.finfo(float).eps) and not BergerParker :
            Shannon = True
        if alpha >= float("inf"):
            BergerParker = True
        if integer:
            alpha = int(alpha)
    elif mode == "iterative":
        if isinstance(alpha, np.ndarray):
            alpha = alpha.flatten().tolist()
        if len(alpha) != 2:
            raise ValueError("In mode \"iterative\" alpha must be a numeric vector containing the star and the stop value. Exiting...")
        start = int(alpha[0])
        if start < 0 :
            raise ValueError("The starting value must be a non-negative number. Exiting...")
        stop = int(alpha[1])
        if stop <= start:
            raise ValueError("Integer part of the starting value, alpha[1], must be strictly greater that the integer part of the stopping value, alpha[2]. Exiting...")
        if start <= 1 and 1 <= stop:
            Shannon = True
    elif mode == "sequential":
        if isinstance(alpha, list):
            alpha = np.asarray(alpha)#.flatten().tolist()
        if len(alpha) < 2:
            raise ValueError("In mode \"sequential\" alpha must be a numeric vector containing at least two values. Exiting...")
        if len(np.where(alpha < 0)[0]) != 0:
            raise ValueError("The alpha values must be non-negative numbers. Exiting...")
        if integer:
            val = np.unique( alpha.astype(int) )
        else:
            val = np.unique( alpha )
    else:
        raise ValueError("The choosen mode is not defined. Exiting...")
    if not( Type == "Hill" or Type == "Rényi"):
        raise ValueError("The choosen Type is not defined. Exiting...")
    isfloat = False # If data are float numbers, transform them in integer, this may allow for a shorter computation time on big datasets.
    if len( rasterm[ rasterm != rasterm.astype(int)]) > 0:
        print("Converting input data in an integer matrix...")
        isfloat = True
        mfactor = 100 ** simplify
        rasterm = np.array(rasterm*mfactor, dtype = "int")
        #Print user prints
    if mode == "single":
        if BergerParker:
            print("Matrix check OK: \nBerger-Parker output matrix will be returned")
        elif Shannon:
            print("Matrix check OK: \nShannon output matrix will be returned")
        else:
            print("Matrix check OK: \nRényi with parameter value=", alpha," output matrix will be returned")
    elif mode == "iterative":
        if BergerParker and not Shannon:
            print("Matrix check OK: \nRényi output matrix will be returned for parameters integer values in [",start,",",stop,"] with Berger-Parker output matrix" )
        elif BergerParker and Shannon:
            print("Matrix check OK: \nRényi output matrix will be returned for parameters integer values in [",start,",",stop,"] with Berger-Parker output matrix (Shannon output matrix is includednot )")
        elif not BergerParker and Shannon:
            print("Matrix check OK: \nRényi output matrix will be returned for parameters integer values in [",start,",",stop,"] (Shannon output matrix is includednot )")
        else:
            print("Matrix check OK: \nRényi output matrix will be returned for parameters integer values in [",start,",",stop,"]")
    if mode == "single":
        if BergerParker:
            print("Matrix check OK: \nBerger-Parker output matrix will be returned")
        elif Shannon:
            print("Matrix check OK: \nShannon output matrix will be returned")
        else:
            print("Matrix check OK: \nRényi with parameter value=", alpha," output matrix will be returned")
    elif mode == "iterative":
        if BergerParker and not Shannon:
            print("Matrix check OK: \nRényi output matrix will be returned for parameters integer values in [",start,",",stop,"] with Berger-Parker output matrix" )
        elif BergerParker and Shannon:
            print("Matrix check OK: \nRényi output matrix will be returned for parameters integer values in [",start,",",stop,"] with Berger-Parker output matrix (Shannon output matrix is includednot )")
        elif not BergerParker and Shannon:
            print("Matrix check OK: \nRényi output matrix will be returned for parameters integer values in [",start,",",stop,"] (Shannon output matrix is includednot )")
        else: 
            print("Matrix check OK: \nRényi output matrix will be returned for parameters integer values in [",start,",",stop,"]")
    #
    ## Derive operational moving window
    #
    if window %2 == 1:
        w = int(( window - 1 ) / 2)
    else:
        raise ValueError("Moving window size must be an odd number.")
    
    if NcCores == 1:
        if mode == "single":
            outS = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
            if Shannon:
                outS = BorderShannonS(outS, rasterm, missing, w, base, debugging, Type)
                outS = ShannonS(outS, rasterm, missing, w, base, debugging, Type)
                print(("\nCalculation of Shannon's index is completenot \n")) # End ShannonD
            elif BergerParker:
                outS = BorderBergerParkerS(outS, rasterm, missing, w, base, debugging, Type)
                outS = BergerParkerS(outS, rasterm, missing, w, base, debugging, Type)
                print(("\nCalculation of Berger-Parker's index is completenot \n"))# End BergerParker
            else:
                outS = BorderIndexS(outS, rasterm, missing, w, alpha, base, debugging, Type)
                outS= IndexS(outS, rasterm, missing, w, alpha, base, debugging, Type)
                print(("\nCalculation of Rényi's index complete.\n"))
            return (outS)
        elif mode == "iterative":
            out = dict()
            for ALPHA in range(start,stop+1):
                outS = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
                if ALPHA == 1:
                    s = "ShannonAlpha 1"
                    outS = BorderShannonS(outS, rasterm, missing, w, base, debugging, Type)
                    out[s] = ShannonS(outS, rasterm, missing, w, base, debugging, Type)
                    print(("\nCalculation of Shannon's index is also completenot \n"))# End ShannonD
                else:
                    s = "RényiAlpha"+str(ALPHA)
                    outS = BorderIndexS(outS, rasterm, missing, w, ALPHA, base, debugging, Type)
                    out[s] = IndexS(out = outS, rasterm = rasterm, missing = missing, w = w, alpha = ALPHA, base = base, debugging = debugging, Type = Type)#, isfloat = isfloat, mfactor = mfactor)
            print(("\nCalculation of Rényi's index complete.\n"))
            if BergerParker:
                outS = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
                s="Berger-Parker"
                outS = BorderBergerParkerS(outS, rasterm, missing, w, base, debugging, Type)
                out[s] = BergerParkerS(outS, rasterm, missing, w, base, debugging, Type)
                print(("\nCalculation of Berger-Parker's index is also completenot \n"))      
            return(out)
        elif mode == "sequential":
            out = dict()
            for ALPHA in val:
                outS = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
                if np.abs(ALPHA-1)< np.finfo(float).eps or ALPHA == 1:
                    s = "ShannonAlpha 1"
                    outS = BorderShannonS(outS, rasterm, missing, w, base, debugging, Type)
                    out[s] = ShannonS(outS, rasterm, missing, w, base, debugging, Type)
                    print(("\nCalculation of Shannon's index is also completenot \n"))#End ShannonD
                else:
                    s = "RényiAlpha"+str(ALPHA)
                    outS = BorderIndexS(outS, rasterm, missing, w, ALPHA, base, debugging, Type)
                    out[s] = IndexS (out = outS, rasterm = rasterm, missing = missing, w = w, alpha = ALPHA, base = base, debugging = debugging, Type = Type)
            print("\nCalculation of Rényi's index complete.\n")
            if BergerParker:
                outS = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
                s="Berger-Parker"
                outS = BorderBergerParkerS(outS, rasterm, missing, w, base, debugging, Type)
                out[s] = BergerParkerS(outS, rasterm, missing, w, base, debugging, Type)
                print(("\nCalculation of Berger-Parker's index is also completenot \n"))
            return(out)
    if NcCores > 1 :
        print("##################### Starting parallel calculation #######################")
        if mode == "single":
            outP = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
            if Shannon: 
                outP = BorderShannonP(outP, rasterm, missing, w, base, debugging, Type, NcCores)
                outP = ShannonP(out = outP, rasterm = rasterm, missing = missing, w = w, base = base, debugging = debugging, Type = Type, NcCores = NcCores)#)#, isfloat = False, mfactor = 0, 
            elif BergerParker:
                outP = BorderBergerParkerP(outP, rasterm, missing, w, base, debugging, Type, NcCores)
                outP = BergerParkerP(out= outP, rasterm = rasterm, missing = missing, w = w, base = base, debugging = debugging, Type = Type,  NcCores = NcCores)#isfloat = isfloat, mfactor = mfactor,
            else:
                outP = BorderIndexP(out = outP, rasterm = rasterm, missing = missing, w = w, alpha = alpha, base = base, debugging = debugging, Type = Type, NcCores = NcCores)#isfloat = False, mfactor = 0, N
                outP = IndexP(out = outP, rasterm = rasterm, missing = missing, w = w, alpha = alpha, base = base, debugging = debugging, Type = Type, NcCores = NcCores)#isfloat = False, mfactor = 0, N
            return(outP)
        elif mode == "iterative":
            out = dict()
            for ALPHA in range(start,stop+1):
                outP = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
                if ALPHA == 1:
                    s = "ShannonAlpha 1"
                    outP = BorderShannonP (outP, rasterm, missing, w, base, debugging, Type, NcCores)
                    outP = ShannonP(out = outP, rasterm = rasterm, missing = missing, w = w, base = base, debugging = debugging, Type = Type, NcCores = NcCores)#isfloat = isfloat, mfactor = mfactor, 
                    out[s] = outP
                    print(("\nCalculation of Shannon's index is also completenot \n"))# End ShannonD
                else:    
                    s = "RényiAlpha" + str(ALPHA)
                    outP = BorderIndexP(out = outP, rasterm = rasterm, missing = missing, w = w, alpha = ALPHA, base = base, debugging = debugging, Type = Type, NcCores = NcCores)
                    outP = IndexP(out= outP, rasterm = rasterm, missing = missing, w = w, alpha = ALPHA, base = base, debugging = debugging, Type = Type, NcCores = NcCores)#isfloat = isfloat, mfactor = mfactor, 
                    out[s] = outP
            print(("\nCalculation of Rényi's index complete.\n"))
            if BergerParker:
                outP = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
                s = "Berger-Parker"
                outP = BorderBergerParkerP(outP, rasterm, missing, w, base, debugging, Type, NcCores)
                outP = BergerParkerP(out = outP, rasterm = rasterm, missing = missing, w = w, base = base, debugging = debugging, Type = Type, NcCores = NcCores)#isfloat = isfloat, mfactor = mfactor, 
                out[s] = outP
                print(("\nCalculation of Berger-Parker's index is also completenot \n"))
            return(out)
        elif mode == "sequential":
            out = dict()
            for ALPHA in val:
                outP = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
                if np.abs(ALPHA-1)< np.finfo(float).eps:
                    s = "ShannonAlpha 1"
                    outP = BorderShannonP (outP, rasterm, missing, w, base, debugging, Type, NcCores)
                    outP = ShannonP(out = outP, rasterm = rasterm, missing = missing, w = w, base = base, debugging = debugging, Type = Type, NcCores = NcCores)#isfloat = isfloat, mfactor = mfactor, 
                    out[s] = outP
                    print(("\nCalculation of Shannon's index is also completenot \n"))# End ShannonD
                else:
                    s = "RényiAlpha"+str(ALPHA)
                    outP = BorderIndexP(out = outP, rasterm = rasterm, missing = missing, w = w, alpha = ALPHA, base = base, debugging = debugging, Type = Type, NcCores = NcCores)
                    outP = IndexP(out= outP, rasterm = rasterm, missing = missing, w = w, alpha = ALPHA, base = base, debugging = debugging, Type = Type, NcCores = NcCores)#isfloat = isfloat, mfactor = mfactor, 
                    out[s] = outP
            print(("\nCalculation of Rényi's index complete.\n"))
            if BergerParker:
                outP = np.repeat(missing,rasterm.shape[0]*rasterm.shape[1]).reshape(rasterm.shape[0],rasterm.shape[1]).astype(np.float64)
                s = "Berger-Parker"
                outP = BorderBergerParkerP(outP, rasterm, missing, w, base, debugging, Type, NcCores)
                outP = BergerParkerP(out = outP, rasterm = rasterm, missing = missing, w = w, base = base, debugging = debugging, Type = Type, NcCores = NcCores)#isfloat = isfloat, mfactor = mfactor, 
                out[s] = outP
                print(("\nCalculation of Berger-Parker's index is also completenot \n"))
            return(out)


###nc=1
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
        printProgressBar(cl + 1, c - w + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
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

def BorderBergerParkerS (out, rasterm, missing, w, base, debugging, Type):
    print("\nStarting Berger-Parker index calculation on the border:\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: ShaBerger-Parkernnon single on borders.")
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
                    p = np.max(d[key]/np.sum(d[key]))
                    if Type == "Rényi":
                        d[key] = np.log( 1/p ) / np.log(base)
                    else:
                        d[key] = (1/p)
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
    print("\nComplete Berger-Parker index calculation on the corners.\n") 
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
                    p = np.max(d[key]/np.sum(d[key]))
                    if Type == "Rényi":
                        d[key] = np.log( 1/p ) / np.log(base)
                    else:
                        d[key] = (1/p)
                else:
                    d[key] = missing
                if key is "U":
                    out[rw, cl] = d[key]
                else:
                    out[r-rw-1, cl] = d[key]
            del(d)
            gc.collect()
        printProgressBar(cl + 1, c - w + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
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
                    p = np.max(d[key]/np.sum(d[key]))
                    if Type == "Rényi":
                        d[key] = np.log( 1/p ) / np.log(base)
                    else:
                        d[key] = (1/p)
                else:
                    d[key] = missing
                if key is "L":
                    out[rw, cl] = d[key]
                else:
                    out[rw, c-cl-1] = d[key]
            del(d)
            gc.collect()
        printProgressBar(rw + 1, r - w + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    print("\nComplete Berger-Parker index calculation on the borders.\n") 
    return (out)

def BergerParkerS (out, rasterm, missing, w, base, debugging, Type):
    print("\nStarting Berger-Parker index calculation:\n")
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
        #            tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) != 0:
                p = np.max(tw_values/np.sum(tw_values))
                if Type=="Rényi":
                    out[rw,cl]= np.log( 1/p ) / np.log(base)
                else:
                    out[rw,cl] = (1/p)
        printProgressBar(rw-w + 1, rasterm.shape[0]-w+1, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    return(out)


def BorderIndexS (out, rasterm, missing, w, alpha, base, debugging, Type):
    print("\nStarting Rényi index calculation on the border, with alpha = ", alpha, ":\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: Rényi index single.")
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
                    p = (d[key]/np.sum(d[key]))
                    if Type == "Rényi":
                        if np.log(np.sum(p**alpha)) == 0:
                            d[key] = 0
                        else:
                            d[key] = (1/(1-alpha)) * np.log(np.sum(p**alpha)) / np.log(base)
                    else:
                        d[key] = (np.sum(p**alpha))**(1/(1-alpha)) 
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
    print("\nComplete Rényi index calculation on the corners.\n") 
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
                    p = (d[key]/np.sum(d[key]))
                    if Type == "Rényi":
                        if np.log(np.sum(p**alpha)) == 0:
                            d[key] = 0
                        else:
                            d[key] = (1/(1-alpha)) * np.log(np.sum(p**alpha)) / np.log(base)
                    else:
                        d[key] = (np.sum(p**alpha))**(1/(1-alpha)) 
                else:
                    d[key] = missing
                if key is "U":
                    out[rw, cl] = d[key]
                else:
                    out[r-rw-1, cl] = d[key]
            del(d)
            gc.collect()
        printProgressBar(cl + 1, c - w + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
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
                    p = (d[key]/np.sum(d[key]))
                    if Type == "Rényi":
                        if np.log(np.sum(p**alpha)) == 0:
                            d[key] = 0
                        else:
                            d[key] = (1/(1-alpha)) * np.log(np.sum(p**alpha)) / np.log(base)
                    else:
                        d[key] = (np.sum(p**alpha))**(1/(1-alpha)) 
                else:
                    d[key] = missing
                if key is "L":
                    out[rw, cl] = d[key]
                else:
                    out[rw, c-cl-1] = d[key]
            del(d)
            gc.collect()
        printProgressBar(rw + 1, r - w + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    print("\nComplete Rényi index index calculation on the borders.\n") 
    return (out)



def IndexS (out, rasterm, missing, w, alpha, base, debugging, Type):
    print("\nStarting Rényi index calculation with parameter value=",alpha,"\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: ShRényi indexannon single.")
    #
    ## Loop over all the pixels
    #
    for rw in range(w, r - w):
        for cl in range(w, c - w):
            tw_labels, tw_values = np.unique(rasterm[(rw-w):(rw+w+1),(cl-w):(cl+w+1)], return_counts=True)
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
    #                tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) != 0:
                p = tw_values/np.sum(tw_values)
                if Type == "Rényi":
                    if np.log(np.sum(p**alpha)) == 0:
                        out[rw,cl] = 0
                    else:
                        out[rw,cl] = (1/(1-alpha)) * np.log(np.sum(p**alpha)) / np.log(base)
                else:
                    out[rw,cl] = (np.sum(p**alpha))**(1/(1-alpha)) 
        printProgressBar(rw-w + 1, r-w+1, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    return(out)

###nc>1

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
    return (out)



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


def BorderBergerParkerP (out, rasterm, missing, w, base, debugging, Type, NcCores):
    print("\nStarting Berger-Parker index calculation on the border:\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: Berger-Parker parallelized on borders.")
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
                    p = np.max(d[key]/np.sum(d[key]))
                    if Type == "Rényi":
                        d[key] = np.log( 1/p ) / np.log(base)
                    else:
                        d[key] = (1/p)
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
    print("\nComplete Berger-Parker index calculation on the corners.\n") 
    def UpBergerParker(cl, c= c, rasterm = rasterm, missing = missing, window= window, w = w, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
        for  rw in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[0:(window-w+rw), cl-w:(w+cl+1)], return_counts=True)#UpBergerParkerLeftd["D"] = np.unique(rasterm[(r-window+w-rw):r, cl-w:(w+cl+1)], return_counts=True)#DownLeft
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) != 0:
                p = np.max(tw_values/np.sum(tw_values))
                if Type=="Rényi":
                    return(np.log( 1/p ) / np.log(base))
                else:
                    return((1/p))
            else:
                return (missing)
    def DownBergerParker(cl, c= c, rasterm = rasterm, missing = missing, window= window, w = w, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
        for  rw in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[(r-window+w-rw):r, cl-w:(w+cl+1)], return_counts=True)#DownLeft#UpLeftd["D"] = 
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) != 0:
                p = np.max(tw_values/np.sum(tw_values))
                if Type=="Rényi":
                    return(np.log( 1/p ) / np.log(base))
                else:
                    return((1/p))
            else:
                return (missing)
    for  rw in range(0, w):  
        out[rw, w:(c-w)] = np.asarray(Parallel(n_jobs = NcCores)(delayed(UpBergerParker)(cl) for cl in range(w,c-w)))
        out[r-rw-1, w:(c-w)] = np.asarray(Parallel(n_jobs = NcCores)(delayed(DownBergerParker)(cl) for cl in range(w,c-w)))
        printProgressBar(rw + 1, w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check

    def LeftBergerParker(rw, r = r, rasterm = rasterm, missing = missing, window= window, w = w, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
        for cl in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[ rw-w:(w+rw+1), 0:(window-w+cl)], return_counts=True)#Left
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) != 0:
                p = np.max(tw_values/np.sum(tw_values))
                if Type=="Rényi":
                    return(np.log( 1/p ) / np.log(base))
                else:
                    return((1/p))
            else:
                return (missing)
    def RightBergerParker(rw, r = r, rasterm = rasterm, missing = missing, window= window, w = w, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
        for  cl in range(0, w):
            tw_labels, tw_values = np.unique(rasterm[ rw-w:(w+rw+1), (c+w-window-cl):c], return_counts=True)#Right
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Working on coords ",rw,",",cl,". classes len: ",len(tw_values),". window size=",window)
            if len(tw_values) != 0:
                p = np.max(tw_values/np.sum(tw_values))
                if Type=="Rényi":
                    return(np.log( 1/p ) / np.log(base))
                else:
                    return((1/p))
            else:
                return (missing)
    for cl in range(0,w):
        out[w:(r-w), cl] = np.asarray(Parallel(n_jobs = NcCores)(delayed(LeftBergerParker)(rw) for rw in range(w,r-w)))
        out[w:(r-w), c-cl-1] = np.asarray(Parallel(n_jobs = NcCores)(delayed(RightBergerParker)(rw) for rw in range(w,r-w)))
        printProgressBar(cl + 1, w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    print("\nComplete Berger-Parker index calculation on the borders.\n") 
    return (out)

def BergerParkerP(out, rasterm, missing, w, base, debugging, Type,  NcCores):#isfloat, mfactor,
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: Berger-Parker parallelized.")
    def BergerParkerOP (rw):
        def BergerParkerOut (cl, rw = rw, rasterm = rasterm, missing = missing, w = w, base = base, debugging = debugging, Type = Type):
            tw_labels, tw_values = np.unique(rasterm[(rw-w):(rw+w+1),(cl-w):(cl+w+1)], return_counts=True)
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                #    tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Berger-Parker - parallelized\nWorking on coords ",rw,",",cl,". classes len: ",len(tw_labels),". window size=", window)
            if len(tw_values) != 0:
                p = np.max(tw_values/np.sum(tw_values))
                if Type == "Rényi":
                    return(np.log(1/p) / np.log (base))
                else:
                    return(1/p)
        printProgressBar(rw - w + 1, r - 2*w , prefix = 'Progress:', suffix = 'Complete', length = 50)#check
        BergerParker_Out = Parallel(n_jobs = NcCores)(delayed(BergerParkerOut)(cl) for cl in range(w,c-w))
        return (BergerParker_Out)
    out[w:(r-w), w:(c-w)] = np.asarray(Parallel(n_jobs = NcCores)(delayed(BergerParkerOP)(rw) for rw in range(w,r - w))).reshape(r-2*w,c-2*w)
    print("\n\n Parallel calculation of Berger-Parker's index complete.\n")
    return(out)


def BorderIndexP (out, rasterm, missing, w, alpha, base, debugging, Type, NcCores):
    print("\nStarting Rényi index calculation on the border:\n")
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    if debugging:
        print("#check: Rényi prallelized.")
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
                    p = (d[key]/np.sum(d[key]))
                    if Type == "Rényi":
                        if np.log(np.sum(p**alpha)) == 0:
                            d[key] = 0
                        else:
                            d[key] = (1/(1-alpha)) * np.log(np.sum(p**alpha)) / np.log(base)
                    else:
                        d[key] = (np.sum(p**alpha))**(1/(1-alpha)) 
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
    print("\nComplete Rényi index calculation on the corners.\n") 
    def UpIndex(cl, c= c, rasterm = rasterm, missing = missing, window= window, w = w, alpha = alpha, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
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
                    if np.log(np.sum(p**alpha)) == 0:
                        return(0)
                    else:
                        return((1/(1-alpha)) * np.log(np.sum(p**alpha)) / np.log(base))
                else:
                    return((np.sum(p**alpha))**(1/(1-alpha))) 
            else:
                return (missing)
    def DownIndex(cl, c= c, rasterm = rasterm, missing = missing, window= window, w = w, alpha = alpha, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
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
                    if np.log(np.sum(p**alpha)) == 0:
                        return(0)
                    else:
                        return((1/(1-alpha)) * np.log(np.sum(p**alpha)) / np.log(base))
                else:
                    return((np.sum(p**alpha))**(1/(1-alpha)))
            else:
                return (missing)
    for  rw in range(0, w):  
        out[rw, w:(c-w)] = np.asarray(Parallel(n_jobs = NcCores)(delayed(UpIndex)(cl) for cl in range(w,c-w)))
        out[r-rw-1, w:(c-w)] = np.asarray(Parallel(n_jobs = NcCores)(delayed(DownIndex)(cl) for cl in range(w,c-w)))
        printProgressBar(rw + 1, w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    def LeftIndex(rw, r = r, rasterm = rasterm, missing = missing, window= window, w = w, alpha = alpha, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
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
                    if np.log(np.sum(p**alpha)) == 0:
                        return(0)
                    else:
                        return((1/(1-alpha)) * np.log(np.sum(p**alpha)) / np.log(base))
                else:
                    return((np.sum(p**alpha))**(1/(1-alpha)))
            else:
                return (missing)
    def RightIndex(rw, r = r, rasterm = rasterm, missing = missing, window= window, w = w, alpha = alpha, base = base, debugging =  debugging, Type = Type, NcCores = NcCores):
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
                    if np.log(np.sum(p**alpha)) == 0:
                        return(0)
                    else:
                        return((1/(1-alpha)) * np.log(np.sum(p**alpha)) / np.log(base))
                else:
                    return((np.sum(p**alpha))**(1/(1-alpha)))
            else:
                return (missing)
    for cl in range(0,w):
        out[w:(r-w), cl] = np.asarray(Parallel(n_jobs = NcCores)(delayed(LeftIndex)(rw) for rw in range(w,r-w)))
        out[w:(r-w), c-cl-1] = np.asarray(Parallel(n_jobs = NcCores)(delayed(RightIndex)(rw) for rw in range(w,r-w)))
        printProgressBar(cl + 1, w, prefix = 'Progress:', suffix = 'Complete', length = 50)#check
    print("\nComplete Rényi index calculation on the borders.\n") 
    return (out)

def IndexP(out, rasterm, missing, w, alpha, base, debugging, Type, NcCores):#isfloat, mfactor, 
    window = int( 2*w +1 )
    window = int( 2*w +1 )
    r = rasterm.shape[0]
    c = rasterm.shape[1]
    def IndexOP (rw):
        def IndexOut (cl, rw = rw, rasterm = rasterm, missing = missing, w = w, alpha=alpha, base = base, debugging = debugging, Type = Type):
            tw_labels, tw_values = np.unique(rasterm[(rw-w):(rw+w+1),(cl-w):(cl+w+1)], return_counts=True)
            if len(np.where(tw_labels == missing)[0]) != 0:
                tw_values = np.delete( tw_values, np.where(tw_labels == missing))
                #tw_labels = np.delete( tw_labels, np.where(tw_labels == missing))
            if debugging:
                print("Rényi - parallelized\nWorking on coords ",rw,",",cl,". classes len: ",len(tw_labels),". window size=", window)
            if len(tw_values) != 0:
                p = tw_values/np.sum(tw_values)
                if Type == "Rényi":
                    if np.log(np.sum(p**alpha)) == 0:
                        return(0)
                    else:
                        return((1/(1-alpha)) * np.log(np.sum(p**alpha)) / np.log(base))
                else:
                    return((np.sum(p**alpha))**(1/(1-alpha)))
            else:
                return (missing)
        printProgressBar(rw - w + 1, r - 2*w , prefix = 'Progress:', suffix = 'Complete', length = 50)#check
        Index_Out = Parallel(n_jobs = NcCores)(delayed(IndexOut)(cl) for cl in range(w,c-w))
        return (Index_Out)
    out[w:(r-w), w:(c-w)] = np.asarray(Parallel(n_jobs = NcCores)(delayed(IndexOP)(rw) for rw in range(w,r-w))).reshape(r-2*w,c-2*w)
    print("\n\n Parallel calculation of Rényi's index complete.\n")
    return(out)



m1 = np.full((4,4),0.5)
m2 = np.full((4,4),0.8)
m3 = np.full((4,4),0.6)
m4 = np.full((4,4),0.9)
m = np.vstack((np.hstack((m1,m2)),np.hstack((m3,m4))))
T = IndexComputation (input=m, missing= -1, Type= "Rényi", alpha=1,window=7, BergerParker= True, mode= "single",NcCores=6)


