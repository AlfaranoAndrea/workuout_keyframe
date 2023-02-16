from math import pi, sqrt, exp
import numpy as np
from scipy.signal import find_peaks
def gauss(n=11,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

def movingAverage(x, k):
    return np.convolve(x, np.ones(k), 'valid')/k 


def normalizeZeroMean(sign):
    m= np.mean(sign)
    processed= sign- m
    return processed



def absolute(sign):
    return np.where(sign > 0 , sign , -sign  )

def findKeyPoints(y, distance):
    max= np.max(y)
    peaks,_= find_peaks(y,distance=distance,prominence=10)   
    return peaks[y[peaks]> max*0.1]