from matplotlib import colors,pyplot
import numpy as np  
import random
import math

def overlap(sdr1,sdr2):
    """
    Return the overlapping SDR || AND operation between 2 SDR
    """
    active = []
    for i in range(len(sdr1)):
        for j in range(len(sdr2)):
            if sdr1[i][j] == sdr2[i][j]:
                if sdr1[i][j] == 1:
                    active.append((i,j))
    size = len(sdr1)
    overlap = np.zeros((size,size))
    for bit in active:
        overlap[bit[0],bit[1]] = 1
    return overlap

def overlap_score(sdr1,sdr2):
    """
    Compute overlap score given 2 sdr 
    """
    score = 0
    for i in range(len(sdr1)):
        for j in range(len(sdr2)):
            if sdr1[i][j] == sdr2[i][j]:
                if sdr1[i][j] == 1:
                    score += 1
    return score

def viz(sdr):
    """
    Turn SDR into grid-like representation. We accept 2d numpy array
    """
    colormap = colors.ListedColormap(["black","red"])
    pyplot.figure(figsize=(5,5))
    pyplot.imshow(sdr,cmap=colormap)
    pyplot.show()

def vizComplete(sdr):
    """
    Turn SDR into grid-like representation. We accept 2d numpy array
    """
    colormap = colors.ListedColormap(["black","red","yellow"])
    pyplot.figure(figsize=(5,5))
    pyplot.imshow(sdr,cmap=colormap)
    pyplot.show()

def generate_sdr(size,sparsity,dimensionality=2):
    """
    Randomly generate an SDR of size and sparsity
    """
    if dimensionality == 2:
        data = np.zeros((size,size))
        active = len(data)**2*sparsity
        while active>0:
            i = random.randint(0,len(data)-1)
            j = random.randint(0,len(data)-1)
            data[i][j] = 1
            active -= 1
        return data
    elif dimensionality == 0:
        return TypeError('Dimensionality  cannot be 0')
    elif dimensionality == 1:
        data = np.zeros(size)
        active = len(data)**2*sparsity
        while active>0:
            i = random.randint(0,len(data)-1)
            data[i] = 1
            active -= 1
        return data

def dimensionalityReduction(input_sdr):
    """
    Transforms a 2d SDR into 1d array
    """
    uintType = "uint32"
    output = np.zeros(len(input_sdr)**2,dtype=uintType)
    p = 0
    for i in range(len(input_sdr)):
        for j in range(len(input_sdr)):
            output[p] = input_sdr[i][j]
            p += 1
    return output

def viz1d(input_sdr):
    """
    Vizualize 1d sdr
    """
    size = int(math.sqrt(len(input_sdr)))
    arr_2d = np.reshape(input_sdr, (size, size))
    viz(arr_2d)
