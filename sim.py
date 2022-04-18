from components.encoders import TimeEncoder, WordEncoder 
from components.sdr import viz,generate_sdr,overlap,overlap_score, vizComplete, dimensionalityReduction
from nupic.pool import SpatialPooler as SP
from nupic.temporal_memory.temporal_memory import TemporalMemoryApicalTiebreak
import numpy as np
from tqdm import tqdm
import matplotlib
import random
import matplotlib.pyplot as plt

def percentOverlap(x1, x2, size):
  """
  Computes the percentage of overlap between vectors x1 and x2.

  @param x1   (array) binary vector
  @param x2   (array) binary vector
  @param size (int)   length of binary vectors

  @return percentOverlap (float) percentage overlap between x1 and x2
  """
  nonZeroX1 = np.count_nonzero(x1)
  nonZeroX2 = np.count_nonzero(x2)
  minX1X2 = min(nonZeroX1, nonZeroX2)
  percentOverlap = 0
  if minX1X2 > 0:
    percentOverlap = float(np.dot(x1, x2))/float(minX1X2)
  return percentOverlap

def sequence(sentence):
        w = WordEncoder()  
        sequence = [] 
        for i in tqdm(range(len(sentence))):
            sequence.append(dimensionalityReduction(w.encode(sentence[i])))
        return sequence
        
def Test1():
    random.seed(1)
    uintType = "uint32"
    inputDimensions = (1024,1)
    columnDimensions = (1024,1)
    inputSize = np.array(inputDimensions).prod()
    columnNumber = np.array(columnDimensions).prod()

    input = generate_sdr(1024,0.02,1)

    activeCols = np.zeros(columnNumber, dtype=uintType)

    sp = SP(
            input_dims=inputDimensions,
            minicolumn_dims=columnDimensions,
            active_minicolumns_per_inh_area=10,
            local_density=-1.0,
            potential_radius=int(0.5*inputSize),
            potential_percent=0.5,
            global_inhibition=False,
            stimulus_threshold=0,
            synapse_perm_inc=0.03,
            synapse_perm_dec=0.008,
            synapse_perm_connected=0.1,
            min_percent_overlap_duty_cycles=0.001,
            duty_cycle_period=1000,
            boost_strength=0.0,
            seed=1,
        )

    print('Connnecting synapses to input space')
    sp.compute(input, True, activeCols)
    print('Calculating overlaps')
    overlaps = sp.calculate_overlap(input)
    print('overlaps', overlaps)
    print('active columns',sp.active_minicolumns_per_inh_area)
    activeColsScores = []
    for i in activeCols.nonzero():
        activeColsScores.append(overlaps[i])
    print('active column scores',activeColsScores)

    bins = np.linspace(min(overlaps), max(overlaps), 28)
    plt.hist(overlaps, bins, alpha=0.5, label="All cols")
    plt.hist(activeColsScores, bins, alpha=0.5, label="Active cols")
    plt.legend(loc="upper right")
    plt.xlabel("Overlap scores")
    plt.ylabel("Frequency")
    plt.title("Figure 1: Column overlap of a SP with random input.")
    plt.show()

def Test2():
    print('Starting...')
    sentence = ['green','red','blue']
    seq = sequence(sentence)
    random.seed(1)
    uintType = "uint32"
    inputDimensions = (len(seq[0]),1)
    columnDimensions = (1024,1)
    inputSize = np.array(inputDimensions).prod()
    columnNumber = np.array(columnDimensions).prod()

    activeCols = np.zeros(columnNumber, dtype=uintType)
    print('Initializing Spatial Pooler')
    sp = SP(
            input_dims=inputDimensions,
            minicolumn_dims=columnDimensions,
            active_minicolumns_per_inh_area=10,
            local_density=-1.0,
            potential_radius=int(0.5*inputSize),
            potential_percent=0.5,
            global_inhibition=False,
            stimulus_threshold=0,
            synapse_perm_inc=0.03,
            synapse_perm_dec=0.008,
            synapse_perm_connected=0.1,
            min_percent_overlap_duty_cycles=0.001,
            duty_cycle_period=1000,
            boost_strength=0.0,
            seed=1,
        )
    #LEARNIG A SEQUENCE
    for i in range(len(seq)):
        print(f'Learning sequence {i}')
        print('Connnecting synapses to input space')
        sp.compute(seq[i], True, activeCols)
        overlaps = sp.get_boosted_overlaps()
        print('Current Overlaps', overlaps)
        print('Potential Pool',len(sp.potential_pools))
Test2()
