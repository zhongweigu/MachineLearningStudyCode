import numpy as np
import python_speech_features as mfcc
from sklearn import preprocessing

def calculating_delta(array):
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0 :
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1

        denominator = 0.0
        numerator = np.zeros(20)

        for j, (end_idx, start_idx) in enumerate(index):
            weight = j + 1
            numerator += weight * (array[end_idx] - array[start_idx])
            denominator += 2 * weight * weight

        deltas[i] = numerator / denominator

    return deltas

def extract_features(audio, rate):
    mfcc_feat = mfcc.mfcc(audio, rate, 0.025 , 0.01, 20, appendEnergy=True)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculating_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat, delta))
    return combined
