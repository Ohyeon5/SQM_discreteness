from utils import plot_graphs
import numpy as np

if __name__ == '__main__':
    valLossLogger = [np.nan, np.nan, 0.7, np.nan, np.nan, 0.1, np.nan, np.nan, 0.01]
    valAccLogger = [np.nan, np.nan, 70, np.nan, np.nan, 90, np.nan, np.nan, 95]
    lossLogger = [1.4, 0.9, 0.7, 0.5, 0.2, 0.09, 0.04, 0.005, 0.001]
    accLogger = [50, 60, 75, 90, 96, 97, 98, 99, 99]
    disc = 1
    spatial = 2
    temporal = 4
    model_path = './saved_models/'
    plot_graphs(valLossLogger, valAccLogger, lossLogger, accLogger, disc, spatial, temporal, model_path)
