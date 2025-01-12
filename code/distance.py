import numpy as np
import statistics
import scipy.stats as stats
from scipy.stats import wasserstein_distance, ks_2samp, energy_distance
from scipy.special import kl_div
import pennylane as qml

def probability_density_function(XX, YY):
    nx = len(XX)
    ny = len(YY)
    n = nx + ny

    min_val = min(np.min(XX), np.min(YY))
    max_val = max(np.max(XX), np.max(YY))
    bins = np.linspace(min_val, max_val, n + 1)

    X_pdf, _ = np.histogram(XX, bins=bins, density=True)
    Y_pdf, _ = np.histogram(YY, bins=bins, density=True)

    return X_pdf, Y_pdf

def sign(XX, YY):
    # XX baseline YY comparable
    if np.median(XX) > np.median(YY):
        return -1
    else:
        return 1

class DistanceMeasure:
    def __init__(self, XX, YY):
        self.XX = XX
        self.YY = YY
        self.sign = sign(XX, YY)

    def wasserstein(self):
        wd = wasserstein_distance(self.XX, self.YY)
        return wd * self.sign
        
    def kuiper_distance(self):

        nx = len(self.XX)
        ny = len(np.array(self.YY))
        n = nx + ny
    
        XY = np.concatenate([self.XX, self.YY])
        X2 = np.concatenate([np.repeat(1/nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1/ny, ny)])
    
        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]
    
        up = 0
        down = 0
        Res = 0
        E_CDF = 0
        F_CDF = 0
        height = 0
        power = 1
    
        for ii in range(0, n-2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            if XY_Sorted[ii+1] != XY_Sorted[ii]: height = F_CDF-E_CDF
            if height > up: up = height
            if height < down: down = height
    
        res = abs(down)**power + abs(up)**power
        return res * self.sign

    def anderson_darling_distance(self):

        # inverse due to baseline relative comparason
        XX = self.YY
        YY = self.XX
    
        nx = len(XX)
        ny = len(np.array(YY))
        n = nx + ny
    
        XY = np.concatenate([XX, YY])
        X2 = np.concatenate([np.repeat(1/nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1/ny, ny)])
    
        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]
    
        Res = 0
        E_CDF = 0
        F_CDF = 0
        G_CDF = 0
        height = 0
        SD = 0
        power = 1
    
        for ii in range(0, n-2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            G_CDF = G_CDF + 1/n
            SD = (n * G_CDF * (1-G_CDF))**0.5
            height = abs(F_CDF - E_CDF)
            if XY_Sorted[ii+1] != XY_Sorted[ii]:
                if SD>0:
                    Res = Res + (height/SD)**power
    
        return Res * self.sign

    def cvm_distance(self):

        nx = len(self.XX)
        ny = len(self.YY)
        n = nx + ny
    
        XY = np.concatenate([self.XX, self.YY])
        X2 = np.concatenate([np.repeat(1/nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1/ny, ny)])
    
        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]
    
        Res = 0;
        E_CDF = 0;
        F_CDF = 0;
        power = 1;
    
        for ii in range(0, n-2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            height = abs(F_CDF - E_CDF)
            if XY_Sorted[ii+1] != XY_Sorted[ii]: Res = Res + height**power
    
        return Res * self.sign

    def kolmogorov_smirnov(self):
        res = ks_2samp(self.XX, self.YY)
        return res[0] * self.sign

    def energy_distance_function(self):
        res = energy_distance(self.XX, self.YY)
        return res * self.sign

    def kruglov_distance(self):
        X_pdf, Y_pdf = probability_density_function(self.XX, self.YY)
        X_cdf = np.cumsum(X_pdf)
        Y_cdf = np.cumsum(Y_pdf)
        distance = np.max(np.abs(X_cdf - Y_cdf))
        return distance * self.sign

    # The fidelity similarity (or Bhattacharya coefficient, Hellinger affinity)
    def fidelity_similarity(self):
        X_pdf, Y_pdf = probability_density_function(self.XX, self.YY)
        result = np.sum(np.sqrt(X_pdf * Y_pdf))
        return result * self.sign

    def harmonic_mean_similarity(self):
        X_pdf, Y_pdf = probability_density_function(self.XX, self.YY)
         # Adding a tiny number to avoid a zero division
        epsilon = 1e-10
        result = 2 * np.sum((X_pdf * Y_pdf) / (X_pdf + Y_pdf + epsilon))
        return result * self.sign

    def hellinger_metric(self):
        X_pdf, Y_pdf = probability_density_function(self.XX, self.YY)
        result = np.sum((np.sqrt(X_pdf) - np.sqrt(Y_pdf))**2)/2
        return result * self.sign

    def patrick_fisher_distance(self):
        X_pdf, Y_pdf = probability_density_function(self.XX, self.YY)
        result = np.sum(np.abs(X_pdf - Y_pdf)**2)/2
        return result * self.sign

    def kullback_leibler_distance(self):
        X_pdf, Y_pdf = probability_density_function(self.XX, self.YY)
        X_pdf += 1e-10 # to aoid zero
        Y_pdf += 1e-10
        result = kl_div(X_pdf, Y_pdf).sum()
        return result * self.sign
