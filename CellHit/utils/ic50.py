import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def ic50_computer(concentrations, responses):

    def hill_equation(concentration, low_asympt, slope,ec50):
        return low_asympt + (1 - low_asympt) / (1 + np.exp(slope*(np.log(concentration) - ec50)))

    popt, pcov = curve_fit(hill_equation, concentrations, responses, p0=[0, 1, 1])
    
    low_asympt, slope, ec50 = popt
    ic50 = np.log(((1-low_asympt)/(0.5-low_asympt))-1)/slope + ec50
    
    return ic50