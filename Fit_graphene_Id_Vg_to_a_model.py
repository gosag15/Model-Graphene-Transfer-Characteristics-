import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model
#IMPORT DATA
dataset = pd.read_excel('84ppm_cnp_shift.xlsx')
X_Vg = dataset.iloc[:, 0].values
Id_y = dataset.iloc[:, 1:].values
#DECLARE VARIABLES
CNP_compiled = []
CNPD_compiled = []
R_model_comp = {}
R_y_dict = {}
CNP_quad_comp = {}  

#SET DRAIN VOLTAGE VALUE(V), DEVICE LENGTH (um), L, DEVICE WIDTH (um), W.
Vd, L, W = 0.002, 0.2, 0.2

#CALCULATE RESISTANCE, AND MOBILITY, AND CNP VALUE(FOR DOPING ESTIMATION)
def func(x,a,b,c):
        return a + b*x + c*x**2
    #Define obtain rough CNP estimate to calculate doping conc.  
                   

def id_vg_model(nbg_minus_vdirac, Rc, mobility, n_dirac):
    return Rc + (L/W)/(1.6e-19*mobility) * (1/((n_dirac**2 + nbg_minus_vdirac**2)**0.5)) 

fet_model = Model(id_vg_model)
for i in range(Id_y.shape[1]):
    R_y = []
    current_col = Id_y[:,i]
    cnp_index = list(current_col).index(min(current_col))
    popt, pcov = curve_fit(func, X_Vg[cnp_index -15: cnp_index + 15], current_col[cnp_index -15: cnp_index + 15])   
    #Find CNP
    new_y = func(X_Vg, *popt)
    cnp_new_index = list(new_y).index(min(new_y))
    CNP_quad_comp[dataset.columns[i+1]] = X_Vg[cnp_new_index]
    nbg_minus_vdirac = (X_Vg - CNP_quad_comp[dataset.columns[i+1]]) *  7.57E10# Cg/e * CNP shift
    
    for item in range(current_col.shape[0]):
#    CONVERT CURRENT TO RESISTANCE  
        res_val = Vd*(current_col[item])**(-1)
        R_y = np.append(R_y, res_val)
    R_y_dict[dataset.columns[i+1]] = R_y 
    params = fet_model.make_params(Rc= 3000, mobility = 200, n_dirac = 0.0001 * 7.57E10)
    #fet_model.set_param_hint('mobility', min = 10, max = 10000)
    #FIT R_Y INTO EQUATION FOR ID_VG
    lower_bound , upper_bound = cnp_index - 25, cnp_index + 25
    result = fet_model.fit(R_y[lower_bound:upper_bound], params, nbg_minus_vdirac = nbg_minus_vdirac[lower_bound:upper_bound])
    R_model = result.best_fit
    R_model_comp[dataset.columns[i+1]] = R_model
       
    print(result.fit_report())
    plt.figure(dataset.columns[i+1])
    
    plt.scatter(X_Vg, R_y_dict[dataset.columns[i+1]], label='actual data', color = 'b')
    #plt.plot(X_Vg[lower_bound:upper_bound], result.init_fit, 'k--', label='initial fit')
    plt.scatter(X_Vg[lower_bound:upper_bound], result.best_fit, label='best fit', color = 'r')
    plt.xlabel('Vg (V)')
    plt.ylabel('Id (A)')
    plt.legend(loc='best')
    plt.show()
    cnp_index_model = X_Vg[lower_bound:upper_bound][list(R_model).index(max(R_model))]
    CNP_compiled.append(cnp_index_model)
   
for index in range(len(CNP_compiled)-1):
    CNPD_iter = abs(CNP_compiled[index]-CNP_compiled[index + 1])
    CNPD_compiled = np.append(CNPD_compiled,CNPD_iter)
mean_CNPD = np.mean(CNPD_compiled)
std_dev_CNPD = np.std(CNPD_compiled)  

    
