from ebpydlm import dlm, trend, seasonality, dynamic, autoReg, longSeason
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import time
from estimate.qrSolve import qrSolve
from estimate.qrFix import qrFixCoefBound
from ebpydlm.access.getters_to_df import export_olspriors, export_modelpriors, export_intervals, export_coefficients
from ebpydlm.access.model_priors import get_component_priors

def test_dlm(Y, X, output, mod, xsec, save_cs, save_var, autotune, iterations):
    plt.ioff() #turn off interactive plotting

    #normalize X
    norm_factor = 1#X.abs().max()
    #X = X/X.abs().max()

    #remove na from timeseries and variables
    X = X.astype('float64')
    X.fillna(0, inplace=True)
    Y.fillna(0, inplace=True)

    #dataframe to structured array
    y_data = np.array([tuple(i) for i in Y.to_numpy()], dtype = np.dtype(list(zip(Y.dtypes.index, Y.dtypes))))
    x1_data = np.array([tuple(i) for i in X.to_numpy()], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))

    #variable matrix
    var_len = len(x1_data.dtype.names)
    dates = y_data.shape[0]
    X_ = np.array([tuple(i) for i in np.zeros(var_len*dates).reshape(dates, var_len)], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))
    
    #model spec to dicts
    cs = list(dict.fromkeys([i.split('~')[0] for i in list(x1_data.dtype.names)]).keys())
    var = list(dict.fromkeys([i.split('~')[1] for i in list(x1_data.dtype.names)]).keys())
    boundaries = {k: (v1,v2) for k,v1,v2 in mod.iloc[:,0:3].to_numpy() if k in var}
    discount = {k: v for k,v in mod.loc[:,['VariableName', 'Memory']].to_numpy() if k in var}

    beta={}
    beta_var = {}
    filt_colnames = []
    mean_priors = {}
    cov_priors = {}

    for cs1 in cs:
        variable = [str(cs1)+'~'+i for i in var]
        #get qr solution
        qrsol = qrSolve(x1_data, y_data[cs1], variable, cs1)
        beta[cs1] = qrsol.qrSolution()[0]
        beta_var[cs1] = qrsol.qrSolution()[1]
        #exclude variable with no causal information
        colnames = qrsol.noCausalVar()
        filt_colnames.extend(colnames)

    #set qr priors to the coefficient boundary if it's outside the range
    QrtoCoefBound = qrFixCoefBound(beta, boundaries)
    beta = QrtoCoefBound.fixQrToCoefBound()

    colnames = filt_colnames
    x1_data_ = x1_data[colnames] # x1_data_ eliminated structured array
                                # x1_data full structured array
                                # X_ empty full structured array

    X_[colnames] = x1_data_
    y_ = Y.T.to_numpy().reshape(1, len(Y.columns), len(Y))  #y_ correct shape for dlm, array
    myDLM = dlm(y_)

    #######model configuration#######
    #intercept
    trd1 = trend(degree=len(Y.columns)-1, name='var_V1', 
                       boundaries=np.array(boundaries['V1']),
                       discount=discount['V1'],
                       w=0)

    #set priors
    mean_priors['V1'] = get_component_priors(trd1, xsec, mod, var, 'V1', beta, beta_var, cs)[0]
    cov_priors['V1'] = get_component_priors(trd1, xsec, mod, var, 'V1', beta, beta_var, cs)[1]
    myDLM = myDLM + trd1   #fit intercept

    #dynamic variables
    variables = list(dict.fromkeys([i.split('~')[1] for i in list(x1_data.dtype.names)]).keys())[1:] #exclude intercept
    for var1 in variables:#:3]:
        exp_match = re.compile(r'~'+var1+'$')
        columns = list(filter(exp_match.search, X.columns))
        x_tr = np.array([[i.tolist() for i in np.array(X_[columns])[j]] for j in range(len(X_))])
        x_ = np.array([x_tr[:,j] for j in range(len(columns))]).reshape(1,len(columns), len(X_))
        dyn = dynamic(
                        features=x_, name = 'var_'+str(var1), 
                        boundaries=boundaries[var1],
                        discount=discount[var1]
                        )
        
        #set priors
        mean_priors[var1] = get_component_priors(dyn, xsec, mod, var, var1, beta, beta_var, cs)[0]
        cov_priors[var1] = get_component_priors(dyn, xsec, mod, var, var1, beta, beta_var, cs)[1]
        myDLM = myDLM + dyn

    #######fit model######
    #componets evolving jointly
    #myDLM.evolveMode(evoType='dependent')
    tuned_discounts = []
    if autotune:
        #myDLM.fitForwardFilter()
        myDLM.tune(maxit=iterations)
        myDLM.fitBackwardSmoother()
        tuned_discounts = myDLM._getDiscounts()
    else:
        myDLM.fit()

    #######backwardSmoother: coefficients and covariance#######
    b = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))
    covb = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))

    coeff = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))
    covcoeff = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))

    for v in var:
        solution_backward = export_coefficients(v, cs, 'backwardSmoother', myDLM, b, covb, coeff, covcoeff)

    #######forwardFilter: coefficients and covariance#######
    b = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))
    covb = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))

    coeff = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))
    covcoeff = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))

    for v in var:
        solution_forward = export_coefficients(v, cs, 'forwardFilter', myDLM, b, covb, coeff, covcoeff)

    #plotting
    if len(cs)==1:
        if save_cs or save_var:
            myDLM.options.separatePlot = False
            myDLM.turnOff('filtered plot')
            myDLM.turnOff('predict plot')
            myDLM.plot(n=0, save=output+'/charts/cs/'+cs[0]+'.png')
            if save_var:
                for v in var:
                    myDLM.turnOn('filtered plot')
                    myDLM.plot(name='var_'+v, n=0, save=output+'/charts/cs_var/'+cs[0]+'_'+v+'.png')
                    myDLM.plotCoef(name='var_'+v, n=0, save=output+'/charts/cs_coef/'+cs[0]+'_'+v+'.png')
    else:
        if save_cs or save_var:
            for cs1 in cs:
                myDLM.turnOff('filtered plot')
                myDLM.turnOff('predict plot')
                myDLM.plot(n=cs.index(cs1), save=output+'/charts/cs/'+cs1+'.png')
                if save_var:
                    for v in var:
                        myDLM.turnOn('filtered plot')
                        myDLM.plot(name='var_'+v, n=cs.index(cs1), save=output+'/charts/cs_var/'+cs1+'_'+v+'.png')
                        myDLM.plotCoef(name='var_'+v, n=cs.index(cs1), save=output+'/charts/cs_coef/'+cs1+'_'+v+'.png')

    #######export results#######
    ols_priors = export_olspriors(beta, beta_var, var, cs)
    model_priors = export_modelpriors(mean_priors, cov_priors, var, cs)

    res_table = pd.DataFrame(myDLM.kalman1, columns = ['y', 'pred.state', 'pred.sysVar', 'pred.obs', 'pred.obsVar', 
                                        'err', 'correction','noiseVar', 'state', 'sysVar', 'obs', 'obsVar', 'bound'])


    return [res_table, pd.DataFrame(x1_data), pd.DataFrame(y_data), colnames, pd.DataFrame(solution_backward[0]), pd.DataFrame(solution_backward[1]), 
        norm_factor,
        pd.DataFrame(solution_backward[2], columns = list(solution_backward[2].dtype.names)),
        pd.DataFrame(solution_backward[3], columns = list(solution_backward[3].dtype.names)),
        pd.DataFrame(np.column_stack([np.mean(solution_backward[2].view((float, len(solution_backward[2].dtype.names))), axis=0), np.array(solution_backward[2].dtype.names)])),
        pd.DataFrame(np.column_stack([np.std(solution_backward[2].view((float, len(solution_backward[2].dtype.names))), axis=0), np.array(solution_backward[2].dtype.names)])),
        pd.DataFrame(myDLM.getMean(filterType='backwardSmoother'), columns = [i for i in cs]),
        export_intervals(myDLM.getInterval(filterType='backwardSmoother'), cs),
        pd.DataFrame(myDLM.getResidual(filterType='backwardSmoother'), columns = [i for i in cs]),
        ols_priors,
        model_priors,
        pd.DataFrame(solution_forward[2], columns = list(solution_forward[2].dtype.names)),
        pd.DataFrame(solution_forward[3], columns = list(solution_forward[3].dtype.names)),
        tuned_discounts]


# from ebpydlm import dlm, trend, seasonality, dynamic, autoReg, longSeason
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import re
# import time
# from ebpydlm.access.getters_to_df import export_olspriors, export_modelpriors, export_intervals, export_coefficients
# from ebpydlm.access.model_priors import get_component_priors

# def test_dlm(Y, X, output, mod, xsec, save_cs, save_var, autotune, iterations):
#     plt.ioff() #turn off interactive plotting

#     #normalize X
#     norm_factor = 1#X.abs().max()
#     #X = X/X.abs().max()
#     X = X.astype('float64')
#     X.fillna(0, inplace=True)
#     Y.fillna(0, inplace=True)

#     #normalize Y
#     # Y = Y/Y.abs().max()
#     # Y.fillna(0, inplace=True)
#     #boundaries = {k: (v1,v2) for k,v1,v2 in mod.to_numpy()}

#     y_data = np.array([tuple(i) for i in Y.to_numpy()], dtype = np.dtype(list(zip(Y.dtypes.index, Y.dtypes))))
#     x1_data = np.array([tuple(i) for i in X.to_numpy()], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))

#     #variable matrix
#     var_len = len(x1_data.dtype.names)
#     dates = y_data.shape[0]
#     X_ = np.array([tuple(i) for i in np.zeros(var_len*dates).reshape(dates, var_len)], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))
    
#     cs = list(dict.fromkeys([i.split('~')[0] for i in list(x1_data.dtype.names)]).keys())
#     var = list(dict.fromkeys([i.split('~')[1] for i in list(x1_data.dtype.names)]).keys())
#     boundaries = {k: (v1,v2) for k,v1,v2 in mod.iloc[:,0:3].to_numpy() if k in var}
#     discount = {k: v for k,v in mod.loc[:,['VariableName', 'Memory']].to_numpy() if k in var}

#     #preprocessing
#     #exclude variable with no causal information
#     dh_list = []
#     beta={}
#     beta_var = {}
#     filt_colnames = []
#     mean_priors = {}
#     cov_priors = {}

#     for cs1 in cs:
#         variable = [str(cs1)+'~'+i for i in var]
#         dh = {}
#         colnames = []
#         for i in variable:
#             ip1 = np.dot(x1_data[i].T, np.roll(x1_data[i],1))
#             ip2 = np.dot(x1_data[i].T, np.roll(x1_data[i],2))
#             ip3 = np.dot(x1_data[i].T, np.roll(x1_data[i],3))
#             diff = [ip1, ip2, ip3] - \
#                     np.roll([ip1, ip2, ip3], 1)
#             #if not any(np.equal(diff, np.rint(diff))):
#             dh[i] = diff[1:]
#             #check linearly dependent columns
#         mm = np.array([tuple(i) for i in x1_data[variable]])
#         q, r = np.linalg.qr(mm)#[:,1:])
#         #qr coefficients
#         beta[cs1] = np.linalg.pinv(r).dot(q.T).dot(y_data[cs1])
#         #qr coefficients variance
#         sigma = (1/(mm.shape[0]-mm.shape[1]-1)) * np.sum((y_data[cs1] - np.dot(np.dot(q, r), beta[cs1]))**2)
#         beta_var[cs1] = np.diag(np.linalg.pinv(mm.T.dot(mm)) * sigma**2)

#         colnames = list(dh.keys())#[l1 for l1 in np.array(x1_data[variable].dtype.names)[np.isclose(np.linalg.eigvals(r), np.zeros(r.shape[0])) == False] if l1 in list(dh.keys())]
#         filt_colnames.extend(colnames)
#         dh_list.append(dh)

#     #set qr priors to the coefficient boundary if it's outside the range
#     beta_arr = np.array([tuple(i) for i in beta.values()]).T
#     coef_min = [i[0] for i in boundaries.values()]
#     coef_max = [i[1] for i in boundaries.values()]
#     for i in list(beta.keys()):
#         beta[i] = np.array([np.maximum(np.minimum(beta_arr[i,:], coef_max[i]), coef_min[i]) for i in range(beta_arr.shape[0])]).T[list(beta.keys()).index(i)]

#     colnames = filt_colnames
#     x1_data_ = x1_data[colnames] # x1_data_ eliminated structured array
#                                 # x1_data full structured array
#                                 # X_ empty full structured array

#     #reintroduce intercept
#     intercept_match = re.compile(r'~V1')
#     intercept_col = list(filter(intercept_match.search, X.columns))
#     X_[colnames] = x1_data_
#     #X_[intercept_col] = 1

#     y_ = Y.T.to_numpy().reshape(1, len(Y.columns), len(Y))  #y_ correct shape for dlm, array
#     myDLM = dlm(y_)

#     #intercept
#     trd1 = trend(degree=len(Y.columns)-1, name='var_V1', 
#                        boundaries=np.array(boundaries['V1']),
#                        discount=discount['V1'],
#                        w=0)

#     #set priors
#     mean_priors['V1'] = get_component_priors(trd1, xsec, mod, var, 'V1', beta, beta_var, cs)[0]
#     cov_priors['V1'] = get_component_priors(trd1, xsec, mod, var, 'V1', beta, beta_var, cs)[1]
#     myDLM = myDLM + trd1   #fit intercept

#     # #set prior mean
#     # if prior_mean['V1']=='NA':
#     #     trd1.createMeanPrior(mean=np.matrix([beta[k][0] for k in beta.keys()]).T)
#     # else:
#     #     trd1.createMeanPrior(mean=np.matrix([[float(prior_mean['V1'])] * len(cs)]).T)
#     # #set prior covariance
#     # if prior_cov['V1']=='NA' and prior_mean['V1']=='NA':
#     #     trd1.createCovPrior(cov=np.matrix([beta_var[k][0] for k in beta_var.keys()]).T)
#     # elif prior_cov['V1']=='NA' and prior_mean['V1']!='NA':
#     #     trd1.createCovPrior(cov=100)
#     # else:
#     #     trd1.createCovPrior(cov=np.matrix([[float(prior_cov['V1'])] * len(cs)]).T)
    
#     # #save priors
#     # mean_priors['V1'] = trd1.meanPrior
#     # cov_priors['V1'] = trd1.covPrior

#     # myDLM = myDLM + trd1   #fit intercept
#     #myDLM = myDLM + seasonality(period=30, cs=len(cs), boundaries=np.array([-np.inf, np.inf]), discount=0.99, name='seasonality') #add seasonality

#     variables = list(dict.fromkeys([i.split('~')[1] for i in list(x1_data.dtype.names)]).keys())[1:] #exclude intercept
#     for var1 in variables:#:3]:
#         exp_match = re.compile(r'~'+var1+'$')
#         columns = list(filter(exp_match.search, X.columns))
#         x_tr = np.array([[i.tolist() for i in np.array(X_[columns])[j]] for j in range(len(X_))])
#         x_ = np.array([x_tr[:,j] for j in range(len(columns))]).reshape(1,len(columns), len(X_))
#         dyn = dynamic(
#                         features=x_, name = 'var_'+str(var1), 
#                         boundaries=boundaries[var1],
#                         discount=discount[var1],
#                         w=0)
        
#         #set priors
#         mean_priors[var1] = get_component_priors(dyn, xsec, mod, var, var1, beta, beta_var, cs)[0]
#         cov_priors[var1] = get_component_priors(dyn, xsec, mod, var, var1, beta, beta_var, cs)[1]
#         myDLM = myDLM + dyn
#         # #set prior mean
#         # if prior_mean[var1]=='NA':
#         #     dyn.createMeanPrior(mean=np.matrix(list(beta.values()))[:,variables.index(var1)+1])
#         # else:
#         #     dyn.createMeanPrior(mean=np.matrix([[float(prior_mean[var1])] * len(cs)]).T)
#         # #set prior covariance
#         # if prior_cov[var1]=='NA' and prior_mean[var1]=='NA':
#         #     dyn.createCovPrior(scale=np.max(np.matrix(list(beta_var.values()))[:,variables.index(var1)+1]))
#         # elif prior_cov[var1]=='NA' and prior_mean[var1]!='NA':
#         #     dyn.createCovPrior(scale=100)
#         # else:
#         #     dyn.createCovPrior(scale=float(prior_cov[var1]))

#         # #save priors
#         # mean_priors[var1] = dyn.meanPrior
#         # cov_priors[var1] = dyn.covPrior
#         # myDLM = myDLM + dyn

#     #componets evolving jointly
#     #myDLM.evolveMode(evoType='dependent')

#     tuned_discounts = []
#     if autotune:
#         #myDLM.fitForwardFilter()
#         myDLM.tune(maxit=iterations)
#         myDLM.fitBackwardSmoother()
#         tuned_discounts = myDLM._getDiscounts()
#     else:
#         myDLM.fit()


#     res_table = pd.DataFrame(myDLM.kalman1, columns = ['y', 'pred.state', 'pred.sysVar', 'pred.obs', 'pred.obsVar', 
#                                             'err', 'correction','noiseVar', 'state', 'sysVar', 'obs', 'obsVar', 'bound'])


#     #define matrices
#     #coefficients and covariance
#     b = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))
#     covb = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))

#     coeff = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))
#     covcoeff = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))

#     for v in var:
#         solution_backward = export_coefficients(v, cs, 'backwardSmoother', myDLM, b, covb, coeff, covcoeff)

#     #define matrices
#     #coefficients and covariance
#     b = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))
#     covb = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))

#     coeff = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))
#     covcoeff = np.array([tuple(i) for i in np.zeros(len(X)*len(X.columns)).reshape(len(X),len(X.columns))], dtype = np.dtype(list(zip(X.dtypes.index, X.dtypes))))

#     for v in var:
#         solution_forward = export_coefficients(v, cs, 'forwardFilter', myDLM, b, covb, coeff, covcoeff)


#         # #transformed variables
#         # transf_v = pd.DataFrame([i[0] for i in myDLM.getMean(name='var_'+str(v), filterType='backwardSmoother')], columns = [i+'~'+str(v) for i in cs])
#         # transf_arr = np.array([tuple(i) for i in transf_v.to_numpy()], 
#         #                             dtype = np.dtype(list(zip(transf_v.dtypes.index, transf_v.dtypes))))
#         # b[[i+'~'+str(v) for i in cs]] = transf_arr[[i+'~'+str(v) for i in cs]]
#         # transf_v_cov = pd.DataFrame([[i] * len(cs) for i in myDLM.getVar(name='var_'+str(v), filterType='backwardSmoother')], 
#         #                             columns = [i+'~'+str(v) for i in cs])
#         # transf_arr_cov = np.array([tuple(i) for i in transf_v_cov.to_numpy()], 
#         #                             dtype = np.dtype(list(zip(transf_v_cov.dtypes.index, transf_v_cov.dtypes))))
#         # covb[[i+'~'+str(v) for i in cs]] = transf_arr_cov[[i+'~'+str(v) for i in cs]]

#         # #coefficients
#         # transf_cv = pd.DataFrame([i for i in myDLM.getLatentState(name='var_'+str(v), filterType='backwardSmoother')], columns = [i+'~'+str(v) for i in cs])
#         # ##########################??? do we need it ???
#         # #this is to account for sign violations after backwardSmoother
#         # # transf_cv = pd.DataFrame([i for i in np.maximum(np.minimum(myDLM.getLatentState(name='var_'+str(v), filterType='backwardSmoother'),
#         # #                                                 boundaries[v][1]), boundaries[v][0])], columns = [i+'~'+str(v) for i in cs])
#         # transf_carr = np.array([tuple(i) for i in transf_cv.to_numpy()], 
#         #                             dtype = np.dtype(list(zip(transf_cv.dtypes.index, transf_cv.dtypes))))
#         # coeff[[i+'~'+str(v) for i in cs]] = transf_carr[[i+'~'+str(v) for i in cs]]
        
#         # transf_cv_cov = pd.DataFrame([[i] * len(cs) for i in myDLM.getLatentCov(name='var_'+str(v), filterType='backwardSmoother')], columns = [i+'~'+str(v) for i in cs])
#         # transf_carr_cov = np.array([tuple(i) for i in transf_cv_cov.to_numpy()], 
#         #                             dtype = np.dtype(list(zip(transf_cv_cov.dtypes.index, transf_cv_cov.dtypes))))
#         # covcoeff[[i+'~'+str(v) for i in cs]] = transf_carr_cov[[i+'~'+str(v) for i in cs]]

#     #plotting
#     if len(cs)==1:
#         if save_cs or save_var:
#             myDLM.options.separatePlot = False
#             myDLM.turnOff('filtered plot')
#             myDLM.turnOff('predict plot')
#             myDLM.plot(n=0, save=output+'/charts/cs/'+cs[0]+'.png')
#             if save_var:
#                 for v in var:
#                     myDLM.turnOn('filtered plot')
#                     myDLM.plot(name='var_'+v, n=0, save=output+'/charts/cs_var/'+cs[0]+'_'+v+'.png')
#                     myDLM.plotCoef(name='var_'+v, n=0, save=output+'/charts/cs_coef/'+cs[0]+'_'+v+'.png')
#     else:
#         if save_cs or save_var:
#             for cs1 in cs:
#                 myDLM.turnOff('filtered plot')
#                 myDLM.turnOff('predict plot')
#                 myDLM.plot(n=cs.index(cs1), save=output+'/charts/cs/'+cs1+'.png')
#                 if save_var:
#                     for v in var:
#                         myDLM.turnOn('filtered plot')
#                         myDLM.plot(name='var_'+v, n=cs.index(cs1), save=output+'/charts/cs_var/'+cs1+'_'+v+'.png')
#                         myDLM.plotCoef(name='var_'+v, n=cs.index(cs1), save=output+'/charts/cs_coef/'+cs1+'_'+v+'.png')

#     #export priors
#     ols_priors = export_olspriors(beta, beta_var, var, cs)
#     #ols_priors.to_csv(output+'/ols_priors'+'sysnum_'+str(xsec['SystemNumber'][xsec['CrossSection']==cs[0]][0])+'.csv', index=False, header=None)

#     model_priors = export_modelpriors(mean_priors, cov_priors, var, cs)
#     #model_priors.to_csv(output+'/model_priors'+'sysnum_'+str(xsec['SystemNumber'][xsec['CrossSection']==cs[0]][0])+'.csv', index=False)

#     return [res_table, pd.DataFrame(x1_data), pd.DataFrame(y_data), colnames, pd.DataFrame(solution_backward[0]), pd.DataFrame(solution_backward[1]), 
#         norm_factor,
#         pd.DataFrame(solution_backward[2], columns = list(solution_backward[2].dtype.names)),
#         pd.DataFrame(solution_backward[3], columns = list(solution_backward[3].dtype.names)),
#         pd.DataFrame(np.column_stack([np.mean(solution_backward[2].view((float, len(solution_backward[2].dtype.names))), axis=0), np.array(solution_backward[2].dtype.names)])),
#         pd.DataFrame(np.column_stack([np.std(solution_backward[2].view((float, len(solution_backward[2].dtype.names))), axis=0), np.array(solution_backward[2].dtype.names)])),
#         pd.DataFrame(myDLM.getMean(filterType='backwardSmoother'), columns = [i for i in cs]),
#         export_intervals(myDLM.getInterval(filterType='backwardSmoother'), cs),
#         pd.DataFrame(myDLM.getResidual(filterType='backwardSmoother'), columns = [i for i in cs]),
#         ols_priors,
#         model_priors,
#         pd.DataFrame(solution_forward[2], columns = list(solution_forward[2].dtype.names)),
#         pd.DataFrame(solution_forward[3], columns = list(solution_forward[3].dtype.names)),
#         tuned_discounts]
