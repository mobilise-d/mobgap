import warnings

import numpy as np

from hklee_algo_improved import hklee_algo_improved
from shin_algo_improved import shin_algo_improved
try:
    from cad2sec import cad2sec
except ModuleNotFoundError:
    print("The cad2sec module is not found, but the program will continue.")
try:
    from GSD_LowbackAcc import GSD_LowBackAcc
except ModuleNotFoundError:
    print("The GSD_LowbackAcc module is not found, but the program will continue.")

def CADENCE(DATA, fs, GS, algs):
    output_cadence = []
    alg_num = 2

    # CADENCE ESTIMATION
    Accelerometer = DATA[:, 0:3]
    print(Accelerometer)
    # Gyroscope = DATA[:, 3:6]  # Not used by algorithm
    BN = len(GS['Start'])
    startvec = np.zeros(BN, dtype=int)
    stopvec = np.zeros(BN, dtype=int)



    #############################################################
    finaltemp = np.array([])
    for i in range(BN):
        #try:
            output_cadence_dict = {'Start': [], 'End': [], 'cadSec': [], 'cadMean': [], 'cadSTD': [], 'steps': []}
            startvec[i] = int(np.floor(GS['Start'][i] * fs))
            if startvec[i] < 1:
                startvec[i] = 1

            stopvec[i] = int(np.floor(GS['End'][i] * fs))
            if stopvec[i] > len(Accelerometer):
                stopvec[i] = len(Accelerometer)

            chosenacc = Accelerometer[startvec[i]:stopvec[i], :]
            warningflag = 0
            totalDur = int(np.floor(GS['End'][i] - GS['Start'][i] + (2 / fs)))
            cadmat = np.zeros((totalDur, alg_num))
            cadinx = np.zeros(alg_num)

            if 'HKLee_Imp' in algs:  # 1
                #IC_HKLee_improved = hklee_algo_improved(chosenacc, fs)
                #cadence_HKLee_imp = cad2sec(IC_HKLee_improved, totalDur) / 2
                #cadmat[:, 0] = cadence_HKLee_imp[:totalDur]
                #cadinx[0] = 1

                #test just to run the algo
                IC_HKLee_improved = hklee_algo_improved(chosenacc, fs, 'norm')
                if len(IC_HKLee_improved) < len(cadmat):
                    cadmat = cadmat[:len(IC_HKLee_improved)]
                    cadmat[:, 1] = IC_HKLee_improved[:totalDur]
                    cadinx[1] = 1

            if 'Shin_Imp' in algs:  # 2
                #IC_Shin_improved = shin_algo_improved(chosenacc, fs)
                #cadence_Shin_improved = cad2sec(IC_Shin_improved, totalDur) / 2
                #cadmat[:, 1] = cadence_Shin_improved[:totalDur]
                #cadinx[1] = 1

                #test just to run the algo
                try:
                    IC_Shin_improved = shin_algo_improved(chosenacc, fs, 'norm')
                    if len(IC_Shin_improved) < len(cadmat):
                        cadmat = cadmat[:len(IC_Shin_improved)]
                        cadmat[:, 1] = IC_Shin_improved[:totalDur]
                        cadinx[1] = 1
                except Exception as e:
                    warnings.warn(str(e))
                    continue



            cadinx = cadinx.astype(bool)
            mycad = 120 * cadmat[:, cadinx]

            output_cadence_dict['Start'].append(startvec[i] / fs)
            output_cadence_dict['End'].append(stopvec[i] / fs)
            finaltemp = mycad.flatten()

            output_cadence_dict['cadSec'].append(finaltemp)

            if len(finaltemp) < totalDur:
                finaltemp = np.append(finaltemp, np.full(totalDur - len(finaltemp), finaltemp[-1]))
            elif len(finaltemp) > totalDur:
                finaltemp = finaltemp[:totalDur]

            finaltemp[finaltemp < 30] = 30
            finaltemp[finaltemp > 200] = 200
            output_cadence_dict['cadMean'].append(np.nanmean(finaltemp))
            output_cadence_dict['cadSTD'].append(np.nanstd(finaltemp))
            output_cadence_dict['steps'].append(int(np.round(np.nansum(finaltemp) / 60)))

            # except Exception as e:
            #    print(f"Error on analysis of walking sequence: {str(e)}")
            #    output_cadence_dict = {
            #        'cadSec': np.nan,
            #        'cadMean': np.nan,
            #        'cadSTD': np.nan,
            #        'steps': np.nan
            #    }
            #    continue

            output_cadence.append(output_cadence_dict)
            #print(f"Completed loop iteration {i + 1}")
            #print(output_cadence_dict)
    return output_cadence