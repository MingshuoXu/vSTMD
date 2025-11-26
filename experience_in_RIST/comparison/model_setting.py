import os
import json

# Get the path of the current directory
currentDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
filePath = os.path.join(currentDir, 'statistics_mean_para', 'datasetPara.json')

with open(filePath, 'r') as json_file:
    meanLen2Velocity = json.load(json_file)
    
model_param_calculations = {
    "paraESTMD": lambda tau: {
        'n3': max(round(tau / 2), 1),
        'tau3': max(tau, 1),
    },
    "paraDSTMD": lambda tau: {
        'n4': max(round(tau * 0.3), 1),
        'tau4': max(round(tau * 0.6), 1),
        'n5': max(round(tau * 0.5), 1),
        'tau5': max(tau, 1),
        'n6': max(round(tau * 0.8), 1),
        'tau6': max(round(tau * 1.6), 1),
    },
    "paraFracSTMD": lambda tau: {
        'tau1': max(tau, 1),
    },
    "paraSTMDNet": lambda tau: {
    },
    "paraSTMDPlus": lambda tau: {
        'n3': max(round(tau * 0.3), 1),
        'tau3': max(round(tau * 0.6), 1),
        'n4': max(round(tau * 0.5), 1),
        'tau4': max(tau, 1),
        'n5': max(round(tau * 0.8), 1),
        'tau5': max(round(tau * 1.6), 1),
    },
    "paraApgSTMD": lambda tau: {
        'n3': max(round(tau * 0.3), 1),
        'tau3': max(round(tau * 0.6), 1),
        'n4': max(round(tau * 0.5), 1),
        'tau4': max(tau, 1),
        'n5': max(round(tau * 0.8), 1),
        'tau5': max(round(tau * 1.6), 1),
    },
    "paraFeedbackSTMD": lambda tau: {
        'n3': max(round(tau / 2), 1),
        'tau3': max(tau, 1),
    },
    "paravSTMD": lambda tau: {
    },
    "paravSTMD_F": lambda tau: {
    },
}

# Output model parameters for each data set
def generate_model_parameters(meanLen2Velocity, model_param_calculations, mode='robustMean'):
    output = {}
    for dataset, val in meanLen2Velocity.items():
        output[dataset] = {}
        tau = val[mode]
        for model, calculation in model_param_calculations.items():
            output[dataset][model] = calculation(tau)
    return output

modelParas = generate_model_parameters(meanLen2Velocity, model_param_calculations, ) #'mean'
