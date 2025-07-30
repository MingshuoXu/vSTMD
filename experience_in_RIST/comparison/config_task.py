import os
import sys

ProjectPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ProjectPath)
import stmd_package_path

apiOpticFlowPth = os.path.join(ProjectPath, 'comparison_models')
sys.path.append(apiOpticFlowPth)


# STMD models in task
stmdModelList = (
    'ESTMD', 'DSTMD', 'FracSTMD',  # some backbone models
    'STMDPlus',  # with contrast patheway
    'ApgSTMD',  # with attention and prediction mechanism
    'FeedbackSTMD', 'FSTMD',  # with Feedback pathway
    'vSTMD', 'vSTMD_F'  # proposed model
    )


# optical flow models in task
opticflowModelList = ('RAFT', 'SEA_RAFT',
                      'MemFlow', 'StreamFlow', 'DpFlow', 'FlowDiffuser') 

directionalStmdList = ('DSTMD', 'STMDPlus', 'ApgSTMD', 'vSTMD', 'vSTMD_F') 

# dataset information
datasetInfo = {
    'GX010071-1': list(range(1300)),
    'GX010220-1': list(range(1300)),
    'GX010228-1': list(range(1300)),
    'GX010230-1': list(range(2400)),
    'GX010231-1': list(range(2400)),
    'GX010241-1': list(range(3600)),
    'GX010250-1': list(range(2000)),
    'GX010266-1': list(range(2400)),
    'GX010290-1': list(range(1300)),
    'GX010291-1': list(range(1300)),
    'GX010303-1': list(range(2400)),
    'GX010307-1': list(range(1000)),
    'GX010315-1': list(range(1000)),
    'GX010321-1': list(range(1000)),
    'GX010322-1': list(range(1300)),
    'GX010327-1': list(range(900)),
    'GX010335-1': list(range(1300)),
    'GX010336-1': list(range(1000)),
    'GX010337-1': list(range(700)),
}

# dataset path
ristDatasetPath = os.path.join('D:/', 'STMD_Dataset', 'RIST')

# model inference output folder
modelOptFolder = os.path.join('D:/', 'STMD_Dataset', 'evaluate_RIST')

# evaluation result folder
evaluateResultFolder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'result', 'RIST_240Hz',
    )



