from inference_STMD_in_RIST import main_infer_STMD
from evaluation_STMD_in_RIST import main_evalu_STMD


if __name__ == '__main__':

    from datetime import datetime
    
    print("inference start time:", datetime.now())

    main_infer_STMD()

    print("evaluation start time:", datetime.now())

    main_evalu_STMD()

    print("finished time:", datetime.now())




