import os
import json
import matplotlib.pyplot as plt
import math
import numpy as np


# List of dataset names
datasetList = [
    'GX010071-1', 'GX010220-1', 'GX010228-1', 'GX010230-1', 'GX010231-1',
    'GX010241-1', 'GX010250-1', 'GX010266-1', 'GX010290-1', 'GX010291-1',
    'GX010303-1', 'GX010307-1', 'GX010315-1', 'GX010321-1', 'GX010322-1',
    'GX010327-1', 'GX010335-1', 'GX010336-1', 'GX010337-1'
]

# Path to the dataset directory
datasetPath = os.path.join('D:/STMD_Dataset', 'RIST')

# Function to calculate the projected length of the bbox along the motion vector
def calculate_projectedLength(bbox, motionVector):
    # Extract width and height from the bbox
    width, height = bbox[2], bbox[3]
    
    # Normalize the motion vector to get the unit direction
    motionDirection = np.array(motionVector)
    motionDirection = motionDirection / np.linalg.norm(motionDirection) if np.linalg.norm(motionDirection) > 0 else [0, 0]
    
    if motionDirection[0] == 0:
        projectedLength = width
    elif motionDirection[1] == 0:
        projectedLength = height
    else:
        if width/height > abs(motionDirection[0] / motionDirection[1]):
            projectedLength = height / abs(motionDirection[1])
        else:
            projectedLength = width / abs(motionDirection[0])
    # # Calculate the projection of width and height along the motion direction
    # # The length of the projection is the absolute dot product of the vector (width, 0) and motion direction
    # width_projection = np.abs(np.dot(motionDirection, np.array([width, 0])))
    # height_projection = np.abs(np.dot(motionDirection, np.array([0, height])))
    
    # # The total length along the motion vector is the sum of these projections
    # projectedLength = math.sqrt(width_projection**2 + height_projection**2)
    return projectedLength

def calculate_para(isShow = False):
    # Dictionary to store the results for each dataset
    recordPara = {}

    # Iterate through each dataset
    for datasetName in datasetList:
        # Construct the file path to the JSON annotation file
        filePath = os.path.join(datasetPath, datasetName, datasetName + '_annotation.json')
        
        # Read the JSON file
        with open(filePath, 'r') as f:
            data = json.load(f)
        
            # Initialize variables to store results
            motionVelocities = []  # Store motion velocity for each frame
            projectedLengths = []  # Store projected lengths for each frame
            
            # Iterate over each frame in the dataset
            for frame_data in data['frames']:
                # Extract the motion_vector and bbox information for the current frame
                motionVector = frame_data['objects']['motion_vector']  # Assuming there is only one object per frame
                bbox = frame_data['objects']['bbox']  # Assume bbox is in [x, y, w, h] format
                
                if len(motionVector):
                    motionVelocities.append(np.linalg.norm(motionVector))
                    # Calculate the projected length of the target along the motion direction
                    projectedLength = calculate_projectedLength(bbox, motionVector)
                    projectedLengths.append(projectedLength)
        

        motionVelocities = np.array(motionVelocities)  # Motion velocities
        projectedLengths = np.array(projectedLengths)  # Projected lengths

        # Calculate the ratio of projected length to motion velocity
        velocity2LengthRatio = np.where(motionVelocities < 1e-4, 0, projectedLengths / motionVelocities)

        # Calculate the mean and standard deviation of the velocity-to-length ratio
        meanVal = np.mean(velocity2LengthRatio)
        stdVal = np.std(velocity2LengthRatio)
        # Remove outliers using 3-sigma rule
        filteredData = velocity2LengthRatio[(velocity2LengthRatio >= meanVal - 3 * stdVal) & (velocity2LengthRatio <= meanVal + 3 * stdVal)]
        # Calculate the robust mean (mean of the filtered data)
        robustMean = np.mean(filteredData) / 2

        # Store the results for the current dataset
        recordPara[datasetName] = {'mean': meanVal, 
                                'robustMean': robustMean,
                                }

        # Create a figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # 3 rows and 1 column of subplots

        # Plot the motion velocities
        axs[0].plot(motionVelocities, label='Motion Velocities', marker='o')
        axs[0].set_title('Motion Velocities')
        axs[0].set_xlabel('Frame Number')
        axs[0].set_ylabel('Pixel/Frame')  # Set y-axis label
        axs[0].grid(True)
        axs[0].legend()

        # Plot the projected lengths
        axs[1].plot(projectedLengths, label='Projected Lengths', marker='x')
        axs[1].set_title('Projected Lengths')
        axs[1].set_xlabel('Frame Number')
        axs[1].set_ylabel('Pixel')  # Set y-axis label
        axs[1].grid(True)
        axs[1].legend()

        # Plot the ratio of projected length to motion velocity
        axs[2].plot(velocity2LengthRatio, label='Projected Length / Motion Velocity', marker='s')
        axs[2].set_title('Projected Length / Motion Velocity')
        axs[2].set_xlabel('Frame Number')
        axs[2].set_ylabel('Ratio')  # Set y-axis label
        axs[2].grid(True)
        # Add horizontal lines for mean and robustMean
        axs[2].axhline(meanVal, color='red', linestyle='--', label='Mean')  # Red dashed line for mean
        axs[2].axhline(robustMean, color='blue', linestyle='--', label='Robust Mean')  # Blue dashed line for robustMean
        axs[2].legend()

        # Add a title for the entire figure
        fig.suptitle(f'{datasetName}', fontsize=16)

        # Adjust the layout of the subplots
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Leave space at the top for the title


    # Create the file path to save the results
    saveFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasetPara.json')
    with open(saveFilePath, 'w') as json_file:
        json.dump(recordPara, json_file, indent=4)

    
    # 保存所有 figure 到文件
    # for fig_num in plt.get_fignums():
    #     fig = plt.figure(fig_num)
    #     fig.savefig(f'figure_{fig_num}.png')  # 保存为 PNG 格式
    # plt.close('all')

    if isShow:
        plt.show()


if __name__ == '__main__':
    # calculate_para(0)
    calculate_para(1)
    