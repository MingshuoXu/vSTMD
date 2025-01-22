import os

import json
import numpy as np
import matplotlib.pyplot as plt


def sparse_list_to_matrix(sparseList, shape=None):
    """
    Convert a sparse list back to a matrix using NumPy's vectorized operations.
    
    Parameters:
    - sparseList (list): A list of non-zero elements in the format [row, col, value].
    - shape (tuple, optional): Shape of the output matrix (rows, cols). If None, it will 
      infer the shape based on the maximum row and column indices in the sparse list.
    
    Returns:
    - numpy.ndarray: The reconstructed matrix.
    """

    # Convert sparseList to NumPy array for vectorized operations
    sparse_array = np.array(sparseList)
    
    # Extract rows, cols, and values
    rows = sparse_array[:, 1].astype(int)
    cols = sparse_array[:, 0].astype(int)
    values = sparse_array[:, 2]
    
    # If shape is not provided, infer it
    if shape is None:
        max_row = rows.max() + 1
        max_col = cols.max() + 1
        shape = (max_row, max_col)
    
    # Create the matrix and assign values using advanced indexing
    matrix = np.zeros(shape)
    matrix[rows, cols] = values
    
    return matrix


def main(modelName = 'ZBS', datasetName = 'GX010220-1'):
    
    modelOptFolder = os.path.join('D:\STMD_Dataset', 'evaluate_RIST')

    inferResultPath = os.path.join(modelOptFolder, datasetName, f'{modelName}_result.json')

    with open(inferResultPath, 'r') as f:
        _data = json.load(f)

    inferResult = _data['response']
    timePerImage = _data['runningtime'] / len(inferResult)

    # Set up the figure and axes once
    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros((270, 480)), cmap='gray', vmin=0, vmax=1)
    ax.set_title('')

    for j, res in enumerate(inferResult):
        if len(res) == 0:
            matrix = np.zeros((270, 480))
        else:
            matrix = sparse_list_to_matrix(res, (270, 480))
        
        # Update the image data instead of re-creating it
        img.set_data(matrix)
        ax.set_title(f'Frame {j}')

        plt.pause(0.01)

    plt.show()

if __name__ == '__main__':
    main(modelName = 'ZBS',
         datasetName = 'GX010220-1')
    
    '''
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
    '''

