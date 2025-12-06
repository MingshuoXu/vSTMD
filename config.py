import os
import sys

import platform

if platform.system() == 'Windows':
    stmd_package_path = os.path.join('D:/', '11_Code', 'Small-Target-Motion-Detectors', 
                            'python')
elif platform.system() == 'Linux':
    stmd_package_path = os.path.join('/mnt', 'windows_D', '11_Code', 
                            'Small-Target-Motion-Detectors', 'python')
# Add the path to the package containing the models
sys.path.append(stmd_package_path)

# dataset path
ristDatasetPath = os.path.join('D:/', 'STMD_Dataset', 'RIST')


# print('Succesfully add path: ''%s''\n' %stmd_package_path)