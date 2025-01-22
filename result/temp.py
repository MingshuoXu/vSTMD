import os
import json

# 便利本文件夹下所有子文件夹中的文件，将文件名或文件夹名以Backbonev2开头的替换为STMDNet开头
def process_files(directory_path):
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for name in files:
            if 'Backbonev2' in name:
                old_file_path = os.path.join(root, name)
                new_file_path = os.path.join(root, name.replace('Backbonev2', 'STMDNet'))
                os.rename(old_file_path, new_file_path)
                print(f'将文件 {old_file_path} 重命名为 {new_file_path}')
            if 'FeedbackSTMDv2' in name:
                old_file_path = os.path.join(root, name)
                new_file_path = os.path.join(root, name.replace('FeedbackSTMDv2', 'STMDNetF'))
                os.rename(old_file_path, new_file_path)
                print(f'将文件 {old_file_path} 重命名为 {new_file_path}')
        
        for name in dirs:
            if 'Backbonev2' in name:
                old_dir_path = os.path.join(root, name)
                new_dir_path = os.path.join(root, name.replace('Backbonev2', 'STMDNet'))
                os.rename(old_dir_path, new_dir_path)
                print(f'将文件夹 {old_dir_path} 重命名为 {new_dir_path}')
            if 'FeedbackSTMDv2' in name:
                old_dir_path = os.path.join(root, name)
                new_dir_path = os.path.join(root, name.replace('FeedbackSTMDv2', 'STMDNetF'))
                os.rename(old_dir_path, new_dir_path)
                print(f'将文件夹 {old_dir_path} 重命名为 {new_dir_path}')




if __name__ == '__main__':
    directory_path = os.path.dirname(os.path.abspath(__file__))
    print(f'开始处理文件夹 {directory_path}')
    process_files(directory_path)

