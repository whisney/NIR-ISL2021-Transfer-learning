import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle

def split_data(data_dir, save_dir):
    CASIA_Iris_Africa_all = os.listdir(os.path.join(data_dir, r'CASIA-Iris-Africa\train\image'))

    CASIA_Iris_Asia_distance_all = os.listdir(os.path.join(data_dir, r'CASIA-Iris-Asia\CASIA-distance\train\image'))
    CASIA_Iris_Asia_Complex_Occlusion_all = os.listdir(os.path.join(
        data_dir, r'CASIA-Iris-Asia\CASIA-Iris-Complex\Occlusion\train\image'))
    CASIA_Iris_Asia_Complex_Off_angle_all = os.listdir(os.path.join(
        data_dir, r'CASIA-Iris-Asia\CASIA-Iris-Complex\Off_angle\train\image'))

    CASIA_Iris_M1_all = os.listdir(os.path.join(data_dir, r'CASIA-Iris-Mobile-V1.0\train\image'))

    for i, name in enumerate(CASIA_Iris_Africa_all):
        CASIA_Iris_Africa_all[i] = 'CASIA-Iris-Africa/train/' + name

    for i, name in enumerate(CASIA_Iris_Asia_distance_all):
        CASIA_Iris_Asia_distance_all[i] = 'CASIA-Iris-Asia/CASIA-distance/train/' + name
    for i, name in enumerate(CASIA_Iris_Asia_Complex_Occlusion_all):
        CASIA_Iris_Asia_Complex_Occlusion_all[i] = 'CASIA-Iris-Asia/CASIA-Iris-Complex/Occlusion/train/' + name
    for i, name in enumerate(CASIA_Iris_Asia_Complex_Off_angle_all):
        CASIA_Iris_Asia_Complex_Off_angle_all[i] = 'CASIA-Iris-Asia/CASIA-Iris-Complex/Off_angle/train/' + name

    for i, name in enumerate(CASIA_Iris_M1_all):
        CASIA_Iris_M1_all[i] = 'CASIA-Iris-Mobile-V1.0/train/' + name

    trainval_test_Kfold = KFold(n_splits=5, shuffle=True)

    CASIA_Iris_Africa_split = []
    for i, (trainval_index, test_index) in enumerate(trainval_test_Kfold.split(CASIA_Iris_Africa_all)):
        trainval_list = []
        test_list = []
        for id in trainval_index:
            trainval_list.append(CASIA_Iris_Africa_all[id])
        for id in test_index:
            test_list.append(CASIA_Iris_Africa_all[id])
        train_list, val_list = train_test_split(trainval_list, test_size=0.125, random_state=10)
        CASIA_Iris_Africa_split.append({'train': train_list, 'val': val_list, 'test': test_list})

    CASIA_Iris_Asia_distance_split = []
    for i, (trainval_index, test_index) in enumerate(trainval_test_Kfold.split(CASIA_Iris_Asia_distance_all)):
        trainval_list = []
        test_list = []
        for id in trainval_index:
            trainval_list.append(CASIA_Iris_Asia_distance_all[id])
        for id in test_index:
            test_list.append(CASIA_Iris_Asia_distance_all[id])
        train_list, val_list = train_test_split(trainval_list, test_size=0.125, random_state=10)
        CASIA_Iris_Asia_distance_split.append({'train': train_list, 'val': val_list, 'test': test_list})

    CASIA_Iris_Asia_Complex_Occlusion_split = []
    for i, (trainval_index, test_index) in enumerate(trainval_test_Kfold.split(CASIA_Iris_Asia_Complex_Occlusion_all)):
        trainval_list = []
        test_list = []
        for id in trainval_index:
            trainval_list.append(CASIA_Iris_Asia_Complex_Occlusion_all[id])
        for id in test_index:
            test_list.append(CASIA_Iris_Asia_Complex_Occlusion_all[id])
        train_list, val_list = train_test_split(trainval_list, test_size=0.125, random_state=10)
        CASIA_Iris_Asia_Complex_Occlusion_split.append({'train': train_list, 'val': val_list, 'test': test_list})

    CASIA_Iris_Asia_Complex_Off_angle_split = []
    for i, (trainval_index, test_index) in enumerate(trainval_test_Kfold.split(CASIA_Iris_Asia_Complex_Off_angle_all)):
        trainval_list = []
        test_list = []
        for id in trainval_index:
            trainval_list.append(CASIA_Iris_Asia_Complex_Off_angle_all[id])
        for id in test_index:
            test_list.append(CASIA_Iris_Asia_Complex_Off_angle_all[id])
        train_list, val_list = train_test_split(trainval_list, test_size=0.125, random_state=10)
        CASIA_Iris_Asia_Complex_Off_angle_split.append({'train': train_list, 'val': val_list, 'test': test_list})

    CASIA_Iris_M1_split = []
    for i, (trainval_index, test_index) in enumerate(trainval_test_Kfold.split(CASIA_Iris_M1_all)):
        trainval_list = []
        test_list = []
        for id in trainval_index:
            trainval_list.append(CASIA_Iris_M1_all[id])
        for id in test_index:
            test_list.append(CASIA_Iris_M1_all[id])
        train_list, val_list = train_test_split(trainval_list, test_size=0.125, random_state=10)
        CASIA_Iris_M1_split.append({'train': train_list, 'val': val_list, 'test': test_list})

    CASIA_Iris_Asia_split = []
    for i in range(5):
        train_list = CASIA_Iris_Asia_distance_split[i]['train'] + CASIA_Iris_Asia_Complex_Occlusion_split[i]['train'] + \
                     CASIA_Iris_Asia_Complex_Off_angle_split[i]['train']
        val_list = CASIA_Iris_Asia_distance_split[i]['val'] + CASIA_Iris_Asia_Complex_Occlusion_split[i]['val'] + \
                   CASIA_Iris_Asia_Complex_Off_angle_split[i]['val']
        test_list = CASIA_Iris_Asia_distance_split[i]['test'] + CASIA_Iris_Asia_Complex_Occlusion_split[i]['test'] + \
                    CASIA_Iris_Asia_Complex_Off_angle_split[i]['test']
        CASIA_Iris_Asia_split.append({'train': train_list, 'val': val_list, 'test': test_list})

    for i in range(5):
        print(i)
        print(len(CASIA_Iris_Africa_split[i]['train']), len(CASIA_Iris_Africa_split[i]['val']),
              len(CASIA_Iris_Africa_split[i]['test']))
        print(len(CASIA_Iris_Asia_split[i]['train']), len(CASIA_Iris_Asia_split[i]['val']),
              len(CASIA_Iris_Asia_split[i]['test']))
        print(len(CASIA_Iris_M1_split[i]['train']), len(CASIA_Iris_M1_split[i]['val']),
              len(CASIA_Iris_M1_split[i]['test']))

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'CASIA_Iris_Africa_split.pkl'), 'wb') as f:
        pickle.dump(CASIA_Iris_Africa_split, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dir, 'CASIA_Iris_Asia_split.pkl'), 'wb') as f:
        pickle.dump(CASIA_Iris_Asia_split, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dir, 'CASIA_Iris_M1_split.pkl'), 'wb') as f:
        pickle.dump(CASIA_Iris_M1_split, f, pickle.HIGHEST_PROTOCOL)

    CASIA_Iris_All = []

    for i in range(5):
        CASIA_Iris_All.append({'train': CASIA_Iris_Africa_split[i]['train'] + CASIA_Iris_Asia_split[i]['train'] +
                                CASIA_Iris_M1_split[i]['train'], 'val': CASIA_Iris_Africa_split[i]['val'] +
                                CASIA_Iris_Asia_split[i]['val'] + CASIA_Iris_M1_split[i]['val'], 'test': CASIA_Iris_Africa_split[i]['test'] +
                                CASIA_Iris_Asia_split[i]['test'] + CASIA_Iris_M1_split[i]['test']})

    with open(os.path.join(save_dir, 'CASIA_Iris_All_split.pkl'), 'wb') as f:
        pickle.dump(CASIA_Iris_All, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    data_dir = r'I:\public_data\NIR-ISL2021'
    save_dir = 'data'
    split_data(data_dir, save_dir)