from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
import glob, os
import time
from tqdm import tqdm
import pickle as pkl


def get_layers():
    file_dir = r'C:/Users/uhrich/TWIN_Share/Baujob_defekt/pkl/pkl_files'  # print job mixed quality
    files = os.listdir(file_dir)
    files.sort()

    # counter for pkl files
    p = 3
    temp_count = 0
    temp_count_prev = 0
    count_process_start = 0
    count_process_end = len(files) - 1
    temp_layer = []
    temp = []
    times = []
    laser_pos = []
    w = 0
    count = 0
    timer = 0
    count_layers = []
    TIME = []
    for count_process in tqdm(range(count_process_start, count_process_end)):  # while count_process <= 550:

        df = pd.read_pickle(file_dir + r'/' + files[count_process])

        # for 3 images per second run to 90, for 1 image per second run to 30
        for i in range(0, 90):  # range(0,90,3):
            Heat = (df['EM0_4'].iloc[i] - 1000) / 10
            Heat = Heat.astype(np.float32)
            if Heat.max() > 200:
                laser_position = np.unravel_index(np.argmax(Heat), Heat.shape)
            else:
                laser_position = None
            timer = timer + 1
            Heat_ix = df['EM0_4'].index[i]
            TIME.append(Heat_ix)
            if timer == 2090:
                if not os.path.isdir('Timestamp'):
                    os.mkdir('Timestamp')
                with open(f'Timestamp/Timer', 'wb') as f:
                    pkl.dump(TIME, f)

            # Heat_seg = Heat[117: 147, 90: 125]#segmentation of built 9 pyramids mixed quality
            #
            # if laser_position is not None and ( 117 <= laser_position[0] < 147 and 90 <= laser_position[1] < 125 ):
            #     laser_position += -np.array([117,90])
            # else:
            #     laser_position = None

            #Heat_seg = Heat[115: 145, 150: 180]  # pyramid mixed quality [115: 145, 150: 180] pyramid 7
            #Heat_seg = Heat[125: 155, 230: 260]  # pyramid mixed quality [115: 145, 150: 180] pyramid 4
            Heat_seg = Heat[140: 170, 125: 155]  # pyramid mixed quality [115: 145, 150: 180] pyramid 8


            if laser_position is not None and (115 <= laser_position[0] < 145 and 150 <= laser_position[1] < 180):
                laser_position += -np.array([115, 150])
            else:
                laser_position = None
            ############### Plot ##################################
            """fig = plt.figure(figsize=(5, 5))
            ax = plt.subplot()
            h = ax.imshow(Heat, interpolation='spline36', cmap='jet',
                        origin='lower', aspect='auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(h, cax=cax)
            plt.show()"""
            ########################################################

            if (temp_count > 0.5 * temp_count_prev):
                temp_count_prev = temp_count
            temp_count = sum(sum(k > 124.9 for k in Heat))

            if np.any(Heat_seg != 124.9) & (temp_count > 0.6 * temp_count_prev):
                # print(w)
                count = count + 1
                w = w + 1
                temp.append(Heat_seg)
                times.append(Heat_ix)
                laser_pos.append(laser_position)
                p = 1
            elif p == 1:
                if count >= 5:
                    count_layers.append((count_process, len(temp_layer)))
                    temp_layer.append([times, temp, laser_pos])
                    temp = []
                    times = []
                    laser_pos = []
                    count = 0
                    p = 2
        # count_process = count_process + 1
    return temp_layer, count_layers


if __name__ == '__main__':
    layers, count_layers = get_layers()
    for cl in count_layers:
        print(cl)
    if not os.path.isdir('pyramid_8'):
        os.mkdir('pyramid_8')
    for i, layer in tqdm(enumerate(layers), total=len(layers)):
        with open(f'pyramid_8/layer_{i}.pkl', 'wb') as f:
            pkl.dump(layer, f)
    with open(f'pyramid_8/file_layer_map.pkl', 'wb') as f:
        pkl.dump(count_layers, f)