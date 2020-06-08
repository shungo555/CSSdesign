import os
import sys


def output_config(file_name, train_data_num, smoothness, nl_max255, batch_size, patch_size, skip_mixed, Mcc, wide_color):
    strings = [
        '-----config-----',
        'train data number: ' + str(train_data_num),
        'smoothness: ' + str(smoothness),
        'max noise level (255): ' + str(nl_max255),
        'training batch size: ' + str(batch_size),
        'training patch size: ' + str(patch_size),
        'skip_mixed: ' + str(skip_mixed),
        'Mcc: ' + str(Mcc),
        'wide_color: ' + str(wide_color),
    ]
    with open(file_name, mode='w', encoding="utf-8") as f:
        f.write("\n".join(strings))
        f.write("\n")
    f.close
    print("saved config")