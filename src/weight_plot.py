import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import argparse
import csv

def main(args):
    weight_array = np.load(args.input + 'sens_log.npy')
    print(weight_array.shape)

    data = np.array(csv_read(args.input + 'data.csv'))

    fig, ax = plt.subplots()
    ax.set_ylim([0, 1])
    artists = []
    wave_length = np.linspace(400, 700, weight_array.shape[1])
    print(wave_length.shape)
    for i in range(weight_array.shape[0]):
        im = ax.plot(wave_length, weight_array[i, :, 0],'r', wave_length, weight_array[i, :, 1],'g', wave_length, weight_array[i, :, 2],'b')
        title = ax.text(0.5, 1.01, 'epoch={}, val_cpsnr={:.2f}'.format(data[i+1,1], float(data[i+1,6])),
                    ha='center', va='bottom',
                    transform=ax.transAxes, fontsize='large')
        artists.append(im + [title])
    anim = ArtistAnimation(fig, artists, interval=100)
    # fig.show()
    fig.suptitle('noise 0')
    anim.save(args.output + 'anim.gif', writer='imagemagick', fps=10)
    anim.save(args.output + 'anim.mp4', writer="ffmpeg")

def csv_read(file_name):
    with open(file_name) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
    return l

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input', dest='input', help='make input directory')
    parser.add_argument('--output', dest='output', help='make output directory')
    parser.add_argument('--noise', dest='noise', default='0',  help='make output directory')
    args = parser.parse_args()
    main(args)