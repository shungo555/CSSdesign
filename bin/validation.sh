for camera in Canon20D Canon300D NikonD40 OlympusE-PL2
do
    for noise in 0 10 20 30 
    do
        python ../src/mytrain_numpy.py --gpu --epochs 100 --noise $noise --output ../../dataset/cave_hsi/val_data/srgb/$camera --gt srgb --dataset cave --validation --camera $camera
    done
done
read -p "finish"