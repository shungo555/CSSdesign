dataset=cave
epochs=100
noise=0
# noise=10
# camera=OlympusE-PL2
# camera=NikonD40

# mkdir ../build/$dataset/$camera"_srgb_noise"$noise
# python ../src/mytrain_numpy.py --gpu --epochs $epochs --noise $noise --output ../build/$dataset/$camera"_srgb_noise"$noise/ --gt srgb --dataset $dataset --camera $camera

train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise
mkdir $train_dir
python ../src/mytrain.py --gpu --epochs $epochs --noise $noise --output $train_dir/ --gt srgb --dataset $dataset
read -p "finish"