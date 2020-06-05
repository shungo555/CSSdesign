dataset=cave
epochs=100
noise=0
# noise=30
# camera=OlympusE-PL2
# camera=NikonD40
# camera=Canon300D
# camera=Canon20D

# train_dir=../build/$dataset/$camera"_srgb_noise"$noise"_200"
# weigt_file=../build/$dataset/$camera"_srgb_noise"$noise/img_best.hdf5
# mkdir $train_dir
# python ../src/mytrain_numpy.py --gpu --epochs $epochs --noise $noise --output $train_dir/ --weight $weigt_file --gt srgb --dataset $dataset --camera $camera

# val_dir=../../dataset/$dataset"_hsi"/val_data/srgb/CSS_change
# python ../src/mytrain.py --gpu --epochs $epochs --noise $noise --output $val_dir/ --gt srgb --dataset $dataset --validation

# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_6_400_ver2"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_ver3"
# weight_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_pretrained"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_0_pretrained_used"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_0_pretrained_used_ver2"
weight_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_pretrained_used_ver2"
train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_pretrained_used_ver3"

mkdir $train_dir
python ../src/mytrain.py --gpu --epochs $epochs --noise $noise --output $train_dir/ --gt srgb --dataset $dataset --weight $weight_dir/img_best.hdf5
read -p "finish"