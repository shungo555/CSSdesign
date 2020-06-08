dataset=cave
epochs=400
noise=0
camera=Canon20D
camera=random


val_dir=../../dataset/$dataset"_hsi"/val_data/srgb/CSS_change/
# val_file=$val_dir"val_data_noise30_grrb.pickle"
val_file=$val_dir"val_data_noise30_randomCSS_gbbr.pickle"

python ../src/mytrain.py --gpu --epochs $epochs --noise $noise --gt srgb --dataset $dataset --validation --valfile $val_file --camera $camera

train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_init_randomCSS_gbbr"
# weight_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_fixed_Canon20D/img_best.hdf5"

mkdir $train_dir
# nohup python ../src/mytrain.py --gpu --epochs $epochs --noise $noise --output $train_dir/ --gt srgb --dataset $dataset --camera $camera --trainable --weight $weight_dir
nohup python ../src/mytrain.py --gpu --epochs $epochs --noise $noise --output $train_dir/ --gt srgb --dataset $dataset --camera $camera --trainable --valfile $val_file
read -p "finish"
