noise=0
dataset=cave

# # for camera in Canon300D NikonD40 OlympusE-PL2
# for camera in Canon20D
# do
#     train_dir=../build/$dataset/$camera"_srgb_noise"$noise"_200"
#     mkdir $train_dir/test
#     for test_noise in 0 10 20 30 
#     do
#         test_dir=$train_dir/test/noise$test_noise
#         mkdir $test_dir
#         mkdir $test_dir/img
#         python ../src/mytest_numpy.py --gpu --noise $test_noise --weight $train_dir/img_best.hdf5 --output $test_dir/ --gt srgb --dataset $dataset --camera $camera
#     done
# done

# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_6_ver2"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_6"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_0_ver2"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_ver2"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_0_400"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_0_pretrained_used_ver2"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_init_Olympus"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_6_400_ver2"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_init_Canon20D_net"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_init_randomCSS"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_init_randomCSS_ver2"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_fixed_Canon20D_100"
train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_init_Canon20D_grrb"
# train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise"_ep_7_init_Canon20D_gbbr"


# for test_noise in 0 10 20 30 
# do
#     test_dir=$train_dir/test/noise$test_noise
#     mkdir $train_dir/test
#     mkdir $test_dir
#     mkdir $test_dir/img
#     python ../src/mytest.py --gpu --noise $test_noise --weight $train_dir/img_best.hdf5 --output $test_dir/ --gt srgb --dataset $dataset
# done
python ../src/weight_plot.py --input $train_dir/ --output $train_dir/ --noise $noise
read -p "finish"