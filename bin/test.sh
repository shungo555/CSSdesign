noise=0
dataset=cave

# for camera in Canon300D NikonD40 OlympusE-PL2
# for camera in NikonD40
# do
#     train_dir=../build/$dataset/$camera"_srgb_noise"$noise
#     mkdir $train_dir/test
#     for test_noise in 0 10 20 30 
#     do
#         test_dir=$train_dir/test/noise$test_noise
#         mkdir $test_dir
#         mkdir $test_dir/img
#         python ../src/mytest_numpy.py --gpu --noise $test_noise --weight $train_dir/img_best.hdf5 --output $test_dir/ --gt srgb --dataset $dataset --camera $camera
#     done
# done

for test_noise in 0 10 20 30 
do
    train_dir=../build/$dataset/"CSS_change_srgb_noise"$noise
    test_dir=$train_dir/test/noise$test_noise
    mkdir $train_dir/test
    mkdir $test_dir
    mkdir $test_dir/img
    python ../src/mytest.py --gpu --noise $test_noise --weight $train_dir/img_best.hdf5 --output $test_dir/ --gt srgb --dataset $dataset
done


read -p "finish"