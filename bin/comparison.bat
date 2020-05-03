set dataset=cave
REM set dataset=tokyotech

REM set train_dir_name=build/%dataset%/canon20D_train_crgb_ft
set train_dir_name=build/%dataset%/canon20D_srgb_noise0

set test_dir_name=../%train_dir_name%/test/noise0
set true_crgb_file_name=../../dataset/%dataset%_hsi/%dataset%_crgb.npy
set true_srgb_file_name=../../dataset/%dataset%_hsi/%dataset%_srgb.npy

REM set true_crgb_file_name=../../dataset/tokyotech/%dataset%_crgb.npy
REM set true_srgb_file_name=../../dataset/tokyotech/%dataset%_srgb.npy
REM set train2_dir_name=../build/%dataset%/canon20D_train_srgb_ft
set test2_dir_name=%train2_dir_name%/test/noise0

set CSS_dir=../CSSdesign/
set bm3d_raw_npy_file_name=%train_dir_name%/test/noise0/raw3.npy
set bm3d_raw_mat_file_name=%train_dir_name%/test/noise0/bm3d_raw.mat
set RI_mat_file_name=%train_dir_name%/test/noise0/RI.mat
set RI_npy_file_name=%train_dir_name%/test/noise0/RI.npy


REM convert npy to mat
python ../lib/convert_data/npy2mat.py --mat ../%bm3d_raw_mat_file_name% --npy ../%bm3d_raw_npy_file_name% --name raw

cd ../../Sensors_ARI/
matlab -nodesktop -nosplash -r "calc_RI('%CSS_dir%%bm3d_raw_mat_file_name%','%CSS_dir%%RI_mat_file_name%'); exit()"

REM REM pause

REM cd %CSS_dir%bin
REM REM convert mat to npy
REM python ../lib/convert_data/mat2npy.py --mat ../%RI_mat_file_name% --npy ../%RI_npy_file_name% --name out

REM REM cPSNR (Ypred, RI)
REM python ../lib/image_tools/evaluation.py --input1 %true_crgb_file_name% --input2 %test_dir_name%/Ypred.npy --output  %test_dir_name%/result.csv --border 1
REM python ../lib/image_tools/evaluation.py --input1 %true_crgb_file_name% --input2 %test_dir_name%/RI.npy --output  %test_dir_name%/result_RI.csv


REM REM CC
REM python ../lib/image_tools/color_correction.py --crgb %true_crgb_file_name% --output %test_dir_name%/crgb2srgb_only.npy --dataset %dataset%
REM python ../lib/image_tools/color_correction.py --crgb %test_dir_name%/Ypred.npy --output %test_dir_name%/crgb2srgb.npy --dataset %dataset%
REM python ../lib/image_tools/color_correction.py --crgb %test_dir_name%/RI.npy --output %test_dir_name%/crgb2srgbRI.npy --dataset %dataset%

REM REM REM cPSNR (Ypred+CC, RI+CC)
REM python ../lib/image_tools/evaluation.py --input1 %true_srgb_file_name% --input2 %test_dir_name%/crgb2srgb_only.npy --output  %test_dir_name%/result_cRGB_CC.csv
REM python ../lib/image_tools/evaluation.py --input1 %true_srgb_file_name% --input2 %test_dir_name%/crgb2srgb.npy --output  %test_dir_name%/result_Ypred_CC.csv --border 1
REM python ../lib/image_tools/evaluation.py --input1 %true_srgb_file_name% --input2 %test_dir_name%/crgb2srgbRI.npy --output  %test_dir_name%/result_RI_CC.csv
REM python ../lib/image_tools/evaluation.py --input1 %true_srgb_file_name% --input2 %test2_dir_name%/Ypred.npy --output  %test2_dir_name%/result.csv --border 1

REM REM output image
REM python ../lib/image_tools/draw_image.py --input %test_dir_name%/crgb2srgb_only.npy --output %test_dir_name%/img/ --title crgb2srgb_only --gamma
REM python ../lib/image_tools/draw_image.py --input %test_dir_name%/Ypred.npy --output %test_dir_name%/img/ --title Ypred --gamma
REM python ../lib/image_tools/draw_image.py --input %test_dir_name%/crgb2srgb.npy --output %test_dir_name%/img/ --title crgb2srgb --gamma
REM python ../lib/image_tools/draw_image.py --input %test_dir_name%/RI.npy --output %test_dir_name%/img/ --title RI --gamma
REM python ../lib/image_tools/draw_image.py --input %test_dir_name%/crgb2srgbRI.npy --output %test_dir_name%/img/ --title crgb2srgbRI --gamma