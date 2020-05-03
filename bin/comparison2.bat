set DATA_ROOT=../
set dataset=cave

set train_noise=10
set test_noise=25

set true_crgb_file_name=../../dataset/%dataset%_hsi/%dataset%_crgb.npy
set true_srgb_file_name=../../dataset/%dataset%_hsi/%dataset%_srgb.npy

set train_dir_name=build/cave/canon20D_train_srgb_noise%train_noise%
set test_dir_name=%train_dir_name%/test/noise%test_noise%

set raw_file=%test_dir_name%/raw3.npy

set gain_file=%test_dir_name%/gain.npy

set bm3d_raw_npy_file=%test_dir_name%/bm3d_raw.npy
set bm3d_raw_mat_file=%test_dir_name%/bm3d_raw.mat
set RI_mat_file=%test_dir_name%/RI.mat
set RI_npy_file=%test_dir_name%/RI.npy
set crgb2srgb_RI_npy_file=%test_dir_name%/crgb2srgbRI.npy

set no_bm3d_raw_mat_file=%test_dir_name%/no_bm3d_raw.mat
set no_bm3d_RI_mat_file=%test_dir_name%/no_bm3d_RI.mat
set no_bm3d_RI_npy_file=%test_dir_name%/no_bm3d_RI.npy
set crgb2srgb_no_bm3d_RI_npy_file=%test_dir_name%/crgb2srgb_no_bm3d_RI.npy

set CSS_dir=../CSSdesign/

REM REM denoise
python %DATA_ROOT%lib/image_tools/denoise.py --input %DATA_ROOT%%raw_file% --output %DATA_ROOT%%bm3d_raw_npy_file% --noise %test_noise% --gain %DATA_ROOT%%gain_file%
pause
REM convert npy to mat
python %DATA_ROOT%lib/convert_data/npy2mat.py --mat %DATA_ROOT%%bm3d_raw_mat_file% --npy %DATA_ROOT%%bm3d_raw_npy_file% --name raw
REM demosaic
cd %DATA_ROOT%../Sensors_ARI/
matlab -nodesktop -nosplash -r "calc_RI('%CSS_dir%%bm3d_raw_mat_file%','%CSS_dir%%RI_mat_file%'); exit()"
pause
cd %CSS_dir%bin
REM convert mat to npy
python %DATA_ROOT%lib/convert_data/mat2npy.py --mat %DATA_ROOT%%RI_mat_file% --npy %DATA_ROOT%%RI_npy_file% --name out
REM REM CC
python %DATA_ROOT%lib/image_tools/color_correction.py --crgb %DATA_ROOT%%RI_npy_file% --output %DATA_ROOT%%crgb2srgb_RI_npy_file% --dataset %dataset%
REM Evaluation
python %DATA_ROOT%lib/image_tools/evaluation.py --input1 %true_srgb_file_name% --input2 %DATA_ROOT%%crgb2srgb_RI_npy_file% --output  %DATA_ROOT%%test_dir_name%/result_RI_CC.csv
REM output images
python %DATA_ROOT%lib/image_tools/draw_image.py --input %DATA_ROOT%%RI_npy_file% --output %DATA_ROOT%%test_dir_name%/img/ --title RI --gamma
python %DATA_ROOT%lib/image_tools/draw_image.py --input %DATA_ROOT%%crgb2srgb_RI_npy_file% --output %DATA_ROOT%%test_dir_name%/img/ --title crgb2srgbRI --gamma

REM convert npy to mat
python %DATA_ROOT%lib/convert_data/npy2mat.py --mat %DATA_ROOT%%no_bm3d_raw_mat_file% --npy %DATA_ROOT%%raw_file% --name raw
REM demosaic
cd %DATA_ROOT%../Sensors_ARI/
matlab -nodesktop -nosplash -r "calc_RI('%CSS_dir%%no_bm3d_raw_mat_file%','%CSS_dir%%no_bm3d_RI_mat_file%'); exit()"
pause
cd %CSS_dir%bin
REM convert mat to npy
python %DATA_ROOT%lib/convert_data/mat2npy.py --mat %DATA_ROOT%%no_bm3d_RI_mat_file% --npy %DATA_ROOT%%no_bm3d_RI_npy_file% --name out
REM REM CC
python %DATA_ROOT%lib/image_tools/color_correction.py --crgb %DATA_ROOT%%no_bm3d_RI_npy_file% --output %DATA_ROOT%%crgb2srgb_no_bm3d_RI_npy_file% --dataset %dataset%
REM Evaluation
python %DATA_ROOT%lib/image_tools/evaluation.py --input1 %true_srgb_file_name% --input2 %DATA_ROOT%%crgb2srgb_no_bm3d_RI_npy_file% --output  %DATA_ROOT%%test_dir_name%/result_no_bm3d_RI_CC.csv
REM output images
python %DATA_ROOT%lib/image_tools/draw_image.py --input %DATA_ROOT%%no_bm3d_RI_npy_file% --output %DATA_ROOT%%test_dir_name%/img/ --title no_bm3d_RI --gamma
python %DATA_ROOT%lib/image_tools/draw_image.py --input %DATA_ROOT%%crgb2srgb_no_bm3d_RI_npy_file% --output %DATA_ROOT%%test_dir_name%/img/ --title crgb2srgb_no_bm3d_RI --gamma