REM python ../src/mytest_numpy.py --gpu --weight ../build/cave/canon20D_train_crgb_ft/img_best.hdf5 --noise 10 --output ../build/cave/canon20D_train_crgb_ft/test/noise10/ --dataset cave --gt crgb 
REM python ../src/mytest_numpy.py --gpu --weight ../build/cave/canon20D_train_srgb_ft/img_best.hdf5 --noise 10 --output ../build/cave/canon20D_train_srgb_ft/test/noise10/ --dataset cave --gt srgb 
REM python ../src/mytest_numpy.py --gpu --weight ../build/cave/canon20D_train_srgb_ft/img_best.hdf5 --noise 20 --output ../build/cave/canon20D_train_srgb_ft/test/noise20/ --dataset cave --gt srgb 
REM python ../src/mytest_numpy.py --gpu --weight ../build/cave/canon20D_train_srgb_ft/img_best.hdf5 --noise 30 --output ../build/cave/canon20D_train_srgb_ft/test/noise30/ --dataset cave --gt srgb 
REM python ../src/mytest_numpy.py --gpu --weight ../build/cave/canon20D_train_srgb_ft/img_best.hdf5 --noise 40 --output ../build/cave/canon20D_train_srgb_ft/test/noise40/ --dataset cave --gt srgb 
REM python ../src/mytest_numpy.py --gpu --weight ../build/cave/canon20D_train_srgb_ft/img_best.hdf5 --noise 0 --output ../build/cave/canon20D_train_srgb_ft/test/noise0/ --dataset cave --gt srgb 

REM python ../src/mytest_numpy.py --gpu --weight ../build/tokyotech/canon20D_train_crgb_ft/img_best.hdf5 --noise 10 --output ../build/tokyotech/canon20D_train_crgb_ft/test/noise10/ --dataset tokyotech --gt crgb 
REM python ../src/mytest_numpy.py --gpu --weight ../build/tokyotech/canon20D_train_srgb_ft/img_best.hdf5 --noise 10 --output ../build/tokyotech/canon20D_train_srgb_ft/test/noise10/ --dataset tokyotech --gt srgb 

REM python ../src/mytest_numpy.py --gpu --weight ../build/tokyotech/canon20D_train_crgb_ff/img_best.hdf5 --noise 0 --output ../build/tokyotech/canon20D_train_crgb_ff/test/noise0/ --dataset tokyotech --gt crgb 
REM python ../src/mytest_numpy.py --gpu --weight ../build/cave/canon20D_train_srgb_noise30/img_best.hdf5 --noise 5 --output ../build/cave/canon20D_train_srgb_noise30/test/noise5/ --dataset cave --gt srgb 

REM python ../src/mytest_numpy.py --gpu --weight ../build/cave/canon20D_train_srgb_ft/img_best.hdf5 --noise 30 --output ../build/cave/canon20D_train_srgb_ft/test/noise30_0/ --dataset cave --gt srgb 

REM python ../src/mytest_numpy.py --gpu --weight ../build/cave/OlympusE-PL2_srgb_noise100/img_best.hdf5 --noise 0 --output ../build/cave/OlympusE-PL2_srgb_noise100/test/noise0/ --dataset cave --gt srgb --camera OlympusE-PL2
REM python ../src/mytest_numpy.py --gpu --weight ../build/cave/OlympusE-PL2_srgb_noise100/img_best.hdf5 --noise 50 --output ../build/cave/OlympusE-PL2_srgb_noise100/test/noise50/ --dataset cave --gt srgb --camera OlympusE-PL2
REM python ../src/mytest_numpy.py --gpu --weight ../build/cave/OlympusE-PL2_srgb_noise100/img_best.hdf5 --noise 100 --output ../build/cave/OlympusE-PL2_srgb_noise100/test/noise100/ --dataset cave --gt srgb --camera OlympusE-PL2
python ../src/mytest.py --gpu --weight ../build/cave/CSS_change_srgb_noise0/img_best.hdf5 --noise 0 --output ../build/cave/CSS_change_srgb_noise0/test/noise0/ --dataset cave --gt srgb
python ../src/mytest.py --gpu --weight ../build/cave/CSS_change_srgb_noise0/img_best.hdf5 --noise 10 --output ../build/cave/CSS_change_srgb_noise0/test/noise10/ --dataset cave --gt srgb
python ../src/mytest.py --gpu --weight ../build/cave/CSS_change_srgb_noise0/img_best.hdf5 --noise 20 --output ../build/cave/CSS_change_srgb_noise0/test/noise20/ --dataset cave --gt srgb
python ../src/mytest.py --gpu --weight ../build/cave/CSS_change_srgb_noise0/img_best.hdf5 --noise 30 --output ../build/cave/CSS_change_srgb_noise0/test/noise30/ --dataset cave --gt srgb

pause