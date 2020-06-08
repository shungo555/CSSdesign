# CSSdesign

## explanation of files
```mytrain.py``` main file for CSS training

```mytest.py``` main file for testing

```my_build_model.py``` define encoder model

```build_model.py``` define decoder model

```mydata.py``` define generator

```bayer_layers.py``` keras bayer model (read by build_model.py)

## mytrain.py

### example 
```mytrain.py --trainable --epochs 100 --noise 0 --output results/ --valfile val_data.pickle```

### options
```--gpu``` if you set this option, you can use the GPU for processing.(default: True)

```--trainable``` if trainable is true, CSS parameters are trainable.(default: False)

```--epochs <int>``` number of training epochs. (default: 100)

```--noise <int>``` training noise level (8bit). (default: 0)

```--output <string>``` output directory (default: results/)

```--weight <string>``` if you have weight file which has already trained, set file name and you can continue to train from that weight.(default: None)

```--gt <string>``` set ground truth as srgb or crgb (default: srgb)

```--dataset <string>``` set dataset cave or tokyotech (default: cave)

```--camera <string>``` set camera name (default: Canon20D)

```--validation``` if you make validation data, set True (default: False)

```--valfile <string>``` set validation file name(default: val_data.pickle)



## mytest.py
### example 
```mytest.py --noise 0 --weight results/img_best.hdf5 --output results/```

### options

```--gpu``` if you set this option, you can use the GPU for processing.(default: True)

```--noise <int>``` training noise level (8bit). (default: 0)

```--output <string>``` output directory (default: results/)

```--weight <string>``` if you have weight file which has already trained, set file name and you can continue to train from that weight.(default: None)

```--gt <string>``` set ground truth as srgb or crgb (default: srgb)

```--dataset <string>``` set dataset cave or tokyotech (default: cave)

```--camera <string>``` set camera name (default: Canon20D)
