filename=M030_ang_3_004

mkdir $filename
mkdir $filename/deepfeature32
mkdir $filename/latent_evp_25  
mkdir $filename/poseimg
mkdir $filename/images_evp_25
mkdir $filename/images_evp_25/cropped

# wav
cp ../../mead_data/wav_16000/$filename.wav ./$filename/

# deepfeature32
cp ../../mead_data/deepfeature32/$filename.npy $filename/deepfeature32/

# latent
cp ../../mead_data/latent_evp_25/test/$filename.npy $filename/latent_evp_25/

# poseimg
cp ../../mead_data/poseimg/$filename.npy.gz ./$filename/poseimg/

# images_evp_25
cp ../../mead_data/images_evp_25/test/$filename/* ./$filename/images_evp_25/cropped/
