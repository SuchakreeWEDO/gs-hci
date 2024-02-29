cd %dataset%\gaussian_splatting\%target%
mkdir input
ffmpeg -i video.mp4 -qscale:v 1 -qmin 1 -vf fps=1 input/%03d.jpg

cd %workspace%\gaussian-splatting
python convert.py -s %dataset%\gaussian_splatting\%target%

python train.py -s ../datasets/android-img-fixisoae --method gs-depth --eval --auto_checkpoint
python train.py -s ../datasets/android-img-fixisoae --method gs --eval --auto_checkpoint
tensorboard --logdir ./output

python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings