cd %dataset%\gaussian_splatting\%target%
mkdir input
ffmpeg -i video.mp4 -qscale:v 1 -qmin 1 -vf fps=1 input/%03d.jpg

cd %workspace%\gaussian-splatting
python convert.py -s %dataset%\gaussian_splatting\%target%

python train.py -s %dataset%\gaussian_splatting\%target% --auto_checkpoint --data_device cpu

cd C:\Users\sucha\workspace\gaussian-splatting