SET dataset=D:\3d-reconstruction\datasets\redroom-colmap2nerf2gs
SET workspace=D:\3d-reconstruction\gs-hci

@REM cd %dataset%
@REM mkdir input
@REM ffmpeg -i video.mp4 -qscale:v 1 -qmin 1 -vf fps=0.5 input/%%03d.jpg

@REM cd %workspace%
@REM python convert.py -s %dataset%

cd %workspace%
python train.py -s %dataset% --method gs --auto_checkpoint

@REM python render.py -m <path to trained model> # Generate renderings
@REM python metrics.py -m <path to trained model> # Compute error metrics on renderings