
SET dataset=D:\3d-reconstruction\datasets\aluchair_ss_vid_halfframe
@REM SET D:\3d-reconstruction\datasets\plant_ss_vid_lessframe
@REM SET D:\3d-reconstruction\datasets\greychair_ss_vid_lessframe
@REM SET D:\3d-reconstruction\datasets\woodchair_ss_vid_lessframe
SET workspace=D:\3d-reconstruction\gs-hci

@REM cd %dataset%
@REM mkdir input
@REM ffmpeg -i video.mp4 -qscale:v 1 -qmin 1 -vf fps=0.5 input/%%03d.jpg

cd %workspace%
python convert.py -s %dataset%

cd %workspace%
python train.py -s %dataset% --method gs --eval --auto_checkpoint


@REM python render.py -m <path to trained model> # Generate renderings
@REM python metrics.py -m <path to trained model> # Compute error metrics on renderings