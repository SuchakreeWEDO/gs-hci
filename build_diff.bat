cd submodules/diff-gaussian-rasterization-depth-dmode
python setup.py install
cd ../../
pip uninstall diff_gaussian_rasterization_depth_dmode -y
pip install submodules/diff-gaussian-rasterization-depth-dmode