conda create -n anker_deliver python=3.8

pip install pytest yapf
pip install numpy plyfile
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install pymeshlab potpourri3d open3d
pip install scikit-learn scikit-learn-intelex
pip install tqdm