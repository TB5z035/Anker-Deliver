conda create -n anker_deliver python=3.8

pip install pytest yapf
pip install numpy plyfile
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch