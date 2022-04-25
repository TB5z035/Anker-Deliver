import os
from plyfile import PlyData
from .uncertainty.utils import add_fields_online

def run():
    for i in os.listdir('input/init'):
        plydata = PlyData.read('input/init/' + i)
        plydata = add_fields_online(plydata, [('label', 'u1')])
        plydata.write('input/init/' + i)

if __name__ == '__main__':
    run()