conda activate paws
export PYTHONPATH='/home/larbez/Documents/suncet/'
source gpu_setVisibleDevices.sh
GPUID=0
cd /home/larbez/Documents/suncet/

python main.py --sel paws_train --fname configs/paws/ClusterVec_train.yaml #--devices cuda:0 cuda:1
