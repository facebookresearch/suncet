conda activate paws
export PYTHONPATH='/home/larbez/Documents/suncet/'
source gpu_setVisibleDevices.sh
GPUID=0
cd /home/larbez/Documents/suncet/

python main.py --sel fine_tune --fname configs/paws/ClusteVec_finetune_imgnt.yaml #configs/paws/ClusteVec_finetune.yaml #
