# ctcl
A code-text cross-modal outcome prediction framework.

## Train the CTCL
cd baselines
python train_CTMR.py --batch_size 64 --device [GPU id]

## Test the CTCL
cd baselines
python train_CTMR.py --batch_size 64 --device [GPU id] --Test --resume_path [Pretrained model path]


