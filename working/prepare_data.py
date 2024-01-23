import pandas as pd
from dataset import MPNSTDataset
def prepare_roi_3d(mri_type):
    df = pd.read_csv(f"/trinity/home/xwan/MPNST_DL/input/{mri_type}/train_{mri_type}.csv")

    img_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/{mri_type}.nii.gz' for pid in df['Patient']]
    seg_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/segmentations.nii.gz' for pid in df['Patient']]
    labels = df['MPNST'].tolist()
    ids = df['Patient'].tolist()

    data_set = [{"image": img, "seg": seg, "label": label, "case_id": pid} for img, seg, label, pid in zip(img_paths, seg_paths, labels, ids)]
    set = MPNSTDataset(data=data_set, mri_type=mri_type, transform=None)
    set.prepare_roi()
    return 

def main():
    prepare_roi_3d('T1')

if __name__ == '__main__':
    main()
