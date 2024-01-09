import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import glob, os

parser = argparse.ArgumentParser()
parser.add_argument("--n_folds", default=5, type=int)
parser.add_argument("--mri_type", default='T1', type=str)
args = parser.parse_args()

if not os.path.exists(f"input/{args.mri_type}"):
    os.mkdir(f"input/{args.mri_type}")

# Choose pid list of experiment 
exp = args.mri_type
images = glob.glob(os.path.join('/trinity/home/xwan/data/MPNST',f"*/{exp}.nii.gz"))
pids = [i.split('MPNST/')[1].split('/')[0] for i in images]

# Generate new pinfo for expriment
train_orig = pd.read_csv("/trinity/home/xwan/data/MPNST/pinfo_MPNST.csv")
train = train_orig[train_orig['Patient'].isin(pids)]
train.to_csv(f"input/{exp}/train_{exp}.csv", index=False)




skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=518)
target = "MPNST"
train = pd.read_csv(f"input/train_{exp}.csv")

for fold, (trn_idx, val_idx) in enumerate(
    skf.split(train, train[target])
):
    train.loc[val_idx, "fold"] = int(fold)


train.to_csv(f"input/train_{exp}.csv", index=False)
