import os
import numpy as np
import matplotlib.pyplot as plt
from neptune.types import File
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def log_confusion_matrix(lit_model, data_module, neptune_logger, cp_dir):
    lit_model.freeze()
    val_data = data_module.val_dataloader()
    y_true = np.array([])
    y_pred = np.array([])
    for batch in val_data:
        x, y = batch['image'], batch['label']
        y = y.cpu().detach().numpy()
        y_hat = lit_model.forward(x).argmax(axis=1).cpu().detach().numpy()
        y_true = np.append(y_true, y)
        y_pred = np.append(y_pred, y_hat)

    fig, ax = plt.subplots(figsize=(16, 12))
    cm = confusion_matrix(y_true, y_pred)
    fig = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig.figure_.savefig(os.path.join(cp_dir,'confusion_matrix.png'))
    neptune_logger.experiment["confusion_matrix"].upload(File.as_image(fig.figure_))
