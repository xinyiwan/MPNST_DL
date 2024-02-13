import os
import numpy as np
import matplotlib.pyplot as plt
from neptune.types import File
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import nibabel as nib
import matplotlib.pyplot as plt


def log_confusion_matrix(lit_model, data_module, neptune_logger, cp_dir):
    lit_model.freeze()
    val_data = data_module.val_dataloader()
    y_true = np.array([])
    y_pred = np.array([])
    for batch in val_data:
        x, y = batch['input'], batch['label']
        y = y.cpu().detach().numpy()
        y_hat = lit_model.forward(x).argmax(axis=1).cpu().detach().numpy()
        y_true = np.append(y_true, y)
        y_pred = np.append(y_pred, y_hat)

    fig, ax = plt.subplots(figsize=(16, 12))
    cm = confusion_matrix(y_true, y_pred)
    fig = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig.plot().figure_.savefig(os.path.join(cp_dir,'confusion_matrix.png'))
    neptune_logger.experiment["confusion_matrix"].upload(File.as_image(fig.plot().figure_))

def plot_batch(dataloader, cp_dir, mode, neptune_logger, num=1):

    plt.rcParams['axes.titlesize'] = 8
    for i, batch in enumerate(dataloader):
        
        columns = len(batch['input'])
        fig, axs = plt.subplots(2, columns, figsize=(columns*3, 6))
        for j in range(columns):
            x, y, z = batch['input'][j][0,:,:,:].shape
            img = batch['input'][j][0,:,:,int(z/2)]
            seg = batch['input'][j][1,:,:,int(z/2)]

            if columns == 1:
                axs[0].set_title(f"{batch['case_id'][j]} Label:{batch['label'][j].item()}")
                axs[0].imshow(img, cmap='gray')
                axs[1].imshow(seg, cmap='gray')
            else:
                axs[0][j].set_title(f"{batch['case_id'][j]} Label:{batch['label'][j].item()}")
                axs[0][j].imshow(img, cmap='gray')
                axs[1][j].imshow(seg, cmap='gray')
        plt.axis('off')
        plt.show()
        plt.savefig(os.path.join(cp_dir,f'{mode}_batch_{i}.png'))
        neptune_logger.experiment[f'{mode}_batch_{i}.png'].upload(File.as_image(fig))
        if i==num:
            break
            


# Test code
# for i, batch in enumerate(dataloader):
#     fig = plt.figure(figsize=(25, 8))
#     columns = len(batch['image'])
#     for j in range(columns):
#         x, y, z = batch['image'][j][0,:,:,:].shape
#         img = batch['image'][j][0,:,:,int(z/2)]
#         a = fig.add_subplot(1, columns, j+1)
#         a.set_title(f"{batch['case_id'][j]}_L_{batch['label'][j]}")
#         plt.imshow(img, cmap='gray')
#         plt.axis('off')
#     plt.show()
#     plt.savefig(os.path.join(cp_dir,f'{mode}_batch_{i}.png'))
#     if i==2:
#         break


# for i, batch in enumerate(dataloader):
#     batch['image'][0]
