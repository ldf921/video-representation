import matplotlib.pyplot as plt

def saveimg(truth, pred, path, n = None):
    if n is None:
        n = truth.shape[0]
    else:
        n = min(n, truth.shape[0])
    fig, axes = plt.subplots((n, 2), figsize=(n * 4, 4))
    for i in range(n):
        axes[i,0].imshow(truth[i])
        axes[i,1].imshow(pred[i])
    for ax in axes.flatten():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.savefig(path)