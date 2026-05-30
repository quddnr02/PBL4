import torch
import numpy as np
import torch.nn.functional as F

class SegMetrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred = torch.argmax(F.softmax(pred, dim=0), dim=0).cpu().detach().flatten().numpy()
            label = label.flatten().cpu().detach().numpy()

            mask = (label >= 0) & (label < self.num_classes)
            category = np.bincount(
                label[mask] * self.num_classes + pred[mask]
                , minlength=self.num_classes ** 2
            ).reshape(self.num_classes, self.num_classes)
            self.confusion_matrix += category

    def get_result(self):
        conf_mat = self.confusion_matrix
        pa = np.diag(conf_mat).sum() / (conf_mat.sum() + 1e-7)
        iou = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat) + 1e-7)
        miou = np.nanmean(iou)

        return pa, miou

if __name__ == '__main__':
    """label = torch.tensor([0, 1, 0, 1, 2])
    pred = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])"""
    label = torch.randint(0, 2, size=(2, 200, 200))
    pred = torch.rand((2, 5, 200, 200))

    metrix = SegMetrics(num_classes=5)
    metrix.update(pred, label)
    print("=========", metrix.get_result())
