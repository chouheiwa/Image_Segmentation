import torch
from sklearn.metrics import roc_auc_score, average_precision_score


# SR : Segmentation Result
# GT : Ground Truth

def get_iou(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # Intersection
    intersection = torch.sum(SR & GT)

    # Union
    union = torch.sum(SR | GT)

    # IoU
    iou = float(intersection) / (float(union) + 1e-6)

    return iou


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum(SR.byte() + GT.byte() == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC


def get_ap(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    return average_precision_score(SR.cpu(), GT.cpu())


def get_auc(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    return roc_auc_score(SR.cpu(), GT.cpu())


class BinaryFilterEvaluator:
    def __init__(self, epoch, total_epoch, type, threshold=0.5):
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.type = type
        self.acc = 0.  # Accuracy
        self.SE = 0.  # Sensitivity (Recall)
        self.SP = 0.  # Specificity
        self.PC = 0.  # Precision
        self.F1 = 0.  # F1 Score
        self.DC = 0.  # Dice Coefficient
        self.MIOU = 0.  # Mean Intersection over Union
        self.AUC = 0.  # Area Under the Curve
        self.AP = 0.  # Average Precision
        self.epoch_loss = 0.
        self.length = 0
        self.threshold = threshold

    def evaluate(self, y_pred, y_true, length, loss):
        self.acc += get_accuracy(y_pred, y_true, self.threshold)
        self.SE += get_sensitivity(y_pred, y_true, self.threshold)
        self.SP += get_specificity(y_pred, y_true, self.threshold)
        self.PC += get_precision(y_pred, y_true, self.threshold)
        self.F1 += get_F1(y_pred, y_true, self.threshold)
        self.DC += get_DC(y_pred, y_true, self.threshold)
        self.MIOU += get_iou(y_pred, y_true, self.threshold)
        # self.AUC += get_auc(y_pred, y_true, self.threshold)
        # self.AP += get_ap(y_pred, y_true, self.threshold)
        self.epoch_loss += loss
        self.length += length

    def calculate(self):
        self.acc /= self.length
        self.SE /= self.length
        self.SP /= self.length
        self.PC /= self.length
        self.F1 /= self.length
        self.DC /= self.length
        self.MIOU /= self.length
        self.AUC /= self.length
        self.AP /= self.length

    def to_log(self):
        if self.type == 'train':
            return 'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, DC: %.4f, MIOU: %.4f, AUC: %.4f, AP: %.4f' % (
                self.epoch + 1, self.total_epoch, self.epoch_loss, self.acc, self.SE, self.SP, self.PC,
                self.F1,
                self.DC, self.MIOU, self.AUC, self.AP
            )

        return '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, DC: %.4f, MIOU: %.4f, AUC: %.4f, AP: %.4f' % (
            self.acc, self.SE, self.SP, self.PC,
            self.F1,
            self.DC, self.MIOU, self.AUC, self.AP
        )

    def to_tensorboard(self):
        return {
            'epoch': self.epoch + 1,
            'loss': self.epoch_loss,
            'type': self.type,
            'acc': self.acc,
            'SE': self.SE,
            'SP': self.SP,
            'PC': self.PC,
            'F1': self.F1,
            'DC': self.DC,
            'MIOU': self.MIOU,
            'AUC': self.AUC,
            'AP': self.AP
        }
