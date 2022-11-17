import numpy as np


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ConfusionMatrix(object):
    def __init__(self, num_label=2):
        self.result = np.zeros((num_label, num_label))
    def update(self, output, targets):
        maxk = 1
        _, preds = output.topk(maxk, 1, True, True)
        preds = preds.cpu().numpy().tolist()
        targets = targets.cpu().numpy().tolist()
        for target, pred in zip(targets, preds):
            target, pred = target, pred[0]
            self.result[target, pred] += 1
    def get_result(self):
        return self.result


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # maxk = max(topk)
    maxk = 1
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    correct_k = correct[:1].view(-1).float().sum(0)
    res = correct_k.mul_(100.0 / batch_size)
    return res
    # res = []
    # for k in topk:
    #     correct_k = correct[:k].view(-1).float().sum(0)
    #     res.append(correct_k.mul_(100.0 / batch_size))
    # return res