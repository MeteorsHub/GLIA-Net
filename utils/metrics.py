import logging

import matplotlib
import matplotlib.figure
import matplotlib.pyplot
import numpy as np
import skimage.measure as measure
import sklearn.metrics
import torch
from medpy.metric import hd95

from utils.project_utils import transpose_move_to_end, one_hot


class _ConfusionMatrix(torch.nn.Module):
    """
    compute cm, tp, fp, fn, tn.
    Threshold only activated when has_background is true(class 0)
    """

    def __init__(self, num_classes=2, threshold=0.5, has_background=True):
        super(_ConfusionMatrix, self).__init__()
        self.cm = torch.nn.Parameter(torch.zeros([num_classes, num_classes], dtype=torch.int64), requires_grad=False)
        self.num_samples = 0
        self.num_classes = num_classes
        self.threshold = threshold
        self.has_background = has_background

        self._fp = torch.nn.Parameter(torch.zeros([num_classes], dtype=torch.int64), requires_grad=False)
        self._fn = torch.nn.Parameter(torch.zeros([num_classes], dtype=torch.int64), requires_grad=False)
        self._tp = torch.nn.Parameter(torch.zeros([num_classes], dtype=torch.int64), requires_grad=False)
        self._tn = torch.nn.Parameter(torch.zeros([num_classes], dtype=torch.int64), requires_grad=False)

        self.last_update = None

    @property
    def fp(self):
        if self.num_classes == 2:
            return self._fp[1]
        return self._fp

    @property
    def fn(self):
        if self.num_classes == 2:
            return self._fn[1]
        return self._fn

    @property
    def tp(self):
        if self.num_classes == 2:
            return self._tp[1]
        return self._tp

    @property
    def tn(self):
        if self.num_classes == 2:
            return self._tn[1]
        return self._tn

    def reset(self):
        self.cm[:] = 0
        self.num_samples = 0
        self._fp[:] = 0
        self._fn[:] = 0
        self._tp[:] = 0
        self._tn[:] = 0
        self.last_update = None

    def __call__(self, y_pred, y_true, update=True, callback_fn=None, **kwargs):
        # - `y_pred` must be in the following shape (batch_size, num_categories, ...), float32 possibility
        # - `y_true` must be in the following shape (batch_size, ...) or (batch_size, num_categories, ...), int64.
        # if update is true, update the state. otherwise return the last_update
        if not update:
            assert self.last_update is not None
            return self.last_update

        if callback_fn is not None:
            y_pred, y_true = callback_fn(y_pred, y_true)

        assert (y_pred.ndim - y_true.ndim) in [0, 1]
        if y_pred.device != self._tn.device:
            y_pred = y_pred.to(self._tn.device)
        if y_true.device != self._tn.device:
            y_true = y_true.to(self._tn.device)

        # y_true to [-1,] shape
        if y_pred.ndim == y_true.ndim:
            if y_true.ndim > 2:
                y_true = transpose_move_to_end(y_true, 1)
            y_true = torch.reshape(y_true, (-1, self.num_classes))
            y_true = torch.argmax(y_true, 1)
        else:
            y_true = torch.reshape(y_true, (-1,))
        # y_pred to [-1,] shape
        if y_pred.ndim > 2:
            y_pred = transpose_move_to_end(y_pred, 1)
        y_pred = torch.reshape(y_pred, [-1, self.num_classes])
        if self.has_background:
            y_pred_positive = y_pred[:, 1:]
            y_pred_positive = torch.argmax(y_pred_positive, 1) + 1
            y_pred_flatten = torch.reshape(y_pred, [-1])
            selection = y_pred_positive + \
                        torch.arange(y_pred_positive.shape[0], device=self._tn.device) * self.num_classes
            cond = torch.take(y_pred_flatten, selection) > self.threshold
            y_pred = torch.where(
                cond,
                y_pred_positive,
                torch.zeros([y_pred_positive.shape[0]], dtype=torch.int64, device=self._tn.device))
        else:
            y_pred = torch.argmax(y_pred, 1)

        # compute confusion matrix
        cm = torch.zeros([self.num_classes, self.num_classes], dtype=torch.int64, device=self._tn.device)

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                y_true_flag = y_true == i
                y_pred_flag = y_pred == j
                cm[i, j] += torch.sum((y_true_flag & y_pred_flag).type(torch.int64))

        self.cm += cm
        self.num_samples += y_pred.shape[0]
        tp, fp, fn, tn = self.compute_tp_fp_fn_tn(cm)
        self.last_update = (cm, tp, fp, fn, tn)
        return cm, tp, fp, fn, tn

    def compute_tp_fp_fn_tn(self, cm):
        fps = cm.sum(dim=0) - torch.diag(cm)
        fns = cm.sum(dim=1) - torch.diag(cm)
        tps = torch.diag(cm)
        tns = cm.sum() - (fps + fns + tps)
        self._fp += fps
        self._tp += tps
        self._fn += fns
        self._tn += tns
        if self.num_classes == 2:
            return tps[1], fps[1], fns[1], tns[1]
        return tps, fps, fns, tns


class _Metric(torch.nn.Module):
    def __init__(self,
                 cm=None,
                 num_classes=2,
                 threshold=0.5,
                 has_background=True,
                 epsilon=1e-7,
                 update=True,
                 callback_fn=None):
        super(_Metric, self).__init__()
        # if cm is not None, ignore num_classes and threshold and has_background
        if cm is None:
            self.cm = _ConfusionMatrix(num_classes, threshold, has_background)
        else:
            self.cm = cm
        self.update = update
        self.callback_fn = callback_fn
        self.epsilon = epsilon

    def reset(self):
        if self.update:
            self.cm.reset()

    def __call__(self, y_pred, y_true, **kwargs):
        raise NotImplementedError

    @property
    def result(self):
        raise NotImplementedError


class TruePositive(_Metric):
    def __init__(self, cm=None, num_classes=2, threshold=0.5, has_background=True, update=True, callback_fn=None):
        super(TruePositive, self).__init__(cm=cm,
                                           num_classes=num_classes,
                                           threshold=threshold,
                                           has_background=has_background,
                                           update=update,
                                           callback_fn=callback_fn)

    def __call__(self, y_pred, y_true, **kwargs):
        _, tp, _, _, _ = self.cm(y_pred, y_true, update=self.update, callback_fn=self.callback_fn)
        return tp

    @property
    def result(self):
        return self.cm.tp


class FalsePositive(_Metric):
    def __init__(self, cm=None, num_classes=2, threshold=0.5, has_background=True, update=True, callback_fn=None):
        super(FalsePositive, self).__init__(cm=cm,
                                            num_classes=num_classes,
                                            threshold=threshold,
                                            has_background=has_background,
                                            update=update,
                                            callback_fn=callback_fn)

    def __call__(self, y_pred, y_true, **kwargs):
        _, _, fp, _, _ = self.cm(y_pred, y_true, update=self.update, callback_fn=self.callback_fn)
        return fp

    @property
    def result(self):
        return self.cm.fp


class FalseNegative(_Metric):
    def __init__(self, cm=None, num_classes=2, threshold=0.5, has_background=True, update=True, callback_fn=None):
        super(FalseNegative, self).__init__(cm=cm,
                                            num_classes=num_classes,
                                            threshold=threshold,
                                            has_background=has_background,
                                            update=update,
                                            callback_fn=callback_fn)

    def __call__(self, y_pred, y_true, **kwargs):
        _, _, _, fn, _ = self.cm(y_pred, y_true, update=self.update, callback_fn=self.callback_fn)
        return fn

    @property
    def result(self):
        return self.cm.fn


class TrueNegative(_Metric):
    def __init__(self, cm=None, num_classes=2, threshold=0.5, has_background=True, update=True, callback_fn=None):
        super(TrueNegative, self).__init__(cm=cm,
                                           num_classes=num_classes,
                                           threshold=threshold,
                                           has_background=has_background,
                                           update=update,
                                           callback_fn=callback_fn)

    def __call__(self, y_pred, y_true, **kwargs):
        _, _, _, _, tn = self.cm(y_pred, y_true, update=self.update, callback_fn=self.callback_fn)
        return tn

    @property
    def result(self):
        return self.cm.tn


class Precision(_Metric):
    def __init__(self, cm=None, num_classes=2, threshold=0.5, has_background=True, epsilon=1e-7, update=True,
                 callback_fn=None):
        super(Precision, self).__init__(cm=cm,
                                        num_classes=num_classes,
                                        threshold=threshold,
                                        has_background=has_background,
                                        epsilon=epsilon,
                                        update=update,
                                        callback_fn=callback_fn)

    def __call__(self, y_pred, y_true, **kwargs):
        _, tp, fp, _, _ = self.cm(y_pred, y_true, update=self.update, callback_fn=self.callback_fn)
        precision = self.compute(tp, fp)
        return precision

    @property
    def result(self):
        return self.compute(self.cm.tp, self.cm.fp)

    def compute(self, tp, fp):
        tp = tp.type(torch.float64) + self.epsilon
        fp = fp.type(torch.float64) + self.epsilon
        return tp / (tp + fp)


class Recall(_Metric):
    def __init__(self, cm=None, num_classes=2, threshold=0.5, epsilon=1e-7, update=True, callback_fn=None):
        super(Recall, self).__init__(cm=cm,
                                     num_classes=num_classes,
                                     threshold=threshold,
                                     epsilon=epsilon,
                                     update=update,
                                     callback_fn=callback_fn)

    def __call__(self, y_pred, y_true, **kwargs):
        _, tp, _, fn, _ = self.cm(y_pred, y_true, update=self.update, callback_fn=self.callback_fn)
        recall = self.compute(tp, fn)
        return recall

    @property
    def result(self):
        return self.compute(self.cm.tp, self.cm.fn)

    def compute(self, tp, fn):
        tp = tp.type(torch.float64) + self.epsilon
        fn = fn.type(torch.float64) + self.epsilon
        return tp / (tp + fn)


class Sensitivity(_Metric):
    def __init__(self, cm=None, num_classes=2, threshold=0.5, epsilon=1e-7, update=True, callback_fn=None):
        super(Sensitivity, self).__init__(cm=cm,
                                          num_classes=num_classes,
                                          threshold=threshold,
                                          epsilon=epsilon,
                                          update=update,
                                          callback_fn=callback_fn)

    def __call__(self, y_pred, y_true, **kwargs):
        _, tp, _, fn, _ = self.cm(y_pred, y_true, update=self.update, callback_fn=self.callback_fn)
        sensitivity = self.compute(tp, fn)
        return sensitivity

    @property
    def result(self):
        return self.compute(self.cm.tp, self.cm.fn)

    def compute(self, tp, fn):
        tp = tp.type(torch.float64) + self.epsilon
        fn = fn.type(torch.float64) + self.epsilon
        return tp / (tp + fn)


class Specificity(_Metric):
    def __init__(self, cm=None, num_classes=2, threshold=0.5, epsilon=1e-7, update=True, callback_fn=None):
        super(Specificity, self).__init__(cm=cm,
                                          num_classes=num_classes,
                                          threshold=threshold,
                                          epsilon=epsilon,
                                          update=update,
                                          callback_fn=callback_fn)

    def __call__(self, y_pred, y_true, **kwargs):
        _, _, fp, _, tn = self.cm(y_pred, y_true, update=self.update, callback_fn=self.callback_fn)
        specificity = self.compute(fp, tn)
        return specificity

    @property
    def result(self):
        return self.compute(self.cm.fp, self.cm.tn)

    def compute(self, fp, tn):
        fp = fp.type(torch.float64) + self.epsilon
        tn = tn.type(torch.float64) + self.epsilon
        return tn / (tn + fp)


class DSC(_Metric):
    def __init__(self, cm=None, num_classes=2, threshold=0.5, epsilon=1e-7, update=True, callback_fn=None):
        super(DSC, self).__init__(cm=cm,
                                  num_classes=num_classes,
                                  threshold=threshold,
                                  epsilon=epsilon,
                                  update=update,
                                  callback_fn=callback_fn)

    def __call__(self, y_pred, y_true, **kwargs):
        _, tp, fp, fn, _ = self.cm(y_pred, y_true, update=self.update, callback_fn=self.callback_fn)
        dsc = self.compute(tp, fp, fn)
        return dsc

    @property
    def result(self):
        return self.compute(self.cm.tp, self.cm.fp, self.cm.fn)

    def compute(self, tp, fp, fn):
        tp = tp.type(torch.float64) + self.epsilon
        fp = fp.type(torch.float64) + self.epsilon
        fn = fn.type(torch.float64) + self.epsilon
        return 2 * tp / (2 * tp + fp + fn)


class AUC(torch.nn.Module):
    def __init__(self, cms=None, thresholds=None,
                 num_classes=2, epsilon=1e-7, update=True, callback_fn=None):
        # thresholds and num_classes only used when cms is None
        super(AUC, self).__init__()
        assert cms is not None or thresholds is not None
        if cms is not None:
            self.num_thresholds = len(cms)
            self.cms = sorted(cms, key=lambda cm: cm.threshold)
            self.thresholds = [cm.threshold for cm in self.cms]
            assert len(set([cm.num_classes for cm in self.cms])) == 1, 'num_classes should be the same'
            assert len(set([cm.has_background for cm in self.cms])) == 1 and self.cms[0].has_background, \
                'has_background should all be True'
        else:
            self.num_thresholds = len(thresholds)
            self.thresholds = ((np.logspace(0, 1, self.num_thresholds + 2) - 1) / 9)[1: -1]
            self.cms = [_ConfusionMatrix(num_classes, threshold=t, has_background=True) for t in self.thresholds]
        self.epsilon = epsilon
        self.num_classes = self.cms[0].num_classes
        self.update = update
        self.callback_fn = callback_fn

    def reset(self):
        if self.update:
            for cm in self.cms:
                cm.reset()

    def __call__(self, y_pred, y_true, **kwargs):
        tps, fps, fns, tns = [[], [], [], []]
        for cm in self.cms:
            _, tp, fp, fn, tn = cm(y_pred, y_true, update=self.update, callback_fn=self.callback_fn)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            tns.append(tn)
        auc = self.compute(tps, fps, fns, tns, render_curve=False)
        return auc

    @property
    def result(self):
        tps, fps, fns, tns = [[], [], [], []]
        for cm in self.cms:
            tps.append(cm.tp)
            fps.append(cm.fp)
            fns.append(cm.fn)
            tns.append(cm.tn)
        auc = self.compute(tps, fps, fns, tns, render_curve=False)
        return auc

    @property
    def curve(self):
        tps, fps, fns, tns = [[], [], [], []]
        for cm in self.cms:
            tps.append(cm.tp)
            fps.append(cm.fp)
            fns.append(cm.fn)
            tns.append(cm.tn)
        _, curve = self.compute(tps, fps, fns, tns, render_curve=True)
        return curve

    def compute(self, tps, fps, fns, tns, render_curve=False):
        tprs = list()
        fprs = list()
        # loop for each threshold
        for tp, fp, fn, tn in zip(tps, fps, fns, tns):
            tp, fp, fn, tn = [item.type(torch.float64) + self.epsilon for item in [tp, fp, fn, tn]]
            tprs.append(tp / (tp + fn))
            fprs.append(fp / (tn + fp))

        if render_curve:
            fig_height = 480
            fig_width = 640
            fig_dpi = 100
            fig = matplotlib.pyplot.figure(figsize=[fig_width / fig_dpi, fig_height / fig_dpi], dpi=100)
            ax = fig.gca()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.set_title('ROC curve')
            # left corner zoom in
            fig_zoom = matplotlib.pyplot.figure(figsize=[fig_width / fig_dpi, fig_height / fig_dpi], dpi=100)
            ax_z = fig_zoom.gca()
            ax_z.set_xlim(0.0, 0.2)
            ax_z.set_ylim(0.8, 1.0)
            ax_z.set_xlabel('False positive rate')
            ax_z.set_ylabel('True positive rate')
            ax_z.set_title('ROC curve (zoomed in at top left)')

        if self.num_classes == 2:
            tpr = [1.0] + [tpr.item() for tpr in tprs] + [0.0]
            fpr = [1.0] + [fpr.item() for fpr in fprs] + [0.0]
            auc = sklearn.metrics.auc(fpr, tpr)
            if render_curve:
                pl, = ax.plot(fpr, tpr, 'o-')
                pl_z, = ax_z.plot(fpr, tpr, 'o-')
                ax.legend([pl], ['auc: %1.4f' % auc], loc='lower right')
                ax_z.legend([pl_z], ['auc: %1.4f' % auc], loc='lower right')
            auc = torch.tensor(auc, dtype=torch.float32, device=self.cms[0].cm.device)
        else:
            aucs, pls, pl_zs, pl_ls, pl_zls = [[], [], [], [], []]
            for i in range(self.num_classes):
                tpr = [1.0] + [tpr[i].item() for tpr in tprs] + [0.0]
                fpr = [1.0] + [fpr[i].item() for fpr in fprs] + [0.0]
                auc = sklearn.metrics.auc(fpr, tpr)
                if render_curve:
                    pl, = ax.plot(fpr, tpr, 'o-')
                    pl_z, = ax_z.plot(fpr, tpr, 'o-')
                    pls.append(pl)
                    pl_zs.append(pl_z)
                    pl_ls.append('class_%d(auc: %1.4f)' % (i, auc))
                    pl_zls.append('class_%d(auc: %1.4f)' % (i, auc))
                aucs.append(auc)
            if render_curve:
                ax.legend(pls, pl_ls, loc='lower right')
                ax_z.legend(pl_zs, pl_zls, loc='lower right')
            auc = torch.tensor(aucs, dtype=torch.float32, device=self.cms[0].cm.device)
        if render_curve:
            fig.canvas.draw()
            fig_zoom.canvas.draw()
            fig_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape([fig_height, fig_width, 3])
            fig_zoom_arr = np.frombuffer(fig_zoom.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                [fig_height, fig_width, 3])
            fig_arr = np.concatenate([fig_arr, fig_zoom_arr], axis=1)
            matplotlib.pyplot.close(fig)
            matplotlib.pyplot.close(fig_zoom)
            return auc, fig_arr
        else:
            return auc


class AP(torch.nn.Module):
    def __init__(self, cms=None, thresholds=None,
                 num_classes=2, epsilon=1e-7, update=True, callback_fn=None):
        super(AP, self).__init__()
        assert cms is not None or thresholds is not None
        if cms is not None:
            self.num_thresholds = len(cms)
            self.cms = sorted(cms, key=lambda cm: cm.threshold)
            self.thresholds = [cm.threshold for cm in self.cms]
            assert len(set([cm.num_classes for cm in self.cms])) == 1, 'num_classes should be the same'
            assert len(set([cm.has_background for cm in self.cms])) == 1 and self.cms[0].has_background, \
                'has_background should all be True'
        else:
            self.num_thresholds = len(thresholds)
            self.thresholds = ((np.logspace(0, 1, self.num_thresholds + 2) - 1) / 9)[1: -1]
            self.cms = [_ConfusionMatrix(num_classes, threshold=t, has_background=True) for t in self.thresholds]
        self.epsilon = epsilon
        self.num_classes = self.cms[0].num_classes
        self.update = update
        self.callback_fn = callback_fn

    def reset(self):
        if self.update:
            for cm in self.cms:
                cm.reset()

    def __call__(self, y_pred, y_true, **kwargs):
        tps, fps, fns = [[], [], []]
        for cm in self.cms:
            _, tp, fp, fn, _ = cm(y_pred, y_true, update=self.update, callback_fn=self.callback_fn)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
        ap = self.compute(tps, fps, fns, render_curve=False)
        return ap

    @property
    def result(self):
        tps, fps, fns = [[], [], []]
        for cm in self.cms:
            tps.append(cm.tp)
            fps.append(cm.fp)
            fns.append(cm.fn)
        ap = self.compute(tps, fps, fns, render_curve=False)
        return ap

    @property
    def curve(self):
        tps, fps, fns = [[], [], []]
        for cm in self.cms:
            tps.append(cm.tp)
            fps.append(cm.fp)
            fns.append(cm.fn)
        _, curve = self.compute(tps, fps, fns, render_curve=True)
        return curve

    @property
    def data_dict(self):
        field_names = ['threshold', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'tpr', 'fpr']
        data = []
        for cm in self.cms:
            tp = int(cm.tp.detach().cpu().numpy())
            fp = int(cm.fp.detach().cpu().numpy())
            fn = int(cm.fn.detach().cpu().numpy())
            tn = int(cm.tn.detach().cpu().numpy())
            item = dict()
            item['threshold'] = cm.threshold
            item['tp'] = tp
            item['fp'] = tp
            item['fn'] = fn
            item['tn'] = tn
            item['precision'] = (float(tp) + self.epsilon) / (float(tp) + float(fp) + self.epsilon)
            item['recall'] = (float(tp) + self.epsilon) / (float(tp) + float(fn) + self.epsilon)
            item['tpr'] = (float(tp) + self.epsilon) / (float(tp) + float(fn) + self.epsilon)
            item['fpr'] = (float(fp) + self.epsilon) / (float(tn) + float(fp) + self.epsilon)
            data.append(item)
        return field_names, data

    def compute(self, tps, fps, fns, render_curve=False):
        precisions = list()
        recalls = list()
        # loop for each threshold
        for tp, fp, fn in zip(tps, fps, fns):
            tp, fp, fn = [item.type(torch.float64) + self.epsilon for item in [tp, fp, fn]]
            precisions.append(tp / (tp + fp))
            recalls.append(tp / (tp + fn))

        if render_curve:
            fig_height = 480
            fig_width = 640
            fig_dpi = 100
            fig = matplotlib.pyplot.figure(figsize=[fig_width / fig_dpi, fig_height / fig_dpi], dpi=fig_dpi)
            ax = fig.gca()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('PR curve')

        if self.num_classes == 2:
            precision = [0.0] + [precision.item() for precision in precisions] + [1.0]
            recall = [1.0] + [recall.item() for recall in recalls] + [0.0]
            ap = sklearn.metrics.auc(recall, precision)
            if render_curve:
                pl, = ax.plot(recall, precision, 'o-')
                ax.legend([pl], ['ap: %1.4f' % ap], loc='lower left')
            ap = torch.tensor(ap, dtype=torch.float32, device=self.cms[0].cm.device)
        else:
            aps, pls, pl_ls = [[], [], []]
            for i in range(self.num_classes):
                precision = [0.0] + [precision[i].item() for precision in precisions] + [1.0]
                recall = [1.0] + [recall[i].item() for recall in recalls] + [0.0]
                ap = sklearn.metrics.auc(recall, precision)
                if render_curve:
                    pl, = ax.plot(recall, precision, 'o-')
                    pls.append(pl)
                    pl_ls.append('class_%d(ap: %1.4f)' % (i, ap))
                aps.append(ap)
            if render_curve:
                ax.legend(pls, pl_ls, loc='lower left')
            ap = torch.tensor(aps, dtype=torch.float32, device=self.cms[0].cm.device)
        if render_curve:
            fig.canvas.draw()
            fig_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape([fig_height, fig_width, 3])
            matplotlib.pyplot.close(fig)
            return ap, fig_arr
        else:
            return ap


class HD95(torch.nn.Module):
    def __init__(self, threshold=0.5, epsilon=1e-7):
        super(HD95, self).__init__()
        self.num_samples = 0
        self.threshold = threshold
        self.epsilon = epsilon
        self.hd95_sum = torch.nn.Parameter(torch.zeros([], dtype=torch.float64), requires_grad=False)

    def reset(self):
        self.num_samples = 0
        self.hd95_sum.zero_()

    def __call__(self, y_pred, y_true, **kwargs):
        # - `y_pred` must be in the following shape (batch_size, num_categories, ...), float32 possibility
        # - `y_true` must be in the following shape (batch_size, ...) or (batch_size, num_categories, ...), int64.
        if 'spacing' not in kwargs:
            logging.getLogger('HD95').debug('no spacing specified.')
            spacing = [np.array([0.5] * (y_pred.ndim - 2))] * y_pred.shape[0]
        else:
            spacing = kwargs['spacing'].detach().cpu().numpy()

        assert (y_pred.ndim - y_true.ndim) in [0, 1]
        if y_pred.device != self.hd95_sum.device:
            y_pred = y_pred.to(self.hd95_sum.device)
        if y_true.device != self.hd95_sum.device:
            y_true = y_true.to(self.hd95_sum.device)

        if y_pred.ndim == y_true.ndim:
            y_true = torch.argmax(y_true, 1)
        y_true = y_true > 0.5
        y_pred = torch.argmax(y_pred, 1)
        y_pred = y_pred > 0.5

        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        hd95_vals = []
        for y_p, y_t, sp in zip(y_pred, y_true, spacing):
            if y_p.any() and y_t.any():
                hd95_vals.append(hd95(y_p, y_t, sp))
        self.hd95_sum += sum(hd95_vals)
        self.num_samples += len(hd95_vals)
        return torch.tensor((sum(hd95_vals) + self.epsilon) / (len(hd95_vals) + self.epsilon))

    @property
    def result(self):
        return (self.hd95_sum + self.epsilon) / (self.num_samples + self.epsilon)


def per_target_transform(y_pred, y_true):
    """Transform per-voxel metrics to per-target. This will lead to meaningless true negatives"""
    # - `y_pred` must be in the following shape (batch_size, num_categories, ...), float32 possibility
    # - `y_true` must be in the following shape (batch_size, ...) or (batch_size, num_categories, ...), int64.
    assert y_pred.ndim - y_true.ndim in [0, 1]
    assert y_pred.ndim > 2, 'only image can be transformed to per_target metric'

    device = y_pred.device
    num_classes = y_pred.shape[1]
    assert num_classes == 2, 'now only support binary classes'

    # reduce num_categories axis
    if y_pred.ndim == y_true.ndim:
        y_true = torch.argmax(y_true, 1)
    y_pred = torch.argmax(y_pred, 1)

    def _is_match(center_1, area_1, center_2, area_2):
        ndim = len(center_1)
        if sum([(center_1[i] - center_2[i]) ** 2 for i in range(ndim)]) ** 0.5 < (
                0.62 * (area_1 ** (1 / ndim) + area_2 ** (1 / ndim))):  # for 3d case using 0.62 factor
            return True
        return False

    per_target_preds = []
    per_target_trues = []
    # split batch
    for y_p, y_t in zip(y_pred, y_true):
        assert y_p.shape == y_t.shape
        # pred Morph Close
        y_p = torch.unsqueeze(torch.unsqueeze(y_p, 0), 0).type(torch.float32)
        kernel_size = 7
        padding = 3
        # Dilated
        y_p = torch.nn.MaxPool3d(kernel_size, stride=1, padding=padding)(y_p)
        # Eroded
        y_p = 1.0 - torch.nn.MaxPool3d(kernel_size, stride=1, padding=padding)(1.0 - y_p)
        y_p = torch.squeeze(torch.squeeze(y_p, 0), 0).type(torch.int64)

        y_p = y_p.detach().cpu().numpy()
        y_t = y_t.detach().cpu().numpy()
        region_area_threshold = 10
        y_p_label = measure.label(y_p)
        y_p_props = measure.regionprops(y_p_label)
        y_p_props = [item for item in y_p_props if item.area > region_area_threshold]  # reduce small noise
        y_t_label = measure.label(y_t)
        y_t_props = measure.regionprops(y_t_label)
        y_t_props = [item for item in y_t_props if item.area > region_area_threshold]  # reduce small noise

        t_matches = []
        target_pred = []
        target_true = []
        for i in range(len(y_p_props)):
            i_match = False
            for j in range(len(y_t_props)):
                if _is_match(y_p_props[i].centroid, y_p_props[i].area, y_t_props[j].centroid, y_t_props[j].area):
                    i_match = True
                    t_matches.append(j)
            if not i_match:  # false positive
                target_pred.append(1)
                target_true.append(0)
        t_matches = set(t_matches)
        for _ in range(len(t_matches)):  # true positive
            target_pred.append(1)
            target_true.append(1)
        for _ in range(len(y_t_props) - len(t_matches)):  # false negative
            target_pred.append(0)
            target_true.append(1)

        per_target_preds.append(target_pred)
        per_target_trues.append(target_true)
    max_len = max([len(item) for item in per_target_preds])
    if max_len == 0:
        max_len = 1  # add one true negative if no targets
    for i in range(len(per_target_preds)):
        for _ in range(max_len - len(per_target_preds[i])):  # pseudo true negative to unify batch len
            per_target_preds[i].append(0)
            per_target_trues[i].append(0)
    per_target_preds = torch.tensor(per_target_preds, dtype=torch.int64, device=device)
    per_target_trues = torch.tensor(per_target_trues, dtype=torch.int64, device=device)
    per_target_preds = one_hot(per_target_preds, 2, axis=1)
    per_target_trues = one_hot(per_target_trues, 2, axis=1)
    return per_target_preds, per_target_trues


def get_evaluation_metric(config, logger, device):
    """
    Returns the evaluation metric function based on provided configuration
    The first of precision, recall... and first of auc, pr can update the inner confusion matrix, make sure to update in order
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """
    metrics = config['eval']['metrics']
    if not isinstance(metrics, list):
        metrics = [metrics]
    curves = config['eval'].get('curves', list())
    threshold = config['eval'].get('probability_threshold', 0.5)
    num_classes = config['model']['num_classes']

    metrics_dict = {}
    curves_dict = {}
    cm = _ConfusionMatrix(num_classes=num_classes, threshold=threshold).to(device)
    cm_per_target = None
    cms = None
    cms_per_target = None
    if any([item.startswith('per_target_') for item in metrics]):
        cm_per_target = _ConfusionMatrix(num_classes=num_classes, threshold=threshold).to(device)
    if any(['auc' in item for item in metrics]) or any(['ap' in item for item in metrics]) \
            or any(['pr' in item for item in curves]) or any(['pr' in item for item in curves]):
        thresholds_type = config['eval'].get('thresholds_type', 'logspace')
        if thresholds_type == 'logspace':
            thresholds = ((np.logspace(0, 1, config['eval']['num_thresholds'] + 2) - 1) / 9)[1: -1]
        elif thresholds_type == 'logspace_pro':
            thresholds = ((np.logspace(0, 1, config['eval']['num_thresholds'] + 2, base=100) - 1) / 99)[1: -1]
        elif thresholds_type == 'linspace':
            thresholds = np.linspace(0.0, 1.0, config['eval']['num_thresholds'] + 2)[1: -1]
        elif thresholds_type == 'uline':
            thresholds = (((np.logspace(0, 1, config['eval']['num_thresholds'] // 2 + 2,
                                        base=10000000000) - 1) / 9999999999)[1: -1]) / 2
            if config['eval']['num_thresholds'] % 2 == 1:
                thresholds = np.append(thresholds, 0.5)
            for i in range(config['eval']['num_thresholds'] // 2 - 1, -1, -1):
                thresholds = np.append(thresholds, 1.0 - thresholds[i])
        else:
            logger.critical('thresholds_type is not supported: %s' % thresholds_type)
            exit(1)
        cms = [_ConfusionMatrix(num_classes, t, True).to(device) for t in thresholds]
        cms_per_target = None
        if any([item.startswith('per_target_') for item in curves]) or any(
                item.startswith('per_target_') for item in metrics):
            cms_per_target = [_ConfusionMatrix(num_classes, t, True).to(device) for t in thresholds]
    update_flags = [True, True, True, True]  # single, multiple, per_target_single, per_target_multiple
    for metric_name in metrics:
        if metric_name.startswith('per_target_'):
            callback_fn = per_target_transform
            update_flag_id = 2
            used_cm = cm_per_target
            used_cms = cms_per_target
        else:
            callback_fn = None
            update_flag_id = 0
            used_cm = cm
            used_cms = cms
        if metric_name.endswith('tp'):
            metrics_dict[metric_name] = TruePositive(used_cm, update=update_flags[update_flag_id],
                                                     callback_fn=callback_fn).to(device)
            update_flags[update_flag_id] = False
        elif metric_name.endswith('fp'):
            metrics_dict[metric_name] = FalsePositive(used_cm, update=update_flags[update_flag_id],
                                                      callback_fn=callback_fn).to(device)
            update_flags[update_flag_id] = False
        elif metric_name.endswith('fn'):
            metrics_dict[metric_name] = FalseNegative(used_cm, update=update_flags[update_flag_id],
                                                      callback_fn=callback_fn).to(device)
            update_flags[update_flag_id] = False
        elif metric_name.endswith('tn'):
            metrics_dict[metric_name] = TrueNegative(used_cm, update=update_flags[update_flag_id],
                                                     callback_fn=callback_fn).to(device)
            update_flags[update_flag_id] = False
        elif metric_name.endswith('precision'):
            metrics_dict[metric_name] = Precision(used_cm, update=update_flags[update_flag_id],
                                                  callback_fn=callback_fn).to(
                device)
            update_flags[update_flag_id] = False
        elif metric_name.endswith('recall'):
            metrics_dict[metric_name] = Recall(used_cm, update=update_flags[update_flag_id],
                                               callback_fn=callback_fn).to(
                device)
            update_flags[update_flag_id] = False
        elif metric_name.endswith('sensitivity'):
            metrics_dict[metric_name] = Sensitivity(used_cm, update=update_flags[update_flag_id],
                                                    callback_fn=callback_fn).to(device)
            update_flags[update_flag_id] = False
        elif metric_name.endswith('specificity'):
            metrics_dict[metric_name] = Specificity(used_cm, update=update_flags[update_flag_id],
                                                    callback_fn=callback_fn).to(device)
            update_flags[update_flag_id] = False
        elif metric_name.endswith('dsc'):
            metrics_dict[metric_name] = DSC(used_cm, update=update_flags[update_flag_id], callback_fn=callback_fn).to(
                device)
            update_flags[update_flag_id] = False
        elif metric_name.endswith('auc'):
            metrics_dict[metric_name] = AUC(used_cms, update=update_flags[update_flag_id + 1],
                                            callback_fn=callback_fn).to(
                device)
            update_flags[update_flag_id + 1] = False
        elif metric_name.endswith('ap'):
            metrics_dict[metric_name] = AP(used_cms, update=update_flags[update_flag_id + 1],
                                           callback_fn=callback_fn).to(
                device)
            update_flags[update_flag_id + 1] = False
        elif metric_name.endswith('hd95'):
            metrics_dict[metric_name] = HD95(threshold).to(device)
        else:
            logger.error('Unrecognized metric: %s' % metric_name)
            continue
    for curve_name in curves:
        if curve_name.startswith('per_target_'):
            callback_fn = per_target_transform
            update_flag_id = 2
            used_cm = cm_per_target
            used_cms = cms_per_target
        else:
            callback_fn = None
            update_flag_id = 0
            used_cm = cm
            used_cms = cms
        if curve_name.endswith('roc'):
            if curve_name.replace('roc', 'auc') not in metrics_dict:
                logger.warning('%s not in metrics but %s in curves. Adding %s to metrics'
                               % (curve_name.replace('roc', 'auc'), curve_name, curve_name.replace('roc', 'auc')))
                metrics_dict[curve_name.replace('roc', 'auc')] = AUC(used_cms, update=update_flags[update_flag_id + 1],
                                                                     callback_fn=callback_fn)
                update_flags[update_flag_id + 1] = False
            curves_dict[curve_name] = metrics_dict[curve_name.replace('roc', 'auc')]
        elif curve_name.endswith('pr'):
            if curve_name.replace('pr', 'ap') not in metrics_dict:
                logger.warning('%s not in metrics but %s in curves. Adding %s to metrics'
                               % (curve_name.replace('pr', 'ap'), curve_name, curve_name.replace('pr', 'ap')))
                metrics_dict[curve_name.replace('pr', 'ap')] = AP(used_cms, update=update_flags[update_flag_id + 1],
                                                                  callback_fn=callback_fn)
                update_flags[update_flag_id + 1] = False
            curves_dict[curve_name] = metrics_dict[curve_name.replace('pr', 'ap')]
    if len(metrics_dict) == 0:
        logger.critical('No metric is added')
        exit(1)
    return metrics_dict, curves_dict
