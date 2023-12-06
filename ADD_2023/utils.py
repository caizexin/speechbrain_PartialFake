import numpy as np 
import sys
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def EER(labels, scores):
    """
    Computes EER (and threshold at which EER occurs) given a list of (gold standard) True/False labels
    and the estimated similarity scores by the verification system (larger values indicates more similar)
    Sources: https://yangcha.github.io/EER-ROC/ & https://stackoverflow.com/a/49555212/1493011
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=True)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer * 100


def compute_tp_tn_fn_fp(y_act, y_pred):
	'''
	True positive - actual = 1, predicted = 1
	False positive - actual = 1, predicted = 0
	False negative - actual = 0, predicted = 1
	True negative - actual = 0, predicted = 0
	'''
	tp = sum((y_act == 1) & (y_pred == 1))
	tn = sum((y_act == 0) & (y_pred == 0))
	fn = sum((y_act == 1) & (y_pred == 0))
	fp = sum((y_act == 0) & (y_pred == 1))
	return tp, tn, fp, fn

def f1score(labels, predicts):
    predicts = predicts[:labels.shape[0]]
    tp, tn, fp, fn = compute_tp_tn_fn_fp(labels, predicts)
    return 2*tp*100 / (2*tp + fn + fp)

