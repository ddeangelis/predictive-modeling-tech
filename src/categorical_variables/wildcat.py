'''
    File name: wildcat.py
    Author: Tyche Analytics Co.
    Note: Wildcat is a library for handling categorical variables
'''

from collections import defaultdict
from utils3 import mean, variance, se
from scipy.sparse import lil_matrix
from scipy.optimize import minimize_scalar
from itertools import product
from tqdm import *
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold
import re
import numpy as np

def mean(xs):
    xs = list(xs)
    return sum(xs)/len(xs)

def js_estimate(zs, ys, ppc=0, npc=0):
    d = defaultdict(lambda:([0]*npc) + [1] * ppc)
    for z, y in zip(zs, ys):
        d[z].append(y)
    group_means = list(map(mean, d.values()))
    xbar = mean(group_means)
    tau_hat2 = variance(group_means) if len(group_means) > 1 else 0
    d_adj = {z: (lambda B:((1-B)*mean(vs) +
                           B*xbar))(se(vs, correct=False)**2 /
                                    (se(vs, correct=False)**2 + tau_hat2))
             for z, vs in d.items()}
    return defaultdict(lambda: xbar, d_adj)

def js_transformer(zs, ys, ppc=0, npc=0):
    d = js_estimate(zs, ys, ppc, npc)
    return lambda xs: [d[x] for x in xs]

def mle_estimate(zs, ys, ppc=0, npc=0):
    ybar = mean(ys)
    d = defaultdict(lambda:([0]*npc) + [1] * ppc)
    for z, y in zip(zs, ys):
        d[z].append(y)
    d = {z:mean(vs) for z,vs in d.items()}
    return defaultdict(lambda: ybar, d)

def zip_estimate(zs, ys, C=1, classifier="logreg", penalty='l2'):
    """Regress binary outcome against zip using regression of the form:
    y ~ z1 + z2 + z3 + z4 + z5.
    """
    background_rate = mean(ys)
    # js = pd.notnull(zs)
    # zs = zs[js]
    # ys = ys[js]
    def to_row(z):
        arr = [0]*111110
        offset = 0
        for i in range(1, 5 + 1):
            zi = z[:i]
            j = int(zi) + offset
            arr[j] = 1
            offset += 10**i
        return arr
    
    N = len(zs)
    A = lil_matrix((N, 111110))
    valids = 0
    for i, z in tqdm(enumerate(zs), total=N):
        if not is_valid_zip(z):
            continue
        valids += 1
        offset = 0
        for d in range(1, 5 + 1):
            zi = z[:d]
            j = int(zi) + offset
            A[i,j] = 1
            offset += 10**d
    print("valids:", valids)
    if classifier=="logreg":
        clf = LogReg(C=C, penalty=penalty)
    elif classifier=="ridge":
        clf = Ridge(alpha=1/C)
    else:
        raise Exception("didn't recognize classifier:", classifier)
    print("fitting logreg")
    clf.fit(A, ys)
    all_zs = ["".join(x) for k in range(1, 5+1)
              for x in product(*("0123456789" for _ in range(k)))]
    all_A = lil_matrix((111110, 111110))
    for i, z in tqdm(enumerate(all_zs), total=111110):
        offset = 0
        for d in range(1, len(z) + 1):
            zi = z[:d]
            j = int(zi) + offset
            all_A[i,j] = 1
            offset += 10**d
    if classifier == "logreg":
        all_yhats = clf.predict_proba(all_A)[:,1]
    else:
        all_yhats = clf.predict(all_A)
    ml_dict = {z:yhat for z, yhat in zip(all_zs, all_yhats)}
    return defaultdict(lambda: background_rate, ml_dict)

def zip_estimate_auto(zs, ys, classifier='logreg', n_splits=10, verbose=False, cutoff=3):
    def perf(C, cutoff=cutoff):
        kf = KFold(n_splits=n_splits, shuffle=True)
        train_losses = []
        test_losses = []
        for i, (train, test) in enumerate(kf.split(zs)):
            if i >= cutoff:
                break
            train_zs, test_zs = zs.iloc[train], zs.iloc[test]
            train_ys, test_ys = ys.iloc[train], ys.iloc[test]
            assert(len(train_zs) + len(test_zs) == len(zs))
            assert(len(train_ys) + len(test_ys) == len(ys))
            d = zip_estimate(train_zs, train_ys, classifier='logreg', C=C)
            train_yhat = [d[z] for z in train_zs]
            test_yhat = [d[z] for z in test_zs]
            train_auc = roc_auc_score(train_ys, train_yhat)
            test_auc = roc_auc_score(test_ys, test_yhat)
            train_loss = log_loss(train_ys, train_yhat)
            test_loss = log_loss(test_ys, test_yhat)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            if verbose:
                print("C:", C)
                print("train LOSS:", train_loss)
                print("test_loss:", test_loss, mean(test_losses))
                print("train auc:", train_auc)
                print("test auc:", test_auc)

        return mean(test_losses)
    f = lambda log10_C: -perf(10**log10_C)
    perfs = [f(log_C) for log_C in np.linspace(-5, 5, 10)]
    print(perfs)
    opt_C = minimize_scalar(f, bounds=(-3, 3))
    return opt_C

def woe_estimate(xs, ys):
    d = defaultdict(list)
    for x, y in zip(xs, ys):
        d[x].append(y)
    POS = sum(ys)
    NEG = len(ys) - POS
    p = POS / (POS + NEG)
    base_woe = log(NEG/POS)
    num_pos = lambda xs:sum(xs)
    num_neg = lambda xs:len(xs) - sum(xs)
    woe_dict = {k:log(((num_pos(vs) + p)/POS) / (num_neg(vs) + (1-p))/NEG)
                for k, vs in d.items()}
    return defaultdict(lambda:base_woe, woe_dict)

def is_valid_zip(z):
    "match five digits, optionally dash, optionally four more"
    regexp = "^[0-9]{5}(-?[0-9]{4})?$" 
    return isinstance(z, str) and not (re.match(regexp, z) is None)
