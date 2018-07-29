'''
    File name: zicata.py
    Author: Tyche Analytics Co.
'''
import pandas as pd
from itertools import product
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.preprocessing import StandardScaler as SS
from utils3 import choose2, fdr
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from math import sqrt

def compare_to_log(xs, bins=100):
    if any(pd.isnull(xs)):
        print("Warning, nans in xs")
    xs = xs.fillna(xs.median())
    plt.subplot(1, 2, 1)
    plt.hist(xs, bins=bins)
    plt.subplot(1, 2, 2)
    plt.hist(np.log10(xs + 1), bins=bins)
    plt.show()
    
def load_zcta_df():
    zcta_df = pd.read_csv("<PATH>/zcta_master.csv", dtype={'zcta5':str}, thousands=",")
    zcta_df.index = zcta_df.zcta5
    zcta_df.columns = [col.lower() for col in zcta_df.columns]
    cols = "intptlat intptlon areasqmi totpop10 medianage pctunder18".split()
    cols += "pctover65 pctwhite1 pctblack1 pctasian1 pcthispanicpop tothhs".split()
    cols += "medianhhinc pctpoor pctgrpquarters pctincollege".split()
    cols += "pctbachelorsormore pctforeignborn occhus pctrenterocc medianhvalue".split()
    cur_cols = ['medianhhinc', 'medianhvalue']
    for cur_col in cur_cols:
        zcta_df[cur_col] = zcta_df[cur_col].apply(convert_currency)
    log_cols = "areasqmi totpop10 tothhs medianhhinc occhus medianhvalue".split()
    zcta_df = zcta_df[cols]
    for lc in log_cols:
        zcta_df[lc] = np.log10(zcta_df[lc] + 1)
    zcta_df = zcta_df.rename(columns={lc:"log"+lc for lc in log_cols})
    return zcta_df

def make_zip_pca(zcta_df):
    pca = PCA(whiten=False)
    ss = SS()
    X = pca.fit_transform(ss.fit_transform(zcta_df.fillna(zcta_df.median())))
    return X

def zip_space(z, comp, X=None, zcta_df=None, verbose=False):
    """Given a zip z, return the compth principal component.  If z is not
    present, coarsen and complement z until at least one zip is found,
    then return the average, averaging naively by zip (i.e. not by population)
    """
    zs = [z]
    d = 5
    while True:
        js = zcta_df.index.isin(zs)
        if any(js):
            break
        else:
            d -= 1
            if verbose:
                print("coarsening zip to:", z[:d])
            zs = complement_zip(z[:d])
    return X[js,comp].mean()

def convert_currency(x):
    if type(x) is str:
        return int(x.replace("$", "").replace(",", ""))
    else:
        return x
    
def zip_feature(z, feature):
    d = 5
    while True:
        #print("recoursing to", d, "digits for", z)
        comps = complement_zip(z[:d])
        comp_df = zcta_df.ix[comps]
        avg = comp_df[feature].mean()
        if len(comp_df) > 0 and pd.notnull(avg):
            return avg
        else:
            d -= 1

def complement_zip(p_zip):
    d = len(p_zip)
    return [p_zip + "".join(pro) for pro in product(*("0123456789" for _ in range(5-d)))]

    
def pca_dict_experiment():
    pca = PCA(whiten=True)
    X = pca.fit_transform(zcta_df.fillna(zcta_df.median()))
    idx_dict = {z:i for i,z in enumerate(zcta_df.index)}
    def pca_comp(z, j):
        if z in idx_dict:
            return X[idx_dict[z], j]
        else:
            d = 4
            while True:
                comps = complement_zip(z[:d])
                eyes = [idx_dict[comp] for comp in comps if comp in idx_dict]
                if eyes:
                    return mean(X[eyes, j])
                else:
                    d -= 1
    return pca_comp

def pca_plot(col_nums=None, ann_cutoff=1/3):
    pca = PCA(whiten=False)
    ss = SS()
    X = pca.fit_transform(ss.fit_transform(zcta_df.fillna(zcta_df.median())))
    ann_factor = 20
    cols = zcta_df.columns
    if col_nums is None:
        col_nums = list(range(len(cols)))
    N = len(col_nums) + 1
    for coli in col_nums:
        for colj in col_nums:
            ploti = col_nums.index(coli)
            plotj = col_nums.index(colj)
            print(ploti, plotj, (plotj * N) + (ploti % N) + 1)
            plt.subplot(N, N, (plotj * N) + (ploti % N) + 1)
            plt.scatter(X[:,coli], X[:,colj], s=0.1)
            plt.xlabel("PC %s" % coli)
            plt.ylabel("PC %s" % colj)
            for i, col in enumerate(cols):
                arr = np.zeros(len(cols))
                arr[i] = 1
                proj = pca.transform([arr])
                x, y = proj[0][[coli, colj]]
                norm = sqrt(x**2 + y**2)
                print("col", i, "has norm", norm, "in graph", coli, colj)
                if norm > ann_cutoff:
                    plt.plot([0, ann_factor * x], [0, ann_factor * y])
                    plt.annotate(col, (ann_factor * x, ann_factor * y))
    plt.subplot(N,N,(N)**2)
    plt.plot(pca.explained_variance_ratio_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    for comp in range(len(pca.explained_variance_ratio_)):
        print("-"*80)
        print(comp)
        print("explained variance ratio:", pca.explained_variance_ratio_[comp])
        sorted_loadings = sorted(zip(zcta_df.columns, pca.components_[comp]), key=lambda xy:xy[1], reverse=True)
        for col, load in sorted_loadings:
            print(col, load)
    plt.show()

class ZipSpacer(object):
    def __init__(self):
        self.zcta_df = load_zcta_df()
        self.X = make_zip_pca(self.zcta_df)

    def zip_space(self, z, comp, verbose=False):
        """Given a zip z, return the compth principal component.  If z is not
        present, coarsen and complement z until at least one zip is found,
        then return the average, averaging naively by zip (i.e. not by population)
        """
        zs = [z]
        d = 5
        while True:
            js = self.zcta_df.index.isin(zs)
            if any(js):
                break
            else:
                d -= 1
                if verbose:
                    print("coarsening zip to:", z[:d])
                zs = complement_zip(z[:d])
        return self.X[js,comp].mean()

    def zip_space_col(self, zs, comp, verbose=False):
        """zip space, but on an array of possibly redundant zips"""
        zs = [z if type(z) is str else "" for z in zs]
        unique_zs = set(zs)
        z_dict = {z:self.zip_space(z, comp, verbose=verbose) for z in unique_zs}
        ys = [z_dict[z] for z in zs]
        return ys

    def get_feature(self, z, feature, verbose=False):
        if not type(z) is str:
            return self.zcta_df[feature].mean()
        zs = [z]
        d = 5
        while True:
            js = self.zcta_df.index.isin(zs)
            if self.zcta_df[feature][js].any():
                break
            else:
                d -= 1
                if verbose:
                    print("coarsening zip to:", z[:d])
                zs = complement_zip(z[:d])
        return self.zcta_df[feature][js].mean()

    def zip_feature_col(self, zs, feature, verbose=False):
        """zip space, but on an array of possibly redundant zips"""
        zs = [z if type(z) is str else "" for z in zs]
        unique_zs = sorted(set(zs))
        z_dict = {z:self.get_feature(z, feature, verbose=verbose) for z in unique_zs}
        ys = [z_dict[z] for z in zs]
        return ys

def visualize_dataset(zs, ys):
    zip_spacer = ZipSpacer()
    num_comps = 10
    X = np.array(transpose([zip_spacer.zip_space_col(zs, i) for i in trange(num_comps)]))
    rs = []
    ps = []
    for i in range(num_comps):
        r, p = pearsonr(X[:,i], ys)
        rs.append(r)
        ps.append(p)
    q = fdr(ps)
    sig_cols = []
    for i in range(num_comps):
        sig = int(ps[i] < q)
        print(i, rs[i], ps[i], "*" * sig)
        if sig:
            sig_cols.append(i)
    N = len(sig_cols)
    for k, (i, j) in enumerate(choose2(range(N))):
        print(i, j)
        plt.subplot(N, N, i*N + j)
        sci = sig_cols[i]
        scj = sig_cols[j]
        plt.scatter(X[:,scj], X[:,sci], color=['r' if y else 'b' for y in ys], s=0.1)
        plt.xlabel(j)
        plt.ylabel(i)
    plt.tight_layout()
    plt.show()
    

class ZicataEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, n_comps=10, pca=True, feature_selection=True):
        self.zip_spacer = ZipSpacer()
        self.n_comps = n_comps
        self.pca = pca
        self.feature_selection = feature_selection
        self.features = None
            
    def fit(self, zs, ys):
        if not self.feature_selection:
            self.features = range(max(self.n_comps, len(self.zip_spacer.zcta_df.columns)))
            return self
        elif self.feature_selection == True:
            if self.pca:
                X = np.array(transpose([self.zip_spacer.zip_space_col(zs, i)
                                        for i in trange(self.n_comps)]))
            else:
                X = np.array(transpose([self.zip_spacer.zip_feature_col(zs, feat)
                                    for feat in tqdm(self.zip_spacer.zcta_df.columns)]))
            corrs = [pearsonr(X[:,i], ys) for i in range(X.shape[1])]
            ps = [p for (r, p) in corrs]
            q = fdr(ps)
            self.features = [i for i, p in enumerate(ps) if p <= q]
            print("selected features:", self.features)
            return self
        else:
            self.features = self.feature_selection
            return self

    def transform(self, zs, pca=None):
        if pca is None:
            pca = self.pca
        if pca:
            return np.array(transpose([self.zip_spacer.zip_space_col(zs, i)
                                       for i in trange(self.n_comps) if i in self.features]))
        else:
            return np.array(transpose([self.zip_spacer.zip_feature_col(zs, feat)
                                       for i, feat
                                       in tqdm(enumerate(self.zip_spacer.zcta_df.columns))
                                       if i in self.features]))

def show(x):
    print(x)
    return x

def log_reg_experiment(zs, ys):
    zicata_estimator = ZicataEstimator()
    X = zicata_estimator.transform(zs, pca=False)
    lr = LogRegCV(penalty='l1', solver='liblinear', verbose=2)
    lr.fit(X, ys)
    yhat = lr.predict_proba(X)[:,1]
    X_pca = zicata_estimator.transform(zs, pca=True)
    lr_pca = LogRegCV(penalty='l1', solver='liblinear', verbose=2)
    lr_pca.fit(X_pca, ys)
    yhat_pca = lr_pca.predict_proba(X_pca)[:,1]
    
