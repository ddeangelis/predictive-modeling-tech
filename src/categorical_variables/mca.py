'''
    File name: mca.py
    Author: Tyche Analytics Co.
    Note: Multiple Correspondence Analysis, e.g. PCA for categorical data
'''
import numpy as np
import scipy
from collections import Counter
from math import sqrt

def make_burt(df, normalize=False):
    cols = df.columns
    N = len(df)
    Js = [len(df[col].unique()) for col in cols]
    K = len(cols)
    elements = [(col,v) for col in cols for v in sorted(df[col].unique())]
    index = {elem:i for i, elem in enumerate(elements)}
    J = M = sum(Js)
    B = np.zeros((M, M), dtype=int)
    for c1, col1 in enumerate(cols):
        for c2, col2 in enumerate(cols):
            counts = Counter(zip(df[col1], df[col2]))
            for (v1, v2), count in counts.items():
                i,j = index[col1, v1], index[col2, v2]
                B[i,j] = count
    if normalize:
        ps = np.array([B[i,i] for i in range(len(B))])/N
        return B/N - np.outer(ps, ps)
    return B

def mca(df, dims=2, normalize=False):
    N = len(df)
    cols = df.columns
    K = len(cols)
    print("Making Burt Matrix")
    B = make_burt(df, normalize=normalize)
    M, M = B.shape
    J = M
    print("Diagonalizing Burt Matrix")
    w, v = np.linalg.eigh(B)
    print("Diagonalization finished")
    print("eigenvalues:", w[-dims:])
    #w, v = w.real, v.real
    lambs_I = np.sqrt(w.real + 10**-10)
    lambs_c = (K/(K-1)*(lambs_I * (lambs_I > 1/K) - 1/K))**2
    avg_inertia = K / (K-1) * (sum(lambs_I**2) - (J-K)/(K**2))
    taus_c = lambs_c / avg_inertia
    assert(np.all(v[:,-dims:].imag == 0))
    v0v1 = v[:,-dims:].real
    # make indicator
    elements = [(col,v) for col in cols for v in sorted(df[col].unique())]
    index = {elem:i for i, elem in enumerate(elements)}
    transform_data = (M, dims, index, v0v1)
    return transform_data

# Thu Nov 9 14:36:28 EST 2017 We refactor the mca function to return
# the necessary data elements to make an MCA transformer, to be passed
# on to mca_transformer below, rather than simply returning the
# function directly.  We do this so that we can pickle the transformer
# data and load it into make_transformer on the production server at
# runtime, because pickling python code can lead to unpredictable bugs.

def mca_transformer(transform_data):
    """accept data from an mca call and make a transformer"""
    M, dims, index, v0v1 = transform_data
    def transform(dfp):
        # dims, index, v0v1
        P = np.zeros((len(dfp), dims), dtype=float)
        print("transforming")
        for i, (_, row) in (enumerate(dfp.iterrows())):
            ivec = np.zeros(M)
            for col, val in zip(row.index, row):
                if (col, val) in index:
                    ivec[index[col, val]] = 1
            proj = ivec.dot(v0v1)
            assert(all(proj.imag == 0))
            P[i,:] = proj.real
        return P
    return transform

def mca_graphical_example(df, label_expansion=1, label_cutoff = 1, colors=None):
    elements = [(col,v) for col in df.columns for v in sorted(df[col].unique())]
    dims = 2
    transform_data = mca(df, dims=dims)
    transform = mca_transformer(transform_data)
    P = transform(df)
    B = make_burt(df, normalize=True)
    M, M = B.shape
    w, v = np.linalg.eig(B)
    assert(np.all(v[:,:dims].imag == 0))
    v0v1 = v[:,:dims].real
    plt.scatter(P[:,0], P[:,1], alpha=0.01, c=colors)
    locations = []
    if label_expansion > 0:
        for i, (col,v) in enumerate(elements):
            x = np.zeros(M)
            x[i] = 1
            xp = x.dot(v0v1) * label_expansion
            norm = sqrt(xp[0]**2 + xp[1]**2)
            if norm > label_cutoff:
                x = xp[0] + 0 #random.random() * 0.1
                y = xp[1] + 3 * (random.random() - 0.5)
                plt.plot([0, xp[0]], [0, xp[1]])
                plt.annotate(col + ":" + v, (x, y), size='x-large')
    plt.xlabel("First Component", size='xx-large')
    plt.ylabel("Second Component", size='xx-large')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()
