'''
    File name: zicata_utils.py
    Author: Tyche Analytics Co.
'''
def fdr(ps,alpha=0.05):
    """Given a list of p-values and a desired significance alpha, find the
    adjusted q-value such that a p-value less than q is expected to be
    significant at alpha, via the Benjamini-Hochberg method
    (appropriate for independent tests).
    """
    ps = sorted(ps)
    m = len(ps)
    ks = [k for k in range(m) if ps[k]<= (k+1)/float(m)*alpha] #k+1 because pvals are 1-indexed.
    K = max(ks) if ks else None
    return ps[K] if K else None #if none are significant
