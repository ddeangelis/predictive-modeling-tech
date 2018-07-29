'''
    File name: streetscope_visualization.py
    Author: Tyche Analytics Co.
'''
from streetscope import StreetScope
import pandas as pd
from collections import Counter, defaultdict
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor as KNN
from tqdm import *

def load_amenity_df():
    a_df = pd.read_csv("amenity_coords.csv")
    a_df.columns = [col.strip() for col in a_df.columns]
    return a_df

def produce_visualization():
    N = 10000
    a_df = load_amenity_df()
    lat_lons = a_df[['lat', 'lon']].sample(N)
    streetscope = STREETSCOPE()
    counts = []
    for i, (lat, lon) in lat_lons.iterrows():
        print(lat, lon)
        count = streetscope.lookup_lat_lon(lat, lon)
        count = defaultdict(int, {k.lower():v for (k, v) in count.items()})
        counts.append(count)
    all_counts = defaultdict(int)
    
    for count in counts:
        for k, v in count.items():
            all_counts[k] += v

    print("total amenity types:", len(all_counts))
    sorted_counts = sorted(all_counts.items(), key=lambda kv:kv[1], reverse=True)
    sorted_types = [k for (k,v) in sorted_counts]
    plt.plot(sorted(all_counts.values(), reverse=True))
    plt.loglog()
    plt.xlabel("Amenity Rank", fontsize='large')
    plt.ylabel("Amenity Count", fontsize='large')
    a_types = ["place_of_worship", "school", "bar", "motorcycle_parking", "tattoo"]

    for a_type in a_types:
        x, y = (sorted_types.index(a_type), all_counts[a_type])
        plt.plot(x, y, 'o', color='b')
        plt.annotate(a_type, (x*1.1, y*1.1))
    
    plt.title("Amenity Count/Rank Distribution, with Selected Examples", fontsize='large')
    plt.savefig("amenity_frequency_rank_plot.png", dpi=300)
    plt.close()
    top_amenities = sorted_types[:100]
    M = np.log10(np.matrix([[count[atype] for atype in top_amenities] for count in counts])+1)
    pca = PCA(n_components=10)
    X = pca.fit_transform(M)
    plt.plot(pca.explained_variance_ratio_[:9], color='b', label='Variance')
    plt.plot(range(8, 10), pca.explained_variance_ratio_[8:10], linestyle='--', color='b')
    plt.plot(np.cumsum(pca.explained_variance_ratio_[:9]), color='g',
             label='Cumulative variance')
    s = sum(pca.explained_variance_ratio_[:8])
    plt.plot(range(8, 10), s + np.cumsum(pca.explained_variance_ratio_[8:10]),
             linestyle='--', color='g')
    #plt.semilogy()
    plt.ylim(0, 1)
    plt.xlabel("Principal Components (Truncated)", fontsize='large')
    plt.ylabel("% Variance Explained", fontsize='large')
    plt.legend()
    plt.title("Explained Variance by Principle Component")
    plt.savefig("pca_explained_variance.png", dpi=300)
    plt.close()
    plt.scatter(X[:,0], X[:,1], s=1)

    for i, amen in enumerate(top_amenities[:5]):
        x, y = pca.components_[:2,i]
        zoom = 10
        x_offset = 0.25
        plt.arrow(0, 0, x*zoom, y*zoom, color='r', head_width=0.10)
        plt.annotate(amen, (x*zoom + x_offset, y*zoom), fontsize='large')

    plt.xlabel("First Principal Component", fontsize='large')
    plt.ylabel("Second Principal Component", fontsize='large')
    plt.title("Locations in Reduced Amenity Space with Selected Loadings", fontsize='large')
    plt.xlim(-2, 10)
    plt.ylim(-3.5, 10)
    plt.savefig("reduced_amenity_space.png", dpi=300)
    
    # loss analysis
    lat_lons, addresses = analyze_addresses()
    loc_amenities = get_amenities(lat_lons, streetscope)
    locs = [loc for loc, a in loc_amenities.items() if a
        and all(pd.notnull(df[df.Location==loc].LossCost2))]
    loc_amenities = {loc:counts for loc, counts in loc_amenities.items() if loc in locs}
    loc_df = df[df.Location.isin(locs)]
    loc_ys = np.array([loc_df[loc_df.Location == loc].LossCost2.mean() for loc in locs])
    loc_M = np.log10(np.array([[loc_amenities[loc][atype] for atype in top_amenities]
                               for loc in locs]) + 1)
    loc_X = pca.transform(loc_M)
    
    plt.scatter(loc_X[:,0], loc_X[:,1], c=loc_ys, cmap='jet')
    for i, amen in enumerate(top_amenities[:5]):
        x, y = pca.components_[:2,i]
        zoom = 10
        x_offset = 0.25
        plt.arrow(0, 0, x*zoom, y*zoom, color='r', head_width=0.10)
        plt.annotate(amen, (x*zoom + x_offset, y*zoom), fontsize='large')
    plt.xlabel("First Principal Component", fontsize='large')
    plt.ylabel("Second Principal Component", fontsize='large')
    plt.title("MCD Locations in Reduced Amenity Space", fontsize='large')
    plt.xlim(-2, 10)
    plt.ylim(-3.5, 10)
    plt.colorbar(label='LossCost2')
    plt.savefig("reduced_amenity_space_w_locs.png", dpi=300)
    plt.close()
    
    knn = KNN(n_neighbors=50)
    knn.fit(loc_X[:,:2], loc_ys)
    x = loc_X[:,0]
    y = loc_X[:,1]
    z = knn.predict(loc_X[:,:2])
    n = 30j
    extent = (min(x), max(x), min(y), max(y))
    xs,ys = np.mgrid[extent[0]:extent[1]:n, extent[2]:extent[3]:n]
    
    resampled = griddata(x, y, z, xs, ys, interp='linear')

    plt.imshow(resampled.T, extent=extent)
    cbar=plt.colorbar(label='LossCost2')
    for i, amen in enumerate(top_amenities[:5]):
        x, y = pca.components_[:2,i]
        zoom = 10
        x_offset = 0.25
        plt.arrow(0, 0, x*zoom, y*zoom, color='r', head_width=0.10)
        plt.annotate(amen, (x*zoom + x_offset, y*zoom), fontsize='large')
    plt.xlabel("First Principal Component", fontsize='large')
    plt.ylabel("Second Principal Component", fontsize='large')
    plt.title("K-Nearest Neighbors Regression", fontsize='large')
    plt.xlim(-2, 10)
    plt.ylim(-3.5, 10)
    plt.savefig("knn.png", dpi=300)
    plt.close()
    
from matplotlib.mlab import griddata

def extract_isocontour(x, y, z):
    N = 30j
    extent = (min(x), max(x), min(y), max(y))

    xs,ys = np.mgrid[extent[0]:extent[1]:N, extent[2]:extent[3]:N]

    resampled = griddata(x, y, z, xs, ys, interp='linear')

    plt.imshow(resampled.T, extent=extent)
    plt.xlim([x.min(),x.max()])
    plt.ylim([y.min(),y.max()])
    cbar=plt.colorbar()
    plt.show()
