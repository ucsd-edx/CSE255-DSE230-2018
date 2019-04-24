from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.pyplot import *


def generate_centers(k=3,radius=1,d=2):
    """
    Generates k centers organized in a circle of radius radius
    """
    centers=np.zeros([k,d])
    i=0
    for angle in np.arange(0,2*np.pi-0.001,(2*np.pi)/k):
        center=np.array([np.cos(angle),np.sin(angle)])*radius
        centers[i,:2]=center
        i+=1
    return centers

def generate_clusters(centers,n=10):
    """
    Generates data frm spherical gaussians centered at the given centers
    The points in each cluster are generated according to a spherical gaussian with radius (variance) 1.
    n points are generated for each cluster.
    The dimension of the space is d. The centers are defined for the first 2 dimensions, the other coordnates are zero.
    """
    k,d=centers.shape

    L=[]
    labels=[]
    for i in range(k):
        
        A=np.random.normal(loc=centers[i], scale=[1.0]*d, size=(n,d))
        L.append(A)
        labels += [i]*n
    X=np.concatenate(L)
    df=pd.DataFrame(X)
    df['label']=labels
    return df

def create_data(k=5, n=500, d=2,radius=3):
    centers=generate_centers(radius=radius,k=k,d=d)
    df=generate_clusters(centers,n=n)
    return df,centers

def format_list(L,form='%3.3f'):
    return ','.join(form%l for l in L)

def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def compute_errors(km,df,centers,k=3):
    A=np.stack([km.labels_,np.array(df['label'])]).transpose()   # Zip true labels with computed labels
    # Count the number of copies for each pair
    counts=np.zeros([k,k])
    pairs,C=np.unique(A,axis=0,return_counts=True) # Count the number of times that each (true,computed) pair occured
    #print('pairs,C=',pairs,C)
    for i in range(len(C)):
        #print(i,pairs[i])
        counts[pairs[i][0],pairs[i][1]]+=C[i]

    examples_per_cluster=sum(counts[:,0])
    permutation_=np.argmax(counts,axis=1) # Find the large counts
    print('mapping of means to true clusters=',permutation_)
    print('classification accuracy=\t',counts[range(k),permutation_]/examples_per_cluster)

    if centers[permutation_,:].shape == km.cluster_centers_.shape:
        diff=centers[permutation_,:]-km.cluster_centers_
        print('error in ocation of centers=\t',format_list(np.sqrt(np.sum(diff**2,axis=1))))
    else:
        print('fewer centers than true centers')                                                               
    return permutation_

def RMSEvsK(n_centers,n,X,_title=""):
    solutions=[]
    _range=range(1,n_centers*2)
    for n_clusters in _range:
        kmeans = KMeans(n_clusters,n_init=10,init='k-means++',verbose=0).fit(X)
        solutions.append(kmeans)

    Inertia=[km.inertia_/n for km in solutions]

    figure(figsize=[10,3])
    subplot(1,2,1)
    plot(_range,Inertia);
    xlabel('# centers in model')
    ylabel('RMSE')
    title(_title+' full scale')
    ylim([0,max(Inertia)])
    grid()
    
    subplot(1,2,2)
    plot(_range,Inertia);
    xlabel('# centers in model')
    ylabel('RMSE')
    title(_title+' Zoomed Scale')
    grid()

def make_scatter(df,centers,centroids,_figsize=[8,8]):
    cmap = cm.get_cmap('Set1') # Colour map (there are many others)
    binary_cmap=cm.get_cmap('binary') # https://matplotlib.org/examples/color/colormaps_reference.html

    x = np.cos(np.linspace(0, np.pi, 20)).tolist()
    y = (0.2+np.sin(np.linspace(0, np.pi, 20))).tolist()
    xy1 = np.column_stack([x, y])

    x = np.cos(np.linspace(np.pi, 2*np.pi, 20)).tolist()
    y = (-0.2+np.sin(np.linspace(np.pi, 2*np.pi, 20))).tolist()
    xy2 = np.column_stack([x, y])

    fig, axes = subplots(nrows=1, ncols=1,figsize=_figsize);
    axes.scatter(centers[:, 0], centers[:, 1], c='g', s=100,label='True Center');
    axes.scatter(centroids[:, 0], centroids[:, 1],marker='v', c='y', s=100,label='centroid');
    df.plot.scatter(0,1,c='predicted_label',s=100,cmap=cmap,marker=(xy1, 0),\
                   figsize=_figsize,colorbar=False,ax=axes,label='cluster_id');
    df.plot.scatter(0,1,c='label_error',s=100,cmap=binary_cmap,marker=(xy2, 0),\
                   figsize=_figsize,colorbar=False,ax=axes,label='incorrect');
    axes.scatter(centers[:, 0], centers[:, 1], c='g', s=100);
    axes.scatter(centroids[:, 0], centroids[:, 1],marker='v', c='y', s=100);
    
def analyze(k,d,n,radius,plotRMSEbsK=True):
    """ 
    Generate k spherical gaussian clusters and analyze the performance of the Kmeans algorithm.
    The gaussian are placed at equal angular intervals on a circle of radius "radius" in dimensions 1 and 2.
        Parameters:
           k = number of generated clusters clusters
           d = dimension of embedding space
           n = number of examples per cluster
           radius: the distance of the clusters from the origin (compare to std=1 for each cluster in each dimension)
    """
    df,centers=create_data(k,d=d,n=n,radius=radius)
    X=np.array(df.iloc[:,:d])
    if plotRMSEbsK:
        RMSEvsK(k,n,X,_title="k=%d, d=%d, n=%d"%(k,d,n))
    km = KMeans(n_clusters=k,n_init=10,init='k-means++',verbose=0).fit(X)
    df['predicted_label']=km.labels_
    permutation_=compute_errors(km,df,centers,k=k)

    def permute(i):
        return permutation_[i]

    df['permuted']=permute(df['predicted_label'])
    df['label_error']=(df['permuted']!=df['label'])*1

    make_scatter(df,centers,km.cluster_centers_)
    return df,centers,km
