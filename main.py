import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import pandas as pd
from pandas.plotting import parallel_coordinates


def read_matrix(filename):
    """
    Read a .matrix file.
    Parameters
    ----------
    filename : string
        The path of the .matrix file
    Returns
    -------
    Numpy array
        The file as a Numpy array
    """
    matrix_file = open(filename, "r")
    lines = matrix_file.read().strip().split("\n")
    matrix_file.close()
    lines = list(' -'.join(line.split('-')).split(' ') for line in lines)
    lines = list(list(int(i) for i in line if i) for line in lines)
    return np.array(lines)

def clean(matrix,missing_value = -1):
    """
    Replace the missing value with random one's.
    Parameters
    ----------
    matrix : Numpy array
        Values matrix
    missing_values : float
        Value to be considered as missing (default -1)
    Returns
    -------
    Numpy array
        missing
    """
    temp_matrix = np.copy(matrix)
    generator = np.random.RandomState(0)
    idx = np.where(temp_matrix == missing_value)
    temp_matrix[idx] = generator.randint(0, 801, len(idx[0]))
    return temp_matrix

def mean_squared_residue_np(matrix,rows, cols, inverted_rows = np.array([])):
    """
    Compute the MSR(Mean Squared Residue) of the submatrix defined by rows,cols and inverted_rows over the matrix.
    Parameters
    ----------
    matrix : Numpy array
        Values matrix
    rows : Numpy array
        Array of rows indexes of submatrix
    cols : Numpy array
        Array of columns indexes of submatrix
    inverted_rows : Numpy array (default np.array([]))
        Array of inverted rows indexesof submatrix
    Returns
    -------
    float 
        The MSR of the submatrix
    """
    matrix2 = matrix[rows][:,cols]
    if inverted_rows.size > 0:
        matrix_inverted = np.flip(matrix[inverted_rows][:,cols],1)
        matrix2 = np.append(matrix2,matrix_inverted,0)

    msr = lambda a: (np.power(a - a.mean(axis=1, keepdims=True) -
                          a.mean(axis=0) + a.mean(), 2).mean())
    return msr(matrix2)

def multiple_deletion_node_np(matrix, msr_threshold=300, alpha=1.2):
    """
    Multiple deletion node on matrix.
    Parameters
    ----------
    matrix : Numpy array
        Values matrix
    msr_threshold : float (default 300)
        Minimum MSR of submatrix to be considered acceptable
    alpha : float (default 1.2)
        Value of alpha 
    Returns
    -------
    Numpy array, Numpy array 
        The rows and columns indexes of submatrix.
    """
    rows = np.arange(0,matrix.shape[0])
    cols = np.arange(0,matrix.shape[1])
    msr = mean_squared_residue_np(matrix,rows,cols)
    print "MSR before multiple_deletion_node\t" + str(msr)
    rows_mean = matrix[rows].mean(axis=1, keepdims=True)
    cols_mean = matrix[rows][:,cols].mean(axis=0)
    deletion = True
    while deletion and msr > msr_threshold:
        
        arr = matrix[rows][:,cols] - rows_mean - cols_mean
        arr += np.mean(matrix[rows][:,cols])
        msr_rows = np.power(arr,2).mean(axis=1)
        rows_to_remove = msr_rows <= (alpha * msr)
        rows = rows[rows_to_remove]
        msr = mean_squared_residue_np(matrix,rows,cols)
        rows_mean = matrix[rows].mean(axis=1, keepdims=True)
        cols_mean = matrix[rows][:,cols].mean(axis=0)
        
        cols_to_remove = np.array([])
        if matrix.shape[1] > 100:
            arr = matrix[rows][:,cols] - rows_mean - cols_mean
            arr += np.mean(matrix[rows][:,cols])
            msr_cols = np.power(arr,2).mean(axis=0)
            cols_to_remove = msr_cols <= (alpha * msr)
            cols = cols[cols_to_remove]
            msr = mean_squared_residue_np(matrix,rows,cols)
            rows_mean = matrix[rows].mean(axis=1, keepdims=True)
            cols_mean = matrix[rows][:,cols].mean(axis=0)
        elements_removed = np.count_nonzero(rows_to_remove == False) + np.count_nonzero(cols_to_remove == False)
        
        if(elements_removed == 0):
            deletion = False
        
    print "MSR after multiple_deletion_node\t" + str(msr)
    return rows,cols

def single_deletion_node_np(matrix,rows,cols,msr_threshold=300):
    """
    Single deletion node on submatrix defined by rows and cols.
    Parameters
    ----------
    matrix : Numpy array
        Values matrix
    rows : Numpy array
        Array of rows indexes of submatrix
    cols : Numpy array
        Array of columns indexes of submatrix    
    msr_threshold : float (default 300)
        Minimum MSR of submatrix to be considered acceptable
    Returns
    -------
    Numpy array, Numpy array 
        The rows and columns indexes of submatrix.
    """
    msr = mean_squared_residue_np(matrix,rows,cols)
    print "MSR before single_deletion_node\t\t" + str(msr)
    rows_mean = matrix[rows].mean(axis=1, keepdims=True)
    cols_mean = matrix[rows][:,cols].mean(axis=0)
    deletion = True
    while msr > msr_threshold:
        
        arr = matrix[rows][:,cols] - rows_mean - cols_mean
        arr += np.mean(matrix[rows][:,cols])
        msr_rows = np.power(arr,2).mean(axis=1)
        msr_cols = np.power(arr,2).mean(axis=0)
        rows_max = np.amax(msr_rows)
        cols_max = np.amax(msr_cols)
        if rows_max > cols_max:
            rows = np.delete(rows,np.argmax(msr_rows))
        else:
            cols = np.delete(cols,np.argmax(msr_cols))
        msr = mean_squared_residue_np(matrix,rows,cols)
        rows_mean = matrix[rows].mean(axis=1, keepdims=True)
        cols_mean = matrix[rows][:,cols].mean(axis=0)
        
    print "MSR after single_deletion_node\t\t" + str(msr)
    return rows,cols

def node_addition_np(matrix,rows,cols):
    """
    Node addition on submatrix defined by rows and cols.
    Parameters
    ----------
    matrix : Numpy array
        Values matrix
    rows : Numpy array
        Array of rows indexes of submatrix
    cols : Numpy array
        Array of columns indexes of submatrix    
    Returns
    -------
    Numpy array, Numpy array, Numpy array
        The rows, columns and inverted rows indexes of submatrix.
    """
    inverted_rows = np.array([])
    matrix_rows = np.arange(0,matrix.shape[0])
    matrix_cols = np.arange(0,matrix.shape[1])
    msr = mean_squared_residue_np(matrix,rows,cols)
    print "MSR before node_addition\t\t" + str(msr)
    rows_mean = matrix[rows].mean(axis=1, keepdims=True)
    cols_mean = matrix[rows][:,cols].mean(axis=0)
    rows_not = np.setdiff1d(matrix_rows,rows)
    cols_not = np.setdiff1d(matrix_cols,cols)
    rows_mean_not = matrix[rows_not].mean(axis=1, keepdims=True)
    cols_mean_not = matrix[rows][:,cols_not].mean(axis=0)
    addition = True
    while addition:
        
        arr = matrix[rows_not][:,cols] - rows_mean_not - cols_mean
        arr += np.mean(matrix[rows][:,cols]) #dubbio
        msr_rows = np.power(arr,2).mean(axis=1)
        rows_to_append = msr_rows < msr
        rows = np.append(rows,rows_not[rows_to_append])
        rows_not = np.setdiff1d(rows_not,rows_not[rows_to_append])
        msr = mean_squared_residue_np(matrix,rows,cols,inverted_rows)
        rows_mean = matrix[rows].mean(axis=1, keepdims=True)
        cols_mean = matrix[rows][:,cols].mean(axis=0)
        rows_mean_not = matrix[rows_not].mean(axis=1, keepdims=True)
        cols_mean_not = matrix[rows][:,cols_not].mean(axis=0)

        arr = matrix[rows][:,cols_not] - rows_mean - cols_mean_not
        arr += np.mean(matrix[rows][:,cols]) #dubbio
        msr_cols = np.power(arr,2).mean(axis=0)
        cols_to_append = msr_cols < msr
        cols = np.append(cols,cols_not[cols_to_append])
        cols_not = np.setdiff1d(cols_not,cols_not[cols_to_append])
        msr = mean_squared_residue_np(matrix,rows,cols,inverted_rows)
        rows_mean = matrix[rows].mean(axis=1, keepdims=True)
        cols_mean = matrix[rows][:,cols].mean(axis=0)
        
        arr = -matrix[rows_not][:,cols] + rows_mean_not - cols_mean
        arr += np.mean(matrix[rows][:,cols]) #dubbio
        msr_rows = np.power(arr,2).mean(axis=1)
        rows_to_append = msr_rows < msr
        if(inverted_rows.size == 0):
            inverted_rows = rows_not[rows_to_append]
        else:
            inverted_rows = np.append(inverted_rows,rows_not[rows_to_append])
        rows_not = np.setdiff1d(rows_not,rows_not[rows_to_append])
        msr = mean_squared_residue_np(matrix,rows,cols,inverted_rows)
        rows_mean = matrix[rows].mean(axis=1, keepdims=True)
        cols_mean = matrix[rows][:,cols].mean(axis=0)
        rows_mean_not = matrix[rows_not].mean(axis=1, keepdims=True)
        cols_mean_not = matrix[rows][:,cols_not].mean(axis=0)
        
        elements_removed = np.count_nonzero(rows_to_append == True) + np.count_nonzero(cols_to_append == True)
    
        if(elements_removed == 0):
            addition = False
        
    print "MSR after node_addition\t\t\t" + str(msr)
    return rows,cols,inverted_rows

def hide_bicluster_np(matrix,rows,cols,inverted_rows = np.array([])):
    """
    Mask the submatrix defined by rows, cols and inverted_rows on matrix with random values.
    Parameters
    ----------
    matrix : Numpy array
        Values matrix
    rows : Numpy array
        Array of rows indexes of submatrix
    cols : Numpy array
        Array of columns indexes of submatrix    
    inverted_rows : Numpy array (default np.array([]))
        Array of inverted rows indexesof submatrix
    Returns
    -------
    Numpy array
        A copy of matrix in which submatrix has been masked.
    """
    matrix2 = np.copy(matrix)
    generator = np.random.RandomState(0)
    for row in rows:
        matrix2[row,cols] = generator.randint(0,801,cols.size)
        #print matrix2[row,cols]
    if inverted_rows.size > 0:
        for row in inverted_rows:
            matrix2[row,cols] = generator.randint(0,801,cols.size)
    print "Last bicluster masked"
    return matrix2

def get_bicluster(matrix,rows,cols,inv = np.array([])):
    """
    Get a submatrix given rows,columns and inveted rows indexes.
    Parameters
    ----------
    matrix : Numpy array
        Values matrix
    rows : Numpy array
        Array of rows indexes of submatrix
    cols : Numpy array
        Array of columns indexes of submatrix    
    inverted_rows : Numpy array (default np.array([]))
        Array of inverted rows indexes of submatrix
    Returns
    -------
    Numpy array
        Submatrix.
    """
    rows = np.append(rows,inv)
    rows.sort()
    cols.sort()
    return matrix[rows][:,cols]

def plot_bicluster(bicluster1, bicluster_name="Bicluster"):
    """
    Plot a bicluster.
    Parameters
    ----------
    bicluster1 : Pandas DataFrame
        Bicluster to plot
    bicluster_name : string (default "Bicluster")
        Array of rows indexes of submatrix
    Returns
    -------
    None
    """
    bicluster = bicluster1.copy()
    bicluster["index"] = bicluster.index.values
    parallel_coordinates(bicluster, "index", linewidth=1.0)
    plt.title(bicluster_name)
    plt.xlabel('column')
    plt.ylabel('expression level')
    plt.gca().legend_ = None
    plt.show()

def find_biclusters_np(matrix, n_of_bicluster=10, msr_threshold=300, alpha=1.2):
    """
    Find biclusters in a given matrix.
    Parameters
    ----------
    matrix : Numpy array
        Values matrix
    n_of_bicluster : int
        Number of desired biclusters
    msr_threshold : float (default 300)
        Minimum MSR of submatrix to be considered acceptable
    alpha : float (default 1.2)
        Value of alpha 
    Returns
    -------
    List of Pandas DataFrame
        The list of biclusters.
    """
    matrixA = np.copy(matrix)
    biclusters = []
    for i in range(n_of_bicluster):
        rowsB, colsB = multiple_deletion_node_np(matrixA, msr_threshold=msr_threshold, alpha=alpha)
        rowsC, colsC = single_deletion_node_np(matrixA, rowsB, colsB, msr_threshold=msr_threshold)
        rowsD,colsD,invD = node_addition_np(matrix, rowsC, colsC)
        print "Bicluster " + str(i) 
        biclusters.append(pd.DataFrame(get_bicluster(matrix,rowsD,colsD,invD)))
        matrixA = hide_bicluster_np(matrixA, rowsD, colsD, invD)
    return biclusters

def main():
    data = read_matrix("Datasets\yeast.matrix")
    data = clean(data)
    start = time.time()
    biclusters = find_biclusters_np(data, n_of_bicluster=10,msr_threshold=300)
    end = (time.time() - start)

    for i,bicluster  in enumerate(biclusters):
        plot_bicluster(bicluster,bicluster_name="Bicluster "+str(i))

    print end,"seconds"

if __name__ == "__main__":
    main()
