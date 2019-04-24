import numpy as np
"""Code for packing and unpacking a numpy array into a byte array.
   the array is flattened if it is not 1D.
   This is intended to be used as the interface for storing 
   
   This code is intended to be used to store numpy array as fields in a dataframe and then store the 
   dataframes in a parquet file.
"""

def packArray(a):
    if type(a)!=np.ndarray:
        raise Exception("input to packArray should be numpy.ndarray. It is instead "+str(type(a)))
    return bytearray(a.tobytes())
def unpackArray(x,data_type=np.int16):
    return np.frombuffer(x,dtype=data_type)