
import numpy as np

def makeKernelSelection(npixels=64,kernel_size=5):
    '''
    output is [x,y,k_size*k_size] (compatible with gather_nd (P x P x F) -> P*P x F)
    '''
    if not kernel_size % 2:
        raise ValueError("Kernel must be odd") #actually also even works, but odd is nicer
    
    half_kernel = int(kernel_size/2)
    diff = np.array([i for i in range(-half_kernel, half_kernel+1)],dtype=np.int64)
    print(diff)
    sel = []
    
    for i in range(npixels):
        i_kernel = diff + i
        i_kernel = np.where(i_kernel <0 , np.zeros_like(i_kernel), i_kernel)
        i_kernel = np.where(i_kernel >=npixels , np.zeros_like(i_kernel)+npixels-1, i_kernel)
        for j in range(npixels):
            j_kernel = diff + j
            j_kernel = np.where(j_kernel <0 , np.zeros_like(j_kernel), j_kernel)
            j_kernel = np.where(j_kernel >=npixels , np.zeros_like(j_kernel)+npixels-1, j_kernel)
            
            vertlist=[]
            for i_k in i_kernel:
                for j_k in j_kernel:
                    vertlist.append([i_k, j_k])
            sel.append(vertlist)
        
    return np.array(sel, dtype=np.int64)
    
    