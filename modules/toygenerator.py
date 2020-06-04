

import skimage
from skimage import color
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import math
from numba import jit
import random
#generate one shape at a time and then add the numpy arrays only where there are zeros

npixel=128


## debug

#add wiggles to an individual object
@jit(nopython=True)      
def addWiggles(image, background=False):
    
    xwiggle=random.random()+0.1#float 0, 1)
    ywiggle=random.random()+0.1
    colidx = random.randint(0,3)
    
    strength=10
    offset=0
    if background:
        offset=-40
        strength=20
        xwiggle *= .1
        ywiggle *= .1
    
    def adaptColour(col,xidx,yidx):
        nomc=col[xidx,yidx,colidx]
        if nomc>254 and not background: return
        if background and nomc<255: return
        
        if background:
            col[xidx,yidx,colidx] = 255+  int(offset + strength *( math.sin(float(xidx)*xwiggle) 
                                                     + math.sin(float(yidx)*ywiggle))  )
            if col[xidx,yidx,colidx]>255:
                col[xidx,yidx,colidx]=255
        else:
            col[xidx,yidx,colidx] = nomc+  int(offset + strength *( math.sin(float(xidx)*xwiggle) 
                                                     + math.sin(float(yidx)*ywiggle))  )
        if col[xidx,yidx,colidx]>254 and not background:
            col[xidx,yidx,colidx]=253
        if col[xidx,yidx,colidx]>255 and background:
            col[xidx,yidx,colidx]=255
        elif col[xidx,yidx,colidx] <0:
            col[xidx,yidx,colidx]=0
           
    for x in range(len(image)):
        for y in range(len(image[x])):
            adaptColour(image,x,y)
            
    return image


def makeRectangle(size, pos, image):
    print('pos', pos)
    posa = [pos[0]-size[0]/2., pos[1]-size[1]/2.]
    print('pos',posa)
    print('size',size[0],size[1])
    print(image[int(pos[1])][int(pos[0])])
    print(image[0][0])
    return patches.Rectangle(posa,size[0],size[1],linewidth=1,edgecolor='y',facecolor='none')

##


def getCenterCoords(t1):
    #yes, that is the format - no, there is a bug in here!!!
    coords1b = float(t1[0][1][0][1]+t1[0][1][0][0])/2.
    coords1a = float(t1[0][1][1][1]+t1[0][1][1][0])/2.
    return coords1a, coords1b

def getWidthAndHeight(t1):
    return  float(t1[0][1][1][1]-t1[0][1][1][0]), float(t1[0][1][0][1]-t1[0][1][0][0])

def labeltoonehot(desc):
    if desc[0][0] == 'rectangle':
        return np.array([1,0,0],dtype='float32')
    if desc[0][0] == 'triangle':
        return np.array([0,1,0],dtype='float32')
    if desc[0][0] == 'circle':
        return np.array([0,0,1],dtype='float32')
    
    raise Exception("labeltoonehot: "+desc[0][0])
   
def createPixelTruth(desc,image, ptruth, objid):
    onehot = labeltoonehot(desc)
    coordsa,coordsb = getCenterCoords(desc)
    coords = np.array([coordsa,coordsb], dtype='float32')
    truth =  np.concatenate([coords,onehot])
    truth = np.expand_dims(np.expand_dims(truth, axis=0),axis=0)
    truth = np.tile(truth, [image.shape[0],image.shape[1],1])
    w,h = getWidthAndHeight(desc)
    whid = np.array([w,h,objid], dtype='float32')
    whid = np.expand_dims(np.expand_dims(whid, axis=0),axis=0)
    whid = np.tile(whid, [image.shape[0],image.shape[1],1])
    truth =  np.concatenate([truth,whid],axis=-1)
    if ptruth is None:
        ptruth = np.zeros_like(truth)+255
        ptruth[:,:,-1] = -1
    truthinfo = np.where(np.tile(np.expand_dims(image[:,:,0]<255, axis=2), [1,1,truth.shape[2]]), truth, ptruth)
    return truthinfo
    

def createMask(ptruth):
    justone = ptruth[:,:,0:1]
    return np.where(justone > 254, np.zeros_like(justone), np.zeros_like(justone)+1.)
    
def addNObjects(ptruth,nobj):
    a = np.array([[[nobj]]],dtype='float32')
    a = np.tile(a, [ptruth.shape[0],ptruth.shape[1],1])
    return np.concatenate([ptruth,a],axis=-1)
    
def checktuple_overlap(t1,t2):
    x1,y1 = getCenterCoords(t1)
    x2,y2 = getCenterCoords(t2)
    diff = (x1-x2)**2 + (y1-y2)**2
    if diff < 30:
        return True
    return False
    
    
def checkobj_overlap(dscs,dsc):
    for d in dscs:
        if checktuple_overlap(dsc,d):
            return True
    return False
    
#fig,ax = plt.subplots(1)
def generate_shape(npixel, seed=None):
    if seed is not None:
        return skimage.draw.random_shapes((npixel, npixel),  max_shapes=1, 
                                      min_size=npixel/3, max_size=npixel/2,
                                      intensity_range=((100, 220),))
    else:
        return skimage.draw.random_shapes((npixel, npixel),  max_shapes=1, 
                                      min_size=npixel/3, max_size=npixel/2, # 5, 3
                                      intensity_range=((100, 220),), random_seed=seed)
    

#adds a shape BEHIND the existing ones
def addshape(image , desclist, npixel, seed=None, addwiggles=False):
    if image is None:
        image, d = generate_shape(npixel)
        if addwiggles:
            image = addWiggles(image)
        return image, image, d, True
    
    new_obj_image, desc = generate_shape(npixel,seed)
    
    if addwiggles:
        new_obj_image=addWiggles(new_obj_image)
    
    #if checkobj_overlap(desclist,desc):
    #    return image, image, desc, False
    
    shape = new_obj_image.shape
    notempty = image < 255 #only where the image was not empty before
    empty = image > 254
    
    newpixels = new_obj_image < 255
    pixelstoadd = np.logical_and(newpixels, empty)
    n_objpixels = np.count_nonzero(newpixels[:,:,0])
    n_newpixels = np.count_nonzero(pixelstoadd[:,:,0])
    visible_fraction = float(n_newpixels)/float(n_objpixels)
    
    #print('n_newpixels/n_objpixels',visible_fraction)
    visible_mask = np.where(pixelstoadd,np.zeros_like(image)+1,np.zeros_like(image))
    newobjectadded = np.where(notempty,image,new_obj_image)
    new_obj_image = np.where(visible_mask>0, new_obj_image, np.zeros_like(image)+255)
    
    if np.all(newobjectadded == image) or visible_fraction<0.1: 
        return image, image, desc, False
    
    return newobjectadded, new_obj_image, desc, True


def create_images(nimages = 1000, npixel=64, seed=None, addwiggles=False):

    '''
    returns features, truth
    
    returns features as
    B x P x P x F
    with F = colours
    
    returns truth as 
    B x P x P x T
    with T = [mask, true_posx, true_posy, ID_0, ID_1, ID_2, true_width, true_height, n_objects]
    '''
    debug = False
    doubledebug = False
    
    pixrange = np.arange(npixel, dtype='float32')
    pix_x = np.tile(np.expand_dims(pixrange,axis=0), [npixel,1])
    pix_x = np.expand_dims(pix_x,axis=2)
    pix_y = np.tile(np.expand_dims(pixrange,axis=1), [1,npixel])
    pix_y = np.expand_dims(pix_y,axis=2)
    pix_coords = np.concatenate([pix_y,pix_x],axis=-1)
    
    
    alltruth = []
    allimages = []
    for e in range(nimages):
        dsc=[]
        image=None
        nobjects = random.randint(1,9)
        addswitch=True
        indivimgs=[]
        indivdesc=[]
        ptruth=None
        i=0
        itcounter=0
        while i < nobjects:
            #print('e,i',e,i)
            itcounter+=1
            new_image, obj, des, addswitch = addshape(image,indivdesc, npixel=npixel, seed=seed, addwiggles=addwiggles)
            if addswitch:
                ptruth = createPixelTruth(des, obj, ptruth, i)
                image = new_image
                if doubledebug:
                    fig,ax =  plt.subplots(1,1)
                    ax.imshow(image)
                    w,h = getWidthAndHeight(des)
                    x,y = getCenterCoords(des)
                    rec=makeRectangle([w,h], [x,y], image)
                    ax.add_patch(rec)
                    fig.savefig("image"+str(e)+"_"+str(i)+".png")
                    ax.imshow(ptruth[:,:,0]+10*ptruth[:,:,1])
                    fig.savefig("image"+str(e)+"_"+str(i)+"_t.png")
                    plt.close()
                i+=1
            else:
                pass #print("skip "+str(e)," ", str(i))
            if itcounter>100: #safety against endless loops for weird object combinations
                break
                
        if addwiggles:
            image = addWiggles(image, background=True)
        #print(image)
        if e < 100 and debug:
            plt.imshow(image)
            #plt.show()
            plt.savefig("image"+str(e)+".png")
            #exit()
            #
        mask = createMask(ptruth)
        ptruth = np.concatenate([mask,ptruth],axis=-1)
        #ptruth = addNObjects(ptruth,i)
        
        
        
        #ax.imshow(image)
        image = np.concatenate([image,pix_coords],axis=-1)
        image = np.expand_dims(image,axis=0)
        ptruth = np.expand_dims(ptruth,axis=0)
        
        if seed is not None:
            seed+=1
        
        #for x in range(ptruth.shape[1]):
        #    for y in range(ptruth.shape[2]):
        #        print(ptruth[0][x][y])
        
        allimages.append(image)
        alltruth.append(ptruth)
    
    return np.concatenate(allimages, axis=0), np.concatenate(alltruth,axis=0)
