import pickle
import torchvision
import numpy as np
import PIL.Image as im

# # download CIFAR10 dataset
# torchvision.datasets.CIFAR10("./cifar10/",train=True, download=True)

def get_images(INDEX:int, dims:tuple = (3,32,32)):
    """Gets images from dataset by index.
    returns 1D array image data.
    - INDEX: i-th image from dataset
    - dims: image dimensions (k,i,j), default=(3,32,32)
    """
    # unpickle batch data
    BATCH1 = "./cifar10/cifar-10-batches-py/data_batch_1"
    with open(BATCH1, 'rb') as fo:
        # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
        dict = pickle.load(fo, encoding='bytes')
        fo.close()

    # # unpickle label_names
    # with open("./cifar10/cifar-10-batches-py/batches.meta","rb") as f:
    #     meta = pickle.load(f,encoding="bytes")
    #     print(meta)
    #     f.close()

    label_names = ['airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    LABEL:int = dict[b"labels"][INDEX] #6
    FNAME = str(dict[b"filenames"][INDEX],encoding="utf8") 
    # add label in front of fname
    FNAME = "./cifar10/__" + label_names[LABEL].upper()\
        + "_" + FNAME 
    IMGDATA = dict[b"data"][INDEX] #3072=32*32*3
    print(label_names[LABEL], FNAME,IMGDATA, sep="\t")

    imgdata = _img_reshape(IMGDATA,'F',dims)#CFAK
    return (FNAME,imgdata)

def _img_reshape(imgdata,order="F",shape=(32,32,3)):
    """make 2D matrix for img preview
    - imgdata: image data
    - order="F": order character for reshaping np.array (CFAK)
    - shape=(32,32,3): destination shape
    """
    arr = np.array(imgdata).reshape(*shape,order=order)
    # print(arr, arr.size)
    return arr

def _imshow(imgdata,fname=None):
    """show image.
    - imgdata: array
    - fname=None: file name. if None, only show image
    """
    # https://www.geeksforgeeks.org/convert-a-numpy-array-to-an-image/
    # https://www.codegrepper.com/code-examples/python/convert+3d+array+to+image+python
    imgdata = _img_reshape(imgdata,'F')#CFAK
    image = im.fromarray(imgdata,"RGB")
    if fname != None:
        image.save(fname,"PNG")
        return image
    else:
        image.show()

if __name__ == "__main__":
############
    dims = (3,32,32)
############
    dict = {}
    for i in range(10,20): #set index range
        fname, data = get_images(i)
        print(data.shape)
        dict[fname] = data.tolist()
    with open("./cifar10/sample_bin_data.json","w") as f:
        import json
        json.dump(dict,f,ensure_ascii=True)
        f.close()