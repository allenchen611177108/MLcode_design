import os

start = "C:/allen_env\deeplearning\metaDataset"
for dirlv1 in (os.listdir(start)):
    node1 = os.path.join(start, dirlv1)
    for dirlv2 in (os.listdir(node1)):
        node2 = os.path.join(node1, dirlv2)
        for dirlv3 in (os.listdir(node2)):
            if not os.path.exists(os.path.join(node2, 'test')):
                os.mkdir(os.path.join(node2, 'test'))
                os.mkdir(os.path.join(node2, 'test', 'pic'))
                os.mkdir(os.path.join(node2, 'test', 'mask'))
            if not os.path.exists(os.path.join(node2, 'train')):
                os.mkdir(os.path.join(node2, 'train'))
                os.mkdir(os.path.join(node2, 'train', 'pic'))
                os.mkdir(os.path.join(node2, 'train', 'mask'))