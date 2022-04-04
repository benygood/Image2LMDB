# Image2LMDB
Convert image folder to lmdb, adapted from https://github.com/Lyken17/Efficient-PyTorch
```
.
├── folder2lmdb.py
├── img
│   ├── train
│   │   ├── bar_dir
│   │   │   ├── 100000.jpg
│   │   │   ├── 100001.jpg
│   │   │   ├── 100002.jpg
│   │   │   ├── 100003.jpg
│   │   │   ├── 100004.jpg
│   │   │   ├── 100005.jpg
│   │   │   ├── 100006.jpg
│   │   │   ├── 100007.jpg
│   │   │   ├── 100008.jpg
│   │   │   └── 100009.jpg
│   │   └── foo_dir
│   │       ├── 100000.jpg
│   │       ├── 100001.jpg
│   │       ├── 100002.jpg
│   │       ├── 100003.jpg
│   │       ├── 100004.jpg
│   │       ├── 100005.jpg
│   │       ├── 100006.jpg
│   │       ├── 100007.jpg
│   │       ├── 100008.jpg
│   │       └── 100009.jpg
│   
│   
├── main.py
├── README.md
└── requirements.txt
```

## Convert image folder to lmdb
```python
python folder2lmdb.py img
```
## how to write Label in lmdb
https://zhuanlan.zhihu.com/p/388983712
```python
class LMDB_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary 
        # for this dataset, but some datasets may include images of 
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label
 
    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)
        
def data2lmdb(dpath, name="train", write_frequency=5000, num_workers=8):
    # 获取自定义的COCO数据集（就是最原始的那个直接从磁盘读取image的数据集）
    dataset=COCO2014(root="/data/jxzhang/coco/",phase=name)
    data_loader = DataLoader(dataset, num_workers=8, collate_fn=lambda x: x)

    lmdb_path = osp.join(dpath,"%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label, _ = data[0]
        temp = LMDB_Image(image,label)
        txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps(temp))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
    
class DatasetLMDB(data.Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length =pickle.loads(txn.get(b'__len__'))
            self.keys= pickle.loads(txn.get(b'__keys__'))
            # self.NP= pickle.loads(txn.get(b'__NP__'))
        # self.class_weights=torch.load("/data/jxzhang/coco/data/classweight.pt")
        self.transform = transform
        self.num_classes=80

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin() as txn:
            byteflow = txn.get(self.keys[index])
        IMAGE= pickle.loads(byteflow)
        img, label = IMAGE.get_image(),IMAGE.label
        return Image.fromarray(img).convert('RGB'),label.copy()

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

```

## Multi lmdb
https://zhuanlan.zhihu.com/p/374875094 \
场景：有两个lmdb数据集，而且很大，进行合并比较耗时，此时可以分别加载然后训练时叠加使用。
```python

#train_data1,train_data2为 lmdb路径
#eys_path_train1，keys_path_train2 为两个数据的键的npy文件路径

training_data1 = LmdbDataset_train(train_data1,transform,keys_path_train1)
training_data2 = LmdbDataset_train(train_data2,transform,keys_path_train2)
​
​
training_data = training_data1 + training_data2
train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=opt.batch_size,
                                               shuffle=False,
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler)

```

```python
class LmdbDataset_train(Dataset):
    def __init__(self,lmdb_path,optimizer,keys_path):
        # super().__init__()
        self.optimizer = optimizer
​
        self.datum=caffe_pb2.Datum()
        self.lmdb_path = lmdb_path
        keys = np.load(keys_path)
        self.keys = keys.tolist()
        self.length = len(self.keys)
    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(buffers=True,write=False)
   
    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
​
        serialized_str = self.txn.get(self.keys[index])
        self.datum.ParseFromString(serialized_str)
        size=self.datum.width*self.datum.height
        
        pixles1=self.datum.data[0:size]
        pixles2=self.datum.data[size:2*size]
        pixles3=self.datum.data[2*size:3*size]
​
        image1=Image.frombytes('L', (self.datum.width, self.datum.height), pixles1)
        image2=Image.frombytes('L', (self.datum.width, self.datum.height), pixles2)
        image3=Image.frombytes('L', (self.datum.width, self.datum.height), pixles3)
​
        img=Image.merge("RGB",(image3,image2,image1))
​
        img =self.optimizer(img)
​
        label=self.datum.label
        return img, label
​
    def __len__(self):
        return self.length
```
## Test it
```python
python main.py img/train.lmdb
```


```
key 0
key 1
torch.Size([2, 224, 224, 3])
key 2
key 3
torch.Size([2, 224, 224, 3])
key 4
key 5
torch.Size([2, 224, 224, 3])
key 6
key 7
torch.Size([2, 224, 224, 3])
key 8
key 9
torch.Size([2, 224, 224, 3])
key 10
key 11
torch.Size([2, 224, 224, 3])
key 12
key 13
torch.Size([2, 224, 224, 3])
key 14
key 15
torch.Size([2, 224, 224, 3])
key 16
key 17
torch.Size([2, 224, 224, 3])
key 18
key 19
torch.Size([2, 224, 224, 3])
```



## Original Repo:
https://github.com/Lyken17/Efficient-PyTorch
