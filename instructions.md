# <span style="color: green"><b>Tải Anaconda3 và tạo môi trường ảo</b></span>

```bash
cd $HOME
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
zsh Anaconda3-2023.03-1-Linux-x86_64.sh
```

- Làm theo hướng dẫn của bộ cài đặt Anaconda3

```bash
export ENV=torch
conda init zsh
conda create -n $ENV python=3.9 -y
conda activate $ENV
```

### <b>Cài đặt thư viện sử dụng</b>

```bash
# Install pytorch with GPU supports
conda install -c conda-forge cudatoolkit=11.8.0 cudnn=8.8.0 -y # CUDA vs cuDNN
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 # pytorch
pip install -r requirements.txt # utilities
```

# <br /><span style="color: green"><b>Thu thập dữ liệu từ Internet</b></span>

- [Bộ dữ liệu công khai về biển số xe máy Việt Nam](https://bit.ly/2QzjLiC)
- [Bộ dữ liệu công khai về biển số xe máy Việt Nam 2](https://bit.ly/3hGqvqQ)
- [Bộ dữ liệu công khai về biển số xe ô tô Việt Nam](http://j.gs/GSB1)
- [Bộ dữ liệu công khai về biển số xe ô tô Việt Nam 2](http://j.gs/GSB2)
- [Bộ dữ liệu biển số xe trên đường phố Việt Nam](http://j.gs/GSB3)

# <br /><span style="color: green"><b>Gán nhãn thủ công với công cụ `labelImg` và phân chia dữ liệu</b></span>

### <b>Tải và cài đặt công cụ</b>

```bash
conda install pyqt=5
conda install -c anaconda lxml
git clone https://github.com/heartexlabs/labelImg.git
cd labelImg
pyrcc5 -o libs/resources.py resources.qrc
# modify the classes names in data/predefined_classes.txt [OPTIONAL]
# Change save format to YOLO [OPTIONAL]
python labelImg.py # starting the tool
```

### <b>Phím tắt công cụ</b>

| Shortcut | Work |
| --- | --- |
| Ctrl+u | Load all images in directory |
| Ctrl+r | Change the default annotation target dir |
| Ctrl+s | Save |
| Ctrl+d | Copy the current label and rect box |
| Ctrl+Shift+d | Delete the current image |
| w | Create a rect box |
| d | Next image |
| a | Previous image |
| del | Delete the selected rect box |
| Ctrl++ | Zoom in |
| Ctrl+- | Zoom out |

### <b>Yêu cầu khi gán nhãn</b>

- Bỏ qua những đối tượng rất nhỏ trong ảnh, YOLO không thể phát hiện được.
- Bỏ qua những đối tượng bị biến dạng xấu như: góc nghiêng quá lớn, bị quay ngang hoặc lộn ngược hay hình ảnh không thể nhận diện ký tự bằng mắt thường.
- Với mỗi hộp giới hạn của đối tượng, cần khoanh thừa ra một khoảng lề căn bằng nhau về 4 phía.
- Tâm của hộp giới hạn phải trùng với tâm của đối tượng, ví dụ: Với phát hiện biển số xe tâm của đối tượng trùng với tâm của hộp bao, còn với phát hiện bốn đỉnh của vật thì phải lấy tâm của hộp bao trùng với góc của biển.

### <b>Định dạng lưu khi gán nhãn phát hiện đối tượng</b>

Đối với bài toán phát hiện đối tượng, ta sử dụng định dạng YOLO để lưu.

### <b>Phân chia dữ liệu</b>

Ta thực hiện phân chia dữ liệu theo tỉ lệ: 70% dùng cho quá trình huấn luyện, 15% dùng cho quá trình xác thực và 15% còn lại dành cho phần đánh giá mô hình.

# <br /><span style="color: green"><b>Chuẩn bị mã nguồn YOLOv7 và dữ liệu để huấn luyện</b></span>

```bash
git clone https://github.com/wongkinyiu/yolov7.git
cd yolov7
export PRJ_ROOT=$(pwd)
# data preparation
cd $PRJ_ROOT/data
mkdir -p fpd/train/{images,labels} fpd/valid/{images,labels} fpd/test/{images,labels} \
    lpd/train/{images,labels} lpd/valid/{images,labels} lpd/test/{images,labels}
# move all training images of License Plate Detection (LPD) to `lpd/train/images`
# move all training labels of LPD to `lpd/train/labels`
# move all validation images of LPD to `lpd/valid/images`
# move all validation labels of LPD to `lpd/valid/labels`
# move all test images of LPD to `lpd/test/images`
# move all test labels of LPD to `lpd/test/labels`
# do the same thing to FPD (Four Points Detection)

# create dataset config file
touch fpd.yaml lpd.yaml
cat > lpd.yaml << EOF
>train: data/lpd/train
>val: data/lpd/valid
>test: data/lpd/test
>
># number of classes
>nc: 1
>
># class names
>names: ["license_plate"]
>EOF
cat > fpd.yaml << EOF
>train: data/fpd/train
>val: data/fpd/valid
>test: data/fpd/test
>
># number of classes
>nc: 4
>
># class names
>names: ["top_left", "top_right", "bottom_right", "bottom_left"]
>EOF

# setting model configuration file
cd $PRJ_ROOT/cfg/training
cp yolov7-w6.yaml yolov7-w6-lpd.yaml
cp yolov7-w6.yaml yolov7-w6-fpd.yaml
# change line 2 of `yolov7-w6-lpd.yaml` to 1 and `yolov7-w6-fpd.yaml` to 4

# download pretrained weight
cd $PRJ_ROOT
wget https://github.com/WongKinYiu/yolov7/releases/download/v0. 1/yolov7-w6_training.pt
# fine-tuning for LPD
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/lpd.yaml \
    --img 1280 1280 --cfg cfg/training/yolov7-w6-lpd.yaml --weights 'yolov7-w6_training.pt' \
    --name yolov7-w6-lpd --hyp data/hyp.scratch.custom.yaml
# fine-tuning for FPD
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/fpd.yaml \
    --img 1280 1280 --cfg cfg/training/yolov7-w6-fpd.yaml --weights 'yolov7-w6_training.pt' \
    --name yolov7-w6-fpd --hyp data/hyp.scratch.custom.yaml

# to test the model
python test.py --data data/lpd.yaml --img 1280 --batch 32 \
    --conf 0.001 --iou 0.65 --device 0 --weights runs/train/yolov7-w6-lpd/weights/best.pt \
    --name yolov7-w6-lpd-test
python test.py --data data/fpd.yaml --img 1280 --batch 32 \
    --conf 0.001 --iou 0.65 --device 0 --weights runs/train/yolov7-w6-fpd/weights/best.pt \
    --name yolov7-w6-fpd-test

# to export onnx
python export.py --weights runs/train/yolov7-w6-lpd/best.pt \
    --grid --end2end --simplify \
    --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 \
    --img-size 1280 1280 --max-wh 1280
python export.py --weights runs/train/yolov7-w6-fpd/best.pt \
    --grid --end2end --simplify \
    --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 \
    --img-size 1280 1280 --max-wh 1280
```

# <br><span style="color: green"><b>Chuẩn bị mã nguồn ViT và dữ liệu cho huấn luyện</b></span>

### <b>Chuẩn bị mã nguồn</b>

```bash
git clone https://github.com/roatienza/deep-text-recognition-benchmark.git vit
cd vit
export PRJ_ROOT=$(pwd)
```

### <b>Chuẩn bị dữ liệu</b>

```bash
# delete old create_lmdb_dataset.py
rm create_lmdb_dataset.py
touch create_lmdb_dataset.py
```

dán đoạn code sau vào tệp tin `create_lmdb_dataset.py` vừa tạo

```python
import os
import lmdb
import cv2
import numpy as np
import argparse
import shutil
import sys

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False
    
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k,v)

def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    # If lmdb file already exists, remove it. Or the new data will add to it.
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.makedirs(outputPath)
    else:
        os.makedirs(outputPath)

    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)

def read_data_from_folder(folder_path):
    image_path_list = []
    label_list = []
    pics = os.listdir(folder_path)
    pics.sort(key = lambda i: len(i))
    for pic in pics:
        image_path_list.append(folder_path + '/' + pic)
        label_list.append(pic.split('_')[0])
    return image_path_list, label_list

def read_data_from_file(file_path):
    image_path_list = []
    label_list = []
    f = open(file_path)
    while True:
        line1 = f.readline()
        line2 = f.readline() 
        if not line1 or not line2:
            break
        line1 = line1.replace('\r', '').replace('\n', '')
        line2 = line2.replace('\r', '').replace('\n', '')
        image_path_list.append(line1)
        label_list.append(line2)

    return image_path_list, label_list

def show_demo(demo_number, image_path_list, label_list):
    print ('\nShow some demo to prevent creating wrong lmdb data')
    print ('The first line is the path to image and the second line is the image label')
    for i in range(demo_number):
        print ('image: %s\nlabel: %s\n' % (image_path_list[i], label_list[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type = str, required = True, help = 'lmdb data output path')
    parser.add_argument('--folder', type = str, help = 'path to folder which contains the images')
    parser.add_argument('--file', type = str, help = 'path to file which contains the image path and label')
    args = parser.parse_args()
    
    if args.file is not None:
        image_path_list, label_list = read_data_from_file(args.file)
        createDataset(args.out, image_path_list, label_list)
        show_demo(2, image_path_list, label_list)
    elif args.folder is not None:
        image_path_list, label_list = read_data_from_folder(args.folder)
        createDataset(args.out, image_path_list, label_list)
        show_demo(2, image_path_list, label_list)
    else:
        print ('Please use --folder or --file to assign the input. Use -h to see more.')
        sys.exit()
```

Phân chia dữ liệu đã có thành 3 phần: 70% huấn luyện, 15% xác thực và 15% kiểm thử.<br>
Mỗi loại sẽ được lưu vào 1 file, ex: data/train.txt, data/valid.txt, data/test.txt.<br>
Mỗi dòng trong tệp tin `*.txt` sẽ bao gồm đường dẫn tuyệt đối đến ảnh, dòng tiếp theo đó sẽ là nhãn của ảnh. Ví dụ, dòng 1 là ảnh chứa ký tự `34998` thì dòng 2 sẽ là `34998`.<br>

<b>Lưu ý:</b><br>
- Mỗi dòng sẽ phải là đường dẫn tuyệt đối đến ảnh.
- Dòng đầu tiên không được để trống.
- Không có dòng nào trống giữa 2 dữ liệu ảnh.


Tạo bộ dữ liệu với thư viện `lmdb`

```bash
mkdir -p output
# train data
python create_lmdb_dataset.py --out output/train --file data/train.txt
python create_lmdb_dataset.py --out output/valid --file data/valid.txt
python create_lmdb_dataset.py --out output/test --file data/test.txt
```

### <b>Huấn luyện mô hình</b>

```bash
# Download pretrained weights
wget https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224_aug.pth
# fine-tuning
RANDOM=$$
python3 train.py --train_data output/train \
    --valid_data ouput/valid --select_data / \
    --batch_ratio 1.0 --Transformation None --FeatureExtraction None \
    --SequenceModeling None --Prediction None --Transformer \
    --TransformerModel=vitstr_tiny_patch16_224 --imgH 224 --imgW 224 \
    --manualSeed=$RANDOM  --sensitive \
    --batch_size=48 --isrand_aug --workers=-1 --scheduler
```

### <b>Đánh giá thực tiễn</b>

```bash
CUDA_VISIBLE_DEVICES=0 python3 test.py --eval_data output/test \
    --benchmark_all_eval --Transformation None --FeatureExtraction None \
    --SequenceModeling None --Prediction None --Transformer \
    --TransformerModel=vitstr_tiny_patch16_224 \
    --sensitive --data_filtering_off  --imgH 224 --imgW 224 \
    --saved_model "./saved_models/$RANDOM/best_accuracy.pth" --infer_model "./best.pt"
```

# <br><span style="color: green"><b>Triển khai hệ thống</b></span>

### <b>Chuẩn bị mã nguồn</b>

```bash
git clone https://github.com/bluedruddigon/alpr.git
cd alpr
export PRJ_ROOT=$(pwd)
python deploy.py
```

Cài đặt ngrok

```bash
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
sudo tar xzf ngrok-*.tgz -C /usr/local/bin
# add your authentication token
ngrok config add-authtoken $AUTH_TOKEN
ngrok http 5000
```

Truy cập trang web `https://dashboard.ngrok.com/cloud-edge/endpoints` và ấn vào đường dẫn công khai cho ứng dụng.
Thực hiện thử nghiệm với ảnh biển số xe!

<br>
<center>
Author: Bùi Hoàng Hải<br>
LICENSE: MIT<br>
PS: Happy Coding!
</center>