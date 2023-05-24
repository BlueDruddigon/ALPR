import cv2
import numpy as np
import torch


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
    
    # compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, (dw, dh)


def preprocess(image_list):
    origin_RGB = []
    resized_data = []
    
    for img in image_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_RGB.append(img)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image).astype(np.float32)
        resized_data.append((image, ratio, dwdh))
    
    return origin_RGB, resized_data


def postprocess(box, ratio, dwdh):
    box -= np.array(dwdh * 2)
    box /= ratio
    box = box.round().astype(np.int32).tolist()
    box = [max(b, 0) for b in box]
    return box


def prepare_ocr(image):
    h, w = image.shape[:2]
    scale = h / w
    two_lines = []
    if scale >= 0.5:
        line1 = image[:h // 2]
        line2 = image[h // 2:]
        two_lines.extend([line1, line2])
    else:
        two_lines.append(image)
    return two_lines


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_points_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(widthA), int(widthB))
    
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_A), int(height_B))
    
    dst = np.array([
      [0, 0],
      [max_width - 1, 0],
      [max_width - 1, max_height - 1],
      [0, max_height - 1],
    ]).astype(np.float32)
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    return warped


def compare_classes(t1, t2):
    return (t1 == t2).all()


def img2text(model, image, converter):
    pred_strs = []
    with torch.no_grad():
        pred = model(image, seqlen=converter.batch_max_length)
        _, pred_index = pred.topk(1, dim=-1, largest=True, sorted=True)
        pred_index = pred_index.view(-1, converter.batch_max_length)
        length_for_pred = torch.IntTensor([converter.batch_max_length - 1])
        pred_str = converter.decode(pred_index[:, 1:], length_for_pred)
        pred_EOS = pred_str[0].find('[s]')
        pred_str = pred_str[0][:pred_EOS]
        
        pred_strs.append(pred_str)
    
    return pred_strs
