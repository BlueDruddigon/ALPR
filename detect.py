import random

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from utils import compare_classes, four_points_transform, img2text, postprocess, prepare_ocr, preprocess
from vitstr import TokenLabelConverter, ViTFeatureExtractor

PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# ONNX sessions
lpd_session = ort.InferenceSession('./weights/LPD/best.onnx', providers=PROVIDERS)
fpd_session = ort.InferenceSession('./weights/FPD/best.onnx', providers=PROVIDERS)

lpd_classes = ['license_plate']
fpd_classes = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
names = lpd_classes + fpd_classes
colors = { name: [random.randint(0, 255) for _ in range(3)] for _, name in enumerate(names) }
colors['license_plate'] = (0, 0, 255)

# OCR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
converter = TokenLabelConverter()
extractor = ViTFeatureExtractor()
model = torch.jit.load('./weights/OCR/best_ocr.pt').to(device)
model.eval()


def session_forward(sess, inputs):
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [i.name for i in sess.get_outputs()]
    inputs = np.ascontiguousarray(inputs)
    outputs = sess.run(output_names, {input_names[0]: inputs})
    return outputs


def ocr(net, images):
    result = []
    for image in images:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('L')
        
        img = extractor(image)
        img = img.to(device)
        ocr_info = img2text(net, img, converter)[0]
        result.append(ocr_info)
    return result


def pipeline(image):
    # prepare inputs
    origin_RGB, resized_data = preprocess([image])
    np_batch = np.concatenate([data[0] for data in resized_data])
    inputs = np.ascontiguousarray(np_batch / 255.)
    
    # forwarding
    outputs = session_forward(lpd_session, inputs)
    if np.size(outputs) == 0:
        return image
    
    # decode outputs
    plates = []
    original_box = dict()
    img = origin_RGB[0]
    for idx, (batch_idx, x0, y0, x1, y1, cls_idx, _) in enumerate(outputs[0]):
        ratio, dwdh = resized_data[int(batch_idx)][1:]
        box = postprocess(np.array([x0, y0, x1, y1]), ratio, dwdh)
        original_box[idx] = box  # adding original box coordinates
        plate = img[box[1]:box[3], box[0]:box[2]]  # extract plate from origin image
        plates.append(plate)  # append
    
    # prepare plate inputs
    plate_RGB, resized_data = preprocess(plates)
    np_batch = np.concatenate([data[0] for data in resized_data])
    inputs = np.ascontiguousarray(np_batch / 255.)
    
    # plate forwarding to get 4P coordinates
    outputs = session_forward(fpd_session, inputs)
    if np.size(outputs) == 0:
        return image
    ocr_info = ''
    color = tuple(colors['license_plate'])
    for idx, plate in enumerate(plate_RGB):
        if idx >= len(outputs):
            break
        output = outputs[idx]
        if len(output) != 4:
            continue
        
        cls_array = np.sort(np.array(output)[..., 5]).astype(np.uint8)
        if not compare_classes(cls_array, np.array([0, 1, 2, 3], dtype=np.uint8)):
            ocr_inputs = prepare_ocr(plate)
            plate_info = ocr(model, ocr_inputs)
            ocr_info = ''.join(plate_info)
            cv2.rectangle(image, tuple(original_box[idx][:2]), tuple(original_box[idx][2:]), color=color, thickness=2)
            cv2.putText(
              image,
              ocr_info, (original_box[idx][0], original_box[idx][1] - 2),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.85,
              color=(100, 100, 0),
              thickness=1,
              lineType=cv2.LINE_AA
            )
            continue
        
        src_dict = { label: [] for label in fpd_classes }
        for batch_idx, x0, y0, x1, y1, cls_idx, _ in output:  # point
            ratio, dwdh = resized_data[int(batch_idx)][1:]
            box = postprocess(np.array([x0, y0, x1, y1]), ratio, dwdh)  # class box
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            src_dict[fpd_classes[int(cls_idx)]].extend([center_x, center_y])
        
        # Do perspective transform
        pts = np.array([
          src_dict['top_left'],
          src_dict['top_right'],
          src_dict['bottom_right'],
          src_dict['bottom_left'],
        ]).astype(np.float32)
        warped = four_points_transform(plate, pts)
        
        # Extract OCR
        plate = warped.copy()
        ocr_inputs = prepare_ocr(plate)
        
        # do OCR
        ocr_info = ocr(model, ocr_inputs)
        ocr_info = ''.join(ocr_info)
        
        cv2.rectangle(image, tuple(original_box[idx][:2]), tuple(original_box[idx][2:]), color=color, thickness=2)
        cv2.putText(
          image,
          ocr_info, (original_box[idx][0], original_box[idx][1] - 2),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.85,
          color=(0, 255, 0),
          thickness=1,
          lineType=cv2.LINE_AA
        )
    
    return image, ocr_info


# if __name__ == '__main__':
#     # pipeline('/home/beosup/Documents/yolov7/data/lpd/test/images/9cd059a1e25b4d5dbb374ddfb7d6fe36.jpg')
#     image_paths = glob.glob('/home/beosup/Documents/yolov7/data/lpd/test/images/*.jpg')
#     os.makedirs('outputs', exist_ok=True)
#     start = time.time()
#     for image_path in image_paths:
#         iname = os.path.basename(image_path)
#         image = cv2.imread(image_path)
#         end = time.time()
#         image = pipeline(image)
#         print(f'Time taken = {time.time() - end:.4f}')
#         cv2.imwrite(f'outputs/{iname}', image)
#     end = time.time()
#     print(f'Time elapsed = {(end - start) / len(image_paths):.4f}')
