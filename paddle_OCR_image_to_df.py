import cv2 as cv
from paddleocr import PaddleOCR
import tensorflow as tf
import numpy as np
import pandas as pd

def paddle_OCR_image_to_df(input_img_path, output_img_path, out_format, language):
  image_cv = cv.imread(input_img_path)
  image_height = image_cv.shape[0]
  image_width = image_cv.shape[1]

  ocr = PaddleOCR(lang=language)
  output = ocr.ocr(input_img_path)

  probabilities = [line[1][1] for line in output[0]]
  texts = [line[1][0] for line in output[0]]

  image_boxes = image_cv.copy()
  boxes = []
  for index in range(0, len(output[0])):
    box = output[0][index][0]
    boxes.append(box)
    cv.rectangle(
      image_boxes, 
      (int(box[0][0]), int(box[0][1])),  
      (int(box[2][0]), int(box[2][1])), 
      (0, 0, 255), 
      1
    )

  # Resolution.
  vert_boxes = []
  horiz_boxes = []
  for box in boxes:
    x_h, x_v = 0,int(box[0][0])
    y_h, y_v = int(box[0][1]),0
    width_h,width_v = image_width, int(box[2][0]-box[0][0])
    height_h,height_v = int(box[2][1]-box[0][1]),image_height

    horiz_boxes.append([
      x_h,
      y_h,
      x_h + width_h,
      y_h + height_h
    ])
    vert_boxes.append([
      x_v,
      y_v,
      x_v+width_v,
      y_v+height_v
    ])

  horiz_out = tf.image.non_max_suppression(
      horiz_boxes,
      probabilities,
      max_output_size = 1000,
      iou_threshold=0.5,
      score_threshold=float('-inf'),
      name=None
  )

  horiz_lines = np.sort(np.array(horiz_out))

  vert_out = tf.image.non_max_suppression(
      vert_boxes,
      probabilities,
      max_output_size = 1000,
      iou_threshold=0.5,
      score_threshold=float('-inf'),
      name=None
  )
  vert_lines = np.sort(np.array(vert_out))
  out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]
  unordered_boxes = []

  for i in vert_lines:
    unordered_boxes.append(vert_boxes[i][0])

  ordered_boxes = np.argsort(unordered_boxes)

  def intersection(box_1, box_2):
    return [box_2[0], box_1[1],box_2[2], box_1[3]]

  def iou(box_1, box_2):
    x_1 = max(box_1[0], box_2[0])
    y_1 = max(box_1[1], box_2[1])
    x_2 = min(box_1[2], box_2[2])
    y_2 = min(box_1[3], box_2[3])

    inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
    if inter == 0:
        return 0
        
    box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
    box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))
    
    return inter / float(box_1_area + box_2_area - inter)

  for i in range(len(horiz_lines)):
    for j in range(len(vert_lines)):
      resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]] )

      for b in range(len(boxes)):
        the_box = [boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
        if(iou(resultant,the_box)>0.1):
          out_array[i][j] = texts[b]

  out_array=np.array(out_array)
  if (out_format == 'csv'):
    pd.DataFrame(out_array).to_csv(output_img_path)
  if (out_format == 'xlsx'):
    pd.DataFrame(out_array).to_excel(output_img_path)
  if (out_format == 'df'):
    return pd.DataFrame(out_array)
