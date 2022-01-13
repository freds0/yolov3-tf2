import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf

import os, tqdm, glob
from pascal_voc_writer import Writer

from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image_path', './data/', 'folder to input images')
flags.DEFINE_string('output_path', './output', 'folder to output annotated images')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_boolean('save_images', False, 'annotated images should also be saved or not')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    os.makedirs(FLAGS.output_path, exist_ok=True)

    for image_filepath in tqdm.tqdm(glob.glob(FLAGS.image_path + '/*.jpg')):
        img_raw = tf.image.decode_image(
            open(image_filepath, 'rb').read(), channels=3)

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        image_height, image_width = img_raw.shape[0], img_raw.shape[1]
        writer = Writer(image_filepath, image_width, image_height)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('xml detections:')
        for index in range(nums[0]):
            xmax, ymax, xmin, ymin = boxes[0][index]
            writer.addObject(class_names[index], int(xmin * image_width), int(ymin * image_height),
                             int(xmax * image_width), int(ymax * image_height))

        filename = os.path.basename(image_filepath)
        annotation_file = os.path.splitext(filename)[0] + '.xml'
        annotation_filepath = os.path.join('output', annotation_file)
        writer.save(annotation_filepath)
        logging.info('xml output saved to: {}'.format(annotation_filepath))

        if FLAGS.save_images:
            logging.info('image detections:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                   np.array(scores[0][i]),
                                                   np.array(boxes[0][i])))

            img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

            filename = os.path.basename(image_filepath)
            output_image_filepath = os.path.join(FLAGS.output_path, filename)
            cv2.imwrite(output_image_filepath, img)
            logging.info('image output saved to: {}'.format(output_image_filepath))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
