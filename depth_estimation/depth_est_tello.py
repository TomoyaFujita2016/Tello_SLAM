import argparse
from matplotlib import pyplot as plt
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from drawnow import drawnow, figure
import traceback
import tellopy
import av
import time

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "./FCRN-DepthPrediction/tensorflow/"))
import models


HEIGHT = 720
WIDTH = 960
CHANNELS = 3
BATCH_SIZE = 1

def predict(model_data_path):
    input_node = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, CHANNELS))
    net = models.ResNet50UpProj({'data': input_node}, BATCH_SIZE, 1, False)
    
    # drone init
    drone = tellopy.Tello()
    drone.connect()
    drone.wait_for_connection(60.0)
        
    with tf.Session() as sess:
        print('Loading the model')
        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        try:
            drone.connect()
            drone.wait_for_connection(60.0)
            drone.set_loglevel(drone.LOG_INFO)
            drone.set_exposure(0)
        
            container = av.open(drone.get_video_stream())
            frame_count = 0
            while True:
                for frame in container.decode(video=0):
                    frame_count = frame_count + 1
                    if (frame_count > 300) and (frame_count % 50 == 0):
                        img = pre_calc_img(frame.to_image())
                        pred = sess.run(net.get_output(), feed_dict={input_node: img})
                        def draw():
                            plt.imshow(pred[0,:,:,0],  interpolation='nearest')
                        drawnow(draw)
        
        except Exception as ex:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print(ex)
        finally:
            drone.quit()
            cv2.destroyAllWindows()

        
def pre_calc_img(img):
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
    return img
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path)
    
    os._exit(0)

if __name__ == '__main__':
    main()
