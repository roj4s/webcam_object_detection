import nn.DarknetFrame as darknet
import cv2

class Detector:

    def __init__(self, config_file='nn/configs/yolov3-tiny.cfg',
                 weights_file="nn/weights/yolov3-tiny.weights",
                 classes_data_file="nn/configs/coco.data"):

        self.net_main, self.meta_main = darknet.load_nn(config_file,
                                                        weights_file,
                                                        classes_data_file)

        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(
            darknet.network_width(self.net_main),
            darknet.network_height(self.net_main),
            3)

    def detect(self, frame, thresh=0.5):
        frame_resized = cv2.resize(frame,
                                   (darknet.network_width(self.net_main),
                                    darknet.network_height(self.net_main)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(
            self.darknet_image, frame_resized.tobytes())

        nn_detections = darknet.detect_image(
            self.net_main, self.meta_main, self.darknet_image, thresh)

        return nn_detections
