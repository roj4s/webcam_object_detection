import nn.DarknetFrame as darknet
import cv2

class Detection:

    def __init__(self, darknet_detection_output, w_scale, h_scale):
        self.w_scale = w_scale
        self.h_scale = h_scale
        self.darknet_detection_output = darknet_detection_output
        self.label = darknet_detection_output[0].decode('utf-8')
        self.confidence = darknet_detection_output[1]
        self.x = darknet_detection_output[2][0] * w_scale
        self.y = darknet_detection_output[2][1] * h_scale
        self.w = darknet_detection_output[2][2] * w_scale
        self.h = darknet_detection_output[2][3] * h_scale

    def __str__(self):
        return str(self.__dict__)

    def as_json(self):
        return self.__dict__


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
        fw = frame.shape[1]
        fh = frame.shape[0]
        nw = darknet.network_width(self.net_main)
        nh = darknet.network_height(self.net_main)
        w_scale = fw/nw
        h_scale = fh/nh

        frame_resized = cv2.resize(frame,
                                   (nw,
                                    nh),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(
            self.darknet_image, frame_resized.tobytes())

        nn_detections = darknet.detect_image(
            self.net_main, self.meta_main, self.darknet_image, thresh)

        detections = [Detection(d, w_scale, h_scale) for d in nn_detections]

        return detections
