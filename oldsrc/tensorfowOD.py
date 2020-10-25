import numpy as np
import warnings
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.utils.linear_assignment_ import linear_assignment
    import tensorflow as tf
    from tensorflow.python.client import timeline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
# from scipy.optimize import linear_sum_assignment
import cv2
from oldsrc import tracker
import time
# Tensorflow localization/detection model
# Single-shot-dectection with mobile net architecture trained on COCO
# dataset
detect_model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'


class OD:
    def __init__(self):
        # Global variables to be used by funcitons of VideoFileClop
        self.frame_count = 0  # frame counter
        self.max_age = 15  # no.of consecutive unmatched detection before
        # a track is deleted
        self.min_hits = 1  # no. of consecutive matches needed to establish a track
        self.tracker_list = []  # list for trackers
        # list for track ID
        self.track_id_list = deque(['1', '2', '3', '4', '5', '6', '7', '7', '8', '9', '10'])
        self.doDetect=True #MNR
        # setup tensorflow graph
        self.detection_graph = tf.Graph()

        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    # Helper function to convert image into numpy array
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        # Helper function to convert normalized box coordinates to pixels

    def box_normal_to_pixel(self, box, dim):
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
        return np.array(box_pixel)

    def box_iou2(self, a, b):
        '''
        Helper funciton to calculate the ratio between intersection and the union of
        two boxes a and b
        a[0], a[1], a[2], a[3] <-> left, up, right, bottom
        '''

        w_intsec = np.maximum(0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
        h_intsec = np.maximum(0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
        s_intsec = w_intsec * h_intsec
        s_a = (a[2] - a[0]) * (a[3] - a[1])
        s_b = (b[2] - b[0]) * (b[3] - b[1])

        return float(s_intsec) / (s_a + s_b - s_intsec)

    def nms(self, boxes, scores, classes, iou_thrd=0.3):
        for d1 in range(len(boxes)):
            for d2 in range(d1+1, len(boxes)):
                iou = self.box_iou2(boxes[d1], boxes[d2])
                if(iou>0.7):
                    print('iou: '+str(iou)+' b1: '+str(boxes[d1])+' c: '+str(classes[d1])+' s: '+str(scores[d1])+'...'+' b2: '+str(boxes[d2])+' c: '+str(classes[d2])+' s: '+str(scores[d2]))

    def assign_detections_to_trackers(self, trackers, detections, z_class, iou_thrd=0.3):
        '''
            From current list of trackers and new detections, output matched detections,
            unmatchted trackers, unmatched detections.
            '''
        IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
        for t, trk in enumerate(trackers):
            # trk = convert_to_cv2bbox(trk)
            for d, det in enumerate(detections):
                #   det = convert_to_cv2bbox(det)
                tmp_tracker = self.tracker_list[t]
                IOU_mat[t, d] = self.box_iou2(trk, det) if (tmp_tracker.object_class==z_class[d]) else 0 # MNR

                # Produces matches
        # Solve the maximizing the sum of IOU assignment problem using the
        # Hungarian algorithm (also known as Munkres algorithm)

        # matched_idx = linear_sum_assignment(-IOU_mat)
        matched_idx = linear_assignment(-IOU_mat)

        unmatched_trackers, unmatched_detections = [], []
        for t, trk in enumerate(trackers):
            if (t not in matched_idx[:, 0]):
                unmatched_trackers.append(t)

        for d, det in enumerate(detections):
            if (d not in matched_idx[:, 1]):
                unmatched_detections.append(d)

        matches = []

        # For creating trackers we consider any detection with an
        # overlap less than iou_thrd to signifiy the existence of
        # an untracked object

        for m in matched_idx:
            if (IOU_mat[m[0], m[1]] < iou_thrd):
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def draw_box_label(self, id, img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
        '''
        Helper funciton for drawing the bounding boxes and the labels
        bbox_cv2 = [left, top, right, bottom]
        '''
        # box_color= (0, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.7
        font_color = (0, 0, 0)
        left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]

        # Draw the bounding box
        cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)

        if show_label:
            # Draw a filled box on top of the bounding box (as the background for the labels)
            cv2.rectangle(img, (left - 2, top - 45), (right + 2, top), box_color, -1, 1)

            # Output the labels that show the x and y coordinates of the bounding box center.
            text_x = 'id=' + str(id)
            cv2.putText(img, text_x, (left, top - 25), font, font_size, font_color, 1, cv2.LINE_AA)
            text_y = 'y=' + str((top + bottom) / 2)
            cv2.putText(img, text_y, (left, top - 5), font, font_size, font_color, 1, cv2.LINE_AA)

        return img

    def track(self, img, z_box, z_score, z_class):
        self.frame_count += 1
        x_box = []
        if len(self.tracker_list) > 0:
            for trk in self.tracker_list:
                x_box.append(trk.box)
        matched, unmatched_dets, unmatched_trks = self.assign_detections_to_trackers(x_box, z_box,z_class, iou_thrd=0.3)
        print('Detected: ' + str(len(z_box)) + ', tracker_list: ' + str(len(self.tracker_list)) + ', matched: ' + str(
            len(matched)) + ', unmatched_dets: ' + str(len(unmatched_dets)) + ', unmatched_trks: ' + str(
            len(unmatched_trks)))

        # Deal with matched detections
        if matched.size > 0:
            for trk_idx, det_idx in matched:
                z = z_box[det_idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = self.tracker_list[trk_idx]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                x_box[trk_idx] = xx
                tmp_trk.box = xx
                tmp_trk.hits += 1

        # Deal with unmatched detections
        if len(unmatched_dets) > 0:
            for idx in unmatched_dets:
                z = z_box[idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = tracker.Tracker()  # Create a new tracker
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.id = self.track_id_list.popleft()  # assign an ID for the tracker
                tmp_trk.object_class = z_class[idx]
                tmp_trk.object_conf = z_score[idx] # maybe i should change this to average #MNR
                print(tmp_trk.id)
                self.tracker_list.append(tmp_trk)
                x_box.append(xx)

        # Deal with unmatched tracks
        if len(unmatched_trks) > 0:
            for trk_idx in unmatched_trks:
                tmp_trk = self.tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                x_box[trk_idx] = xx
            # The list of tracks to be annotated
            good_tracker_list = []
            for trk in self.tracker_list:
                if ((trk.hits >= self.min_hits) and (trk.no_losses <= self.max_age)):
                    good_tracker_list.append(trk)
                    x_cv2 = trk.box
                    img = self.draw_box_label(trk.id, img, x_cv2)  # Draw the bounding boxes on the
                    # images
            # Book keeping
            deleted_tracks = filter(lambda x: x.no_losses > self.max_age, self.tracker_list)

            for trk in deleted_tracks:
                self.track_id_list.append(trk.id)

            tracker_list = [x for x in self.tracker_list if x.no_losses <= self.max_age]
            return img

    def detector(self, image, thr=0.5):
        # print('new image')
        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            t1 = time.time()
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded}, options=run_options, run_metadata=run_metadata)
            t2 = time.time()
            print('run tensor: '+str(t2-t1))
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)

            cls = classes.tolist()

            # The ID for car is 3
            idx_vec = [i for i, v in enumerate(cls) if (scores[i] > thr)]  # (v == 1) and

            if len(idx_vec) != 0:
                tmp_boxes = []
                tmp_classes = []
                tmp_confidence = []
                for idx in idx_vec:
                    dim = image.shape[0:2]
                    box = self.box_normal_to_pixel(boxes[idx], dim)
                    # box_h = box[2] - box[0]
                    # box_w = box[3] - box[1]
                    # ratio = box_h / (box_w + 0.01)
                    #
                    # # if ((ratio < 0.8) and (box_h>20) and (box_w>20)):
                    tmp_boxes.append(box)
                    tmp_classes.append(scores[idx])
                    tmp_confidence.append(classes[idx])
                    # print(boxes[idx], ', confidence: ', scores[idx], 'class:', classes[idx])  # , 'ratio:', ratio
                    '''   
                    else:
                        print('wrong ratio or wrong size, ', box, ', confidence: ', scores[idx], 'ratio:', ratio)
                    '''
                # self.car_boxes = tmp_car_boxes

                return tmp_boxes, tmp_classes, tmp_confidence

        return 0, 0, 0


run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()


def getBoxedImage(img, boxes, scores, classes):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(img)
    rect = patches.Rectangle((10,20), 40, 20,
                             linewidth=3, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    for i in range(len(boxes)):
        rect = patches.Rectangle((boxes[i][1], boxes[i][0]), boxes[i][3] - boxes[i][1], boxes[i][2] - boxes[i][0],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(boxes[i][1], boxes[i][0], 'id=' + str(classes[i]) + ' ' + str(scores[i] * 100) + "%", style='italic',
                fontsize=15, color='red')  # ,bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        cv2.rectangle(image, (boxes[i][1], boxes[i][0]), (boxes[i][3],  boxes[i][2] ), (0, 255, 0), 2)
        print(boxes[i], ', confidence: ', scores[i], 'class:', classes[i])  # , 'ratio:', ratio

    return img


if __name__ == "__main__":
    det = OD()
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('test_images/race.mp4')
    i = 0
    avg_net = 0
    avg_track = 0

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    # initialize OpenCV's special multi-object tracker
    trackers = cv2.MultiTracker_create()

    while (cap.isOpened()):# and i < 50):
        i = i + 1
        # print('image: '+str(i))
        ret, image = cap.read()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (ret == True):
            boxes, scores, classes = det.detector(image)
            if(boxes != 0):
                image = getBoxedImage(image, boxes, scores, classes )
            # if (boxes != 0):
            #     for idx, box in enumerate(boxes):
            #         print('Detect: '+str(classes[idx])+', '+str(scores[idx])+'%')
            #         (x, y, w, h) = [int(v) for v in box]
            #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # if(i%5==0):
            #     t1 = time.time()
            #     boxes, scores, classes = det.detector(image)
            #     if (boxes == 0):
            #         print('no find')
            #         continue
            #     t2 = time.time()
            #     avg_net = avg_net + t2 - t1
            #
            #     # should check previous objects ...
            #     for box in boxes:
            #         # create a new object tracker for the bounding box and add it
            #         # to our multi-object tracker
            #         # tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            #         tracker = OPENCV_OBJECT_TRACKERS["mosse"]()
            #         trackers.add(tracker, image, box)
            # else:
            #     # grab the updated bounding box coordinates (if any) for each
            #     # object that is being tracked
            #     (success, boxes) = trackers.update(image)
            #
            #     # loop over the bounding boxes and draw then on the frame
            #     for box in boxes:
            #         (x, y, w, h) = [int(v) for v in box]
            #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #
            #     # show the output frame
            #     cv2.imshow("Frame", image)
            #     key = cv2.waitKey(1) & 0xFF
            # cv2.imwrite('test_images/'+str(i)+'.jpg', image)
            cv2.imshow("SSD", image)
            key = cv2.waitKey(1) & 0xFF
            # # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
    print('times  detector:'+str(avg_net/i)+', track: '+str(avg_track/i))
    # Create the Timeline object, and write it to a json
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)
