import argparse
import copy
import json
import os
import time
from typing import Tuple, Dict, Any
from mmcv.fileio.io import dump,load
import torch
import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.evaluate import NuScenesEval
from pyquaternion import Quaternion
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.splits import create_splits_scenes
import tqdm
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
import pycocotools.mask as mask_util
# from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from torchvision.transforms.functional import rotate
import cv2
import argparse
import random
from nuscenes.eval.common.loaders import load_gt, add_center_dist
# from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox,  DetectionMetricData,DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, dist_pr_curve, visualize_sample
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from IPython import embed
from matplotlib import pyplot as plt
from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.constants import TP_METRICS, DETECTION_NAMES, DETECTION_COLORS, TP_METRICS_UNITS, \
    PRETTY_DETECTION_NAMES, PRETTY_TP_METRICS
from nuscenes.utils.data_classes import LidarPointCloud
import mmcv


Axis = Any





from typing import Callable

import numpy as np

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData


def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """

    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.






def class_tp_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_recall: float,
                   dist_th_tp: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot the true positive curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name:
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Get metric data for given detection class with tp distance threshold.

    md = md_list[(detection_name, dist_th_tp)]
    min_recall_ind = round(100 * min_recall)
    if min_recall_ind <= md.max_recall_ind:
        # For traffic_cone and barrier only a subset of the metrics are plotted.
        rel_metrics = [m for m in TP_METRICS if not np.isnan(metrics.get_label_tp(detection_name, m))]
        ylimit = max([max(getattr(md, metric)[min_recall_ind:md.max_recall_ind + 1]) for metric in rel_metrics]) * 1.1
    else:
        ylimit = 1.0

    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Error', xlim=1,
                        min_recall=min_recall)
    ax.set_ylim(0, ylimit)

    # Plot the recall vs. error curve for each tp metric.
    for metric in TP_METRICS:
        tp = metrics.get_label_tp(detection_name, metric)

        # Plot only if we have valid data.
        if tp is not np.nan and min_recall_ind <= md.max_recall_ind:
            recall, error = md.recall[:md.max_recall_ind + 1], getattr(md, metric)[:md.max_recall_ind + 1]
        else:
            recall, error = [], []

        # Change legend based on tp value
        if tp is np.nan:
            label = '{}: n/a'.format(PRETTY_TP_METRICS[metric])
        elif min_recall_ind > md.max_recall_ind:
            label = '{}: nan'.format(PRETTY_TP_METRICS[metric])
        else:
            label = '{}: {:.2f} ({})'.format(PRETTY_TP_METRICS[metric], tp, TP_METRICS_UNITS[metric])
        if metric == 'trans_err':
            label += f' ({md.max_recall_ind})'  # add recall
            print(f'Recall: {detection_name}: {md.max_recall_ind/100}')
        ax.plot(recall, error, label=label)
    ax.axvline(x=md.max_recall, linestyle='-.', color=(0, 0, 0, 0.3))
    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


class DetectionBox_modified(DetectionBox):
    def __init__(self, *args, token=None, visibility=None, index=None, **kwargs):
        '''
        add annotation token
        '''
        super().__init__(*args, **kwargs)
        self.token = token
        self.visibility = visibility
        self.index = index

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'token': self.token,
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name,
            'visibility': self.visibility,
            'index': self.index

        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(
            token=content['token'],
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            detection_name=content['detection_name'],
            detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
            attribute_name=content['attribute_name'],
            visibility=content['visibility'],
            index=content['index'],
        )


def center_in_image(box, intrinsic: np.ndarray, imsize: Tuple[int, int], vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    center_3d = box.center.reshape(3, 1)
    center_img = view_points(center_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(center_img[0, :] > 0, center_img[0, :] < imsize[0])
    visible = np.logical_and(visible, center_img[1, :] < imsize[1])
    visible = np.logical_and(visible, center_img[1, :] > 0)
    visible = np.logical_and(visible, center_3d[2, :] > 1)

    in_front = center_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))


def exist_corners_in_image_but_not_all(box, intrinsic: np.ndarray, imsize: Tuple[int, int],
                                       vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible in images but not all corners in image .
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if any(visible) and not all(visible) and all(in_front):
        return True
    else:
        return False

def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    # with open(result_path) as f:
    #     data = json.load(f)
    data = load(result_path)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta

def load_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False):
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """

    # Init.
    if box_cls == DetectionBox_modified:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'
    index_map = {}
    for scene in nusc.scene:
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        index_map[first_sample_token] = 1
        index = 2
        while sample['next'] != '':
            sample = nusc.get('sample', sample['next'])
            index_map[sample['token']] = index
            index += 1

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox_modified:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                sample_boxes.append(
                    box_cls(
                        token=sample_annotation_token,
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=attribute_name,
                        visibility=sample_annotation['visibility_token'],
                        index=index_map[sample_token]
                    )
                )
            elif box_cls == TrackingBox:
                assert False
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations


def filter_eval_boxes_by_id(nusc: NuScenes,
                            eval_boxes: EvalBoxes,
                            id=None,
                            verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param is: the anns token set that used to keep bboxes.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on anns
        total += len(eval_boxes[sample_token])
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.token in id:
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After anns based filtering: %d" % anns_filter)

    return eval_boxes


def filter_eval_boxes_by_visibility(
        ori_eval_boxes: EvalBoxes,
        visibility=None,
        verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param is: the anns token set that used to keep bboxes.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    eval_boxes = copy.deepcopy(ori_eval_boxes)
    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):
        # Filter on anns
        total += len(eval_boxes[sample_token])
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.visibility == visibility:
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After visibility based filtering: %d" % anns_filter)

    return eval_boxes


def filter_by_sample_token(ori_eval_boxes, valid_sample_tokens=[],  verbose=False):
    eval_boxes = copy.deepcopy(ori_eval_boxes)
    for sample_token in eval_boxes.sample_tokens:
        if sample_token not in valid_sample_tokens:
            eval_boxes.boxes.pop(sample_token)
    return eval_boxes


def filter_eval_boxes_by_overlap(nusc: NuScenes,
                                 eval_boxes: EvalBoxes,
                                 verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. basedon overlap .
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    cams = ['CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_FRONT_LEFT']

    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on anns
        total += len(eval_boxes[sample_token])
        sample_record = nusc.get('sample', sample_token)
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            count = 0
            for cam in cams:
                '''
                copy-paste form nuscens
                '''
                sample_data_token = sample_record['data'][cam]
                sd_record = nusc.get('sample_data', sample_data_token)
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                sensor_record = nusc.get('sensor', cs_record['sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                cam_intrinsic = np.array(cs_record['camera_intrinsic'])
                imsize = (sd_record['width'], sd_record['height'])
                new_box = Box(box.translation, box.size, Quaternion(box.rotation),
                              name=box.detection_name, token='')

                # Move box to ego vehicle coord system.
                new_box.translate(-np.array(pose_record['translation']))
                new_box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                new_box.translate(-np.array(cs_record['translation']))
                new_box.rotate(Quaternion(cs_record['rotation']).inverse)

                if center_in_image(new_box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
                    count += 1
                # if exist_corners_in_image_but_not_all(new_box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
                #    count += 1

            if count > 1:
                with open('center_overlap.txt', 'a') as f:
                    try:
                        f.write(box.token + '\n')
                    except:
                        pass
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    verbose = True

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After anns based filtering: %d" % anns_filter)

    return eval_boxes

def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field

def filter_eval_boxes(nusc: NuScenes,
                      eval_boxes: EvalBoxes,
                      max_dist_x: Dict[str, float],
                      max_dist_y: Dict[str, float],
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          abs(box.ego_translation[0]) < max_dist_x[box.__getattribute__(class_field)] \
                                          and abs(box.ego_translation[1]) < max_dist_y[box.__getattribute__(class_field)]]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

        # Perform bike-rack filtering.
        sample_anns = nusc.get('sample', sample_token)['anns']
        bikerack_recs = [nusc.get('sample_annotation', ann) for ann in sample_anns if
                         nusc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack']
        bikerack_boxes = [Box(rec['translation'], rec['size'], Quaternion(rec['rotation'])) for rec in bikerack_recs]
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.__getattribute__(class_field) in ['bicycle', 'motorcycle']:
                in_a_bikerack = False
                for bikerack_box in bikerack_boxes:
                    if np.sum(points_in_box(bikerack_box, np.expand_dims(np.array(box.translation), axis=1))) > 0:
                        in_a_bikerack = True
                if not in_a_bikerack:
                    filtered_boxes.append(box)
            else:
                filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes
        bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
        print("=> After bike rack filtering: %d" % bike_rack_filter)

    return eval_boxes

from math import sin,cos
def rpy_to_quaternion(rpy: np.ndarray) -> np.ndarray:
    """
    Converts an rpy angle to a quaternion.
    :param rpy: An rpy angle.
    :return: A quaternion.
    """
    assert len(rpy) == 3
    roll, pitch, yaw = rpy
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp
    return np.array([w, x, y, z], dtype=rpy.dtype)
class NuScenesEval_custom(NuScenesEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """

    def __init__(self,
                #  nusc: NuScenes,
                 dataroot: str,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 overlap_test=False,
                 eval_mask=False,
                 data_infos=None,
                 annfile='1',
                 ):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """

        # self.nusc = nusc
        self.data_root = dataroot
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.overlap_test = overlap_test
        self.eval_mask = eval_mask
        self.data_infos = data_infos
        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                   verbose=verbose)

        # filtered_bbox = []
        # for bbox in self.pred_boxes.all:
        #     x, y = bbox.translation[:2]
        #     if -20<x<20 and 10<y<10:
        #         filtered_bbox.append(bbox)
        # self.pred_boxes.all = filtered_bbox

        import pickle
        data_zsy = pickle.load(open(annfile, 'rb'))
        infos, metadata = data_zsy['infos'], data_zsy['metadata']
        box_cls = DetectionBox_modified
        all_annotations = EvalBoxes()
        for info in infos:
            gt_boxes=[] 
            # if info['token'] not in self.pred_boxes.sample_tokens:
            #     continue
            for gt_box, gt_name in zip(info['gt_boxes'],info['gt_names']):
                x,y,z,w,l,h,yaw = gt_box
                if -20 < x < 20 and -10 < y < 10:
                    gt_boxes.append(
                            box_cls(
                                token=info['token'],
                                sample_token=info['token'],
                                translation=[x,y,z],
                                size=[w,l,h],
                                rotation=rpy_to_quaternion(np.array([0,0,yaw])),
                                velocity=[1,1],
                                num_pts=1,
                                detection_name=gt_name,
                                detection_score=-1.0,  # GT samples do not have a score.
                                attribute_name='',
                                visibility=None,
                                index=info['frame_idx'],
                            )
                        )
            all_annotations.add_boxes(info['token'], gt_boxes)
            
                
        # xyzwlhy
        # self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox_modified, verbose=verbose)
        self.gt_boxes = all_annotations

       
        # assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            # "Samples in split doesn't match samples in predictions."

        # # Add center distances.
        # self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        # self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).

        # if verbose:
        #     print('Filtering predictions')
        # self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range_x, self.cfg.class_range_y, verbose=verbose)
        # if verbose:
        #     print('Filtering ground truth annotations')
        # self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range_x, self.cfg.class_range_y, verbose=verbose)

        # if self.overlap_test:
        #     self.pred_boxes = filter_eval_boxes_by_overlap(self.nusc, self.pred_boxes)

        #     self.gt_boxes = filter_eval_boxes_by_overlap(self.nusc, self.gt_boxes, verbose=True)

        self.all_gt = copy.deepcopy(self.gt_boxes)
        self.all_preds = copy.deepcopy(self.pred_boxes)
        self.sample_tokens = self.gt_boxes.sample_tokens

        # self.index_map = {}
        # for scene in nusc.scene:
        #     first_sample_token = scene['first_sample_token']
        #     sample = nusc.get('sample', first_sample_token)
        #     self.index_map[first_sample_token] = 1
        #     index = 2
        #     while sample['next'] != '':
        #         sample = nusc.get('sample', sample['next'])
        #         self.index_map[sample['token']] = index
        #         index += 1

    def update_gt(self, type_='vis', visibility='1', index=1):
        if type_ == 'vis':
            self.visibility_test = True
            if self.visibility_test:
                '''[{'description': 'visibility of whole object is between 0 and 40%',
                'token': '1',
                'level': 'v0-40'},
                {'description': 'visibility of whole object is between 40 and 60%',
                'token': '2',
                'level': 'v40-60'},
                {'description': 'visibility of whole object is between 60 and 80%',
                'token': '3',
                'level': 'v60-80'},
                {'description': 'visibility of whole object is between 80 and 100%',
                'token': '4',
                'level': 'v80-100'}]'''

                self.gt_boxes = filter_eval_boxes_by_visibility(self.all_gt, visibility, verbose=True)

        elif type_ == 'ord':

            valid_tokens = [key for (key, value) in self.index_map.items() if value == index]
            # from IPython import embed
            # embed()
            self.gt_boxes = filter_by_sample_token(self.all_gt, valid_tokens)
            self.pred_boxes = filter_by_sample_token(self.all_preds, valid_tokens)
        self.sample_tokens = self.gt_boxes.sample_tokens


    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()

        # print(self.cfg.dist_fcn_callable, self.cfg.dist_ths)
        # self.cfg.dist_ths = [0.3]
        # self.cfg.dist_fcn_callable
        self.cfg.class_names = ['bicycle', 'pedestrian', 'traffic_cone', 
                                'generic_object', 'vehicle'] # classes to be printed

        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)
            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))
            
    def main(self,
            plot_examples: int = 0,
            render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                sample_token,
                                self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                # Don't render test GT.
                                self.pred_boxes,
                                eval_range=max(self.cfg.class_range.values()),
                                savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        # print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        import pandas as pd

        # 创建一个字典，包含表格数据
        data = {
            'Object Class': [],
            'AP': [],
            'ATE': [],
            'ASE': [],
            'AOE': [],
            'AVE': [],
            # 'AAE': [],

        }
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in reversed(class_aps.keys()):
            data['Object Class'].append(class_name)
            data['AP'].append(round(class_aps[class_name], 3))
            data['ATE'].append(round(class_tps[class_name]['trans_err'], 3))
            data['ASE'].append(round(class_tps[class_name]['scale_err'], 3))
            data['AOE'].append(round(class_tps[class_name]['orient_err'], 3))
            data['AVE'].append(round(class_tps[class_name]['vel_err'], 3))

            # data['AAE'].append(class_tps[class_name]['attr_err'])

            # print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
            #     % (class_name, class_aps[class_name],
            #         class_tps[class_name]['trans_err'],
            #         class_tps[class_name]['scale_err'],
            #         class_tps[class_name]['orient_err'],
            #         class_tps[class_name]['vel_err'],
            #         class_tps[class_name]['attr_err']))
        # 将字典转换为 DataFrame

        df = pd.DataFrame(data)
        print(df)
        # print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        # class_aps = metrics_summary['mean_dist_aps']
        # class_tps = metrics_summary['label_tp_errors']
        # for class_name in class_aps.keys():
        #     print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
        #         % (class_name, class_aps[class_name],
        #             class_tps[class_name]['trans_err'],
        #             class_tps[class_name]['scale_err'],
        #             class_tps[class_name]['orient_err'],
        #             class_tps[class_name]['vel_err'],
        #             class_tps[class_name]['attr_err']))

        return metrics_summary


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='data/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=0,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = NuScenesEval_custom(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                                    output_dir=output_dir_, verbose=verbose_)
    for vis in ['1', '2', '3', '4']:
        nusc_eval.update_gt(type_='vis', visibility=vis)
        print(f'================ {vis} ===============')
        nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
    #for index in range(1, 41):
    #    nusc_eval.update_gt(type_='ord', index=index)
    #
