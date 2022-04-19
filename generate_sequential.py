#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Author: Yuhang He
# Email: yuhanghe01@gmail.com
# Note: this script is used to concatenate a sequence of scans together
import os
import numpy as np
from collections import deque
from numpy.linalg import inv

import glob
import proj_pcd2img

def parse_calibration(filename):
    """ read calibration file with given filename
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    # file = open(filename)

    file_list = [line_tmp.rstrip('\n') for line_tmp in open( filename, 'r' ).readlines() ]
    file_list.reverse()

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in file_list:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses

def parse_poses_default(filename, calibration):
    """ read poses file with per-scan poses from given filename

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    # file_list = [line_tmp.rstrip('\n') for line_tmp in open( filename, 'r' ).readlines() ]
    # file_list.reverse()

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


def load_velo_scan( velo_scan_filename ):
    velo_scan = np.fromfile( velo_scan_filename, dtype=np.float32 )
    velo_scan = velo_scan.reshape( (-1, 4 ) )

    return velo_scan

def load_label( label_filename ):
    label = np.fromfile( label_filename, dtype = np.int32 )
    label = label.reshape( (-1) )

    return label

def load_veloscan_gtlabel_predlabel( velo_scan_file_name,
                                     gt_label_dir,
                                     pred_label_dir ):
    velo_scan = load_velo_scan( velo_scan_file_name )
    velo_scan_basename = os.path.basename( velo_scan_file_name )

    gt_label_basename = os.path.splitext( velo_scan_basename )[0] + '.label'
    gt_label = load_label( os.path.join( gt_label_dir, gt_label_basename ) )

    pred_label_basename = gt_label_basename
    pred_label = load_label( os.path.join( pred_label_dir, pred_label_basename ) )

    return velo_scan, gt_label, pred_label


def process_one_folder( folder ):
    print("Processing {} ".format(folder), end="", flush=True)
    calib_file_name = os.path.join( folder, 'calib.txt' )
    pose_file_name = os.path.join( folder, 'poses.txt' )
    velo_scan_dir = os.path.join( folder, 'velodyne' )
    gt_label_dir = os.path.join( folder, 'label_gt' )
    pred_label_dir = os.path.join( folder, 'label_pred' )

    velo_scan_output_dir = os.path.join( folder, 'velodyne_concat' )
    if not os.path.exists( velo_scan_output_dir ):
        os.makedirs( velo_scan_output_dir )

    gt_label_output_dir = os.path.join( folder, 'label_gt_concat' )
    if not os.path.exists( gt_label_output_dir ):
        os.makedirs( gt_label_output_dir )

    pred_label_output_dir = os.path.join( folder, 'label_pred_concat' )
    if not os.path.exists( pred_label_output_dir ):
        os.makedirs( pred_label_output_dir )

    scan_files = sorted( glob.glob( os.path.join( velo_scan_dir, '*bin' ) ),
                         reverse = True )

    assert len( scan_files ) > 0

    calibration = parse_calibration( calib_file_name )
    poses = parse_poses( pose_file_name, calibration )

    #step0: calculate the row number at first in oder to allocate the memory in advance
    row_num = 0
    for scan_filename in scan_files:
        velo_scan_tmp = load_velo_scan( scan_filename )
        row_num_tmp, _ = velo_scan_tmp.shape
        row_num += row_num_tmp

    concat_velo_scan = np.zeros( shape = (row_num, 4),
                                 dtype = np.float32 )
    concat_gt_label = np.zeros( shape = ( row_num, 1),
                                dtype = np.int32 )
    concat_pred_label = np.zeros( shape = ( row_num, 1 ),
                                  dtype = np.int32 )

    #step1: load the last frame

    velo_scan_sentinel, gt_label_sentinel, pred_label_sentinel = load_veloscan_gtlabel_predlabel( scan_files[0],
                                                                                            gt_label_dir,
                                                                                            pred_label_dir )

    row_start = 0
    row_num_tmp = velo_scan_sentinel.shape[0]
    concat_velo_scan[ row_start:row_start+row_num_tmp, :] = velo_scan_sentinel

    gt_label_sentinel = gt_label_sentinel.reshape( (-1, 1) )
    pred_label_sentinel = pred_label_sentinel.reshape( (-1, 1) )
    concat_gt_label[row_start:row_start + row_num_tmp, :] = gt_label_sentinel
    concat_pred_label[row_start:row_start + row_num_tmp,:] = pred_label_sentinel
    row_start += row_num_tmp
    pose_sentinel = poses[0]

    #we process the scan in reversible
    for idx, scan_filename in enumerate( scan_files ):
        if idx == 0:
            continue
        if idx % 100 == 0:
            print('processed {}/{} scans.'.format( idx, len(scan_files) ) )
        # read scan and labels, get pose
        velo_scan_tmp, label_gt_tmp, label_pred_tmp = load_veloscan_gtlabel_predlabel( scan_files[ idx ],
                                                                                       gt_label_dir,
                                                                                       pred_label_dir )
        label_gt_tmp = label_gt_tmp.reshape( (-1, 1) )
        label_pred_tmp = label_pred_tmp.reshape( (-1, 1) )
        points = np.ones( ( velo_scan_tmp.shape ) )
        points[:, 0:3] = velo_scan_tmp[:, 0:3]
        remission = velo_scan_tmp[:, 3]

        pose_tmp = poses[idx]

        diff = np.matmul(inv(pose_sentinel), pose_tmp)
        tpoints = np.matmul(diff, points.T).T
        tpoints[:, 3] = remission
        row_num_tmp = tpoints.shape[0]
        concat_velo_scan[row_start:row_start + row_num_tmp, :] = tpoints
        concat_gt_label[row_start:row_start + row_num_tmp, :] = label_gt_tmp
        concat_pred_label[row_start:row_start + row_num_tmp, :] = label_pred_tmp
        row_start += row_num_tmp

    #write the result
    output_velo_scan_file_name = os.path.join( velo_scan_output_dir, 'velo_scan.bin' )
    concat_velo_scan.tofile( output_velo_scan_file_name )
    output_label_gt_file_name = os.path.join( gt_label_output_dir, 'velo_scan.label' )
    concat_gt_label = np.squeeze( concat_gt_label )
    concat_gt_label.tofile( output_label_gt_file_name )
    output_label_pred_file_name = os.path.join( pred_label_output_dir, 'velo_scan.label' )
    concat_pred_label = np.squeeze( concat_pred_label )
    concat_pred_label.tofile( output_label_pred_file_name )

    print("finished.")

def process_one_folder_with_deque( folder ):
    print("Processing {}.".format(folder), end = "", flush = True )
    calib_file_name = os.path.join( folder, 'calib.txt' )
    pose_file_name = os.path.join( folder, 'poses.txt' )
    velo_scan_dir = os.path.join( folder, 'velodyne' )
    gt_label_dir = os.path.join( folder, 'label_gt' )
    pred_label_dir = os.path.join( folder, 'label_pred' )

    velo_scan_output_dir = os.path.join( folder, 'velodyne_concat' )
    #velo_scan_output_dir = velo_scan_output_dir.replace('sequences_result3', 'sequences_result4')
    if not os.path.exists( velo_scan_output_dir ):
        os.makedirs( velo_scan_output_dir )

    gt_label_output_dir = os.path.join( folder, 'label_gt_concat' )
    #gt_label_output_dir = gt_label_output_dir.replace('sequences_result3', 'sequences_result4')
    if not os.path.exists( gt_label_output_dir ):
        os.makedirs( gt_label_output_dir )

    pred_label_output_dir = os.path.join( folder, 'label_pred_concat' )
    #pred_label_output_dir = pred_label_output_dir.replace('sequences_result3','sequences_result4')
    if not os.path.exists( pred_label_output_dir ):
        os.makedirs( pred_label_output_dir )

    frame_ID_output_dir = os.path.join( folder, 'frame_ID_concat' )
    #frame_ID_output_dir = frame_ID_output_dir.replace('sequences_result3', 'sequences_result4')
    if not os.path.exists( frame_ID_output_dir ):
        os.makedirs( frame_ID_output_dir )

    loc_2D_output_dir = os.path.join( folder, 'loc_2D_concat' )
    #loc_2D_output_dir = loc_2D_output_dir.replace('sequences_result3', 'sequences_result4')
    if not os.path.exists( loc_2D_output_dir ):
        os.makedirs( loc_2D_output_dir )

    scan_files = sorted( glob.glob( os.path.join( velo_scan_dir, '*bin' ) ),
                         reverse = False )
    #scan_files = list()
    # for i in range(2395, 2460, 1):
    #     velo_filename = '{:06d}.bin'.format(i)
    #     scan_files.append(os.path.join(velo_scan_dir, velo_filename))

    assert len( scan_files ) > 0

    calibration = parse_calibration( calib_file_name )
    poses = parse_poses_default( pose_file_name, calibration )

    history = deque()

    for i, scan_filename in enumerate(scan_files):
        if i % 1000 == 0:
            print('processed {}/{} lines.'.format( i, len( scan_files) ) )
        # read scan and labels, get pose
        scan = np.fromfile( scan_filename, dtype=np.float32 )
        scan = scan.reshape((-1, 4))

        loc_2D = proj_pcd2img.get_pcd_2D_loc( scan, calib_file_name )

        velo_scan_basename = os.path.basename( scan_filename )

        gt_label_basename = os.path.splitext(velo_scan_basename)[0] + '.label'
        gt_label = load_label(os.path.join(gt_label_dir, gt_label_basename))

        pred_label_basename = gt_label_basename
        pred_label = load_label(os.path.join(pred_label_dir, pred_label_basename))

        # convert points to homogenous coordinates (x, y, z, 1)
        points = np.ones((scan.shape))
        points[:, 0:3] = scan[:, 0:3]
        remissions = scan[:, 3]

        frame_ID_num = int( velo_scan_basename.replace('.bin','') )
        frame_ID = np.ones( shape = (scan.shape[0]), dtype = np.int32 )*frame_ID_num

        pose_idx = int(os.path.splitext(velo_scan_basename)[0])
        pose = poses[pose_idx]

        # prepare single numpy array for all points that can be written at once.
        if i == len( scan_files ) -1:
            start = 0
            num_concat_points = points.shape[0]
            num_concat_points += sum([past["points"].shape[0] for past in history])
            concated_points = np.zeros((num_concat_points * 4), dtype=np.float32)
            concated_ori_points = np.zeros((num_concat_points*4), dtype=np.float32)
            concated_labels_gt = np.zeros((num_concat_points), dtype=np.int32)
            concated_labels_pred = np.zeros((num_concat_points), dtype=np.int32)
            concated_frame_ID = np.zeros((num_concat_points), dtype=np.int32)
            concated_loc_2D = np.zeros((num_concat_points * 2), dtype=np.int32)

            concated_points[4 * start:4 * (start + points.shape[0])] = scan.reshape((-1))
            concated_ori_points[4 * start:4 * (start + points.shape[0])] = scan.reshape((-1))
            concated_labels_gt[start:start + points.shape[0]] = gt_label
            concated_labels_pred[start:start + points.shape[0]] = pred_label
            concated_frame_ID[start:start + points.shape[0]] = frame_ID
            concated_loc_2D[2*start:2*(start + points.shape[0])] = loc_2D.reshape((-1))
            start += points.shape[0]
            camera_pose_dict = dict()
            for past in history:
                diff = np.matmul(inv(pose), past["pose"])
                tpoints = np.matmul(diff, past["points"].T).T
                tpoints[:, 3] = past["remissions"]
                tpoints = tpoints.reshape((-1))
                frame_idx = past['frame_idx']
                camera_pose_mat = np.array([[0.5, 0., 0., 1.0],
                                            [2.0, 0.9, -0.9, 1.0],
                                            [2.0, 0.9, 0.9, 1.0],
                                            [2.0,-0.9,0.9, 1.0],
                                            [2.0,-0.9,-0.9, 1.0]], dtype=np.float32)
                camera_pose_mat_conv = np.matmul( diff, camera_pose_mat.T).T
                camera_pose_dict[ frame_idx ] = camera_pose_mat_conv


                concated_points[4 * start:4 * (start + past["points"].shape[0])] = tpoints
                concated_ori_points[4 * start:4*(start + past["points"].shape[0])] = past["points"].reshape((-1))
                concated_labels_gt[start:start + past["points"].shape[0]] = past["labels_gt"]
                concated_labels_pred[start:start + past["points"].shape[0]] = past["labels_pred"]
                concated_frame_ID[start:start + past['points'].shape[0]] = past['frame_ID']
                concated_loc_2D[2*start:2*(start + past['points'].shape[0])] = past['loc_2D'].reshape((-1))
                start += past["points"].shape[0]

            output_velo_scan_file_name = os.path.join(velo_scan_output_dir, 'velo_concat.bin')
            concated_points.tofile(output_velo_scan_file_name)

            output_velo_scan_ori_file_name = os.path.join(velo_scan_output_dir, 'velo_concat_ori.bin')
            concated_ori_points.tofile(output_velo_scan_ori_file_name)

            output_label_gt_file_name = os.path.join(gt_label_output_dir, 'label_gt_concat.label')
            concat_gt_label = np.squeeze(concated_labels_gt)
            concat_gt_label.tofile(output_label_gt_file_name)

            output_label_pred_file_name = os.path.join(pred_label_output_dir, 'label_pred_concat.label')
            concat_pred_label = np.squeeze(concated_labels_pred)
            concat_pred_label.tofile(output_label_pred_file_name)

            output_label_frame_ID_filename = os.path.join( frame_ID_output_dir, 'frame_ID_concat.label' )
            concated_frame_ID = np.squeeze( concated_frame_ID )
            concated_frame_ID.tofile( output_label_frame_ID_filename )

            output_loc_2D_filename = os.path.join( loc_2D_output_dir, 'loc_2D_concat.label' )
            concated_loc_2D.tofile( output_loc_2D_filename )

        # append current data to history queue.
        history.appendleft({
            "points": points,
            "labels_gt": gt_label,
            "labels_pred": pred_label,
            "remissions": remissions,
            "pose": pose.copy(),
            "frame_ID": frame_ID,
            "loc_2D": loc_2D,
            "frame_idx": pose_idx
        })

if __name__ == '__main__':
    ROOT_DIR = 'sequences'
    for i in range( 1 ):
        folder = '{:02d}'.format( i )
        print('processing folder: {}'.format( folder ))
        folder = os.path.join( ROOT_DIR, folder )
        process_one_folder_with_deque( folder )
    print('Done!')