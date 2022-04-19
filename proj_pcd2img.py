import os
import cv2
import numpy as np
import struct
import ctypes

def rgb_to_float(color):
    """ Converts an RGB list to the packed float format used by PCL
        From the PCL docs:
        "Due to historical reasons (PCL was first developed as a ROS package),
         the RGB information is packed into an integer and casted to a float"
        Args:
            color (list): 3-element list of integers [0-255,0-255,0-255]
        Returns:
            float_rgb: RGB value packed as a float
    """
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])

    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb

def float_to_rgb(float_rgb):
    """ Converts a packed float RGB format to an RGB list    
        
        Args:
            float_rgb: RGB value packed as a float
            
        Returns:
            color (list): 3-element list of integers [0-255,0-255,0-255]
    """
    s = struct.pack('>f', float_rgb)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value
			
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)
			
    color = [r,g,b]
			
    return color


def read_calib_file( abs_file_path ):
    """Read a calibration file and parse into a dictionary"""
    data_dict = dict()
    if not os.path.exists( abs_file_path ):
        print('input file does not exist: {}'.format( abs_file_path ) )
        raise ValueError('file does not exists: {}'.format( abs_file_path ) )
    with open( abs_file_path, 'r' ) as f:
        for line_tmp in f.readlines():
            line_tmp = line_tmp.rstrip('\n')
            key_tmp, val_tmp = line_tmp.split(':', 1)
            val_tmp = val_tmp.strip()
            try:
                data_dict[ str( key_tmp ) ] = np.array([float(x) for x in val_tmp.split(' ')])
            except ValueError:
                pass

    return data_dict

def load_calib( abs_file_path ):
    """"load and compute intrinsic and extrinsic calibration params"""
    data = dict()
    filedata = read_calib_file( abs_file_path )
    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P0'], (3, 4))
    P_rect_10 = np.reshape(filedata['P1'], (3, 4))
    P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    P_rect_30 = np.reshape(filedata['P3'], (3, 4))

    data['P_rect_00'] = P_rect_00
    data['P_rect_10'] = P_rect_10
    data['P_rect_20'] = P_rect_20
    data['P_rect_30'] = P_rect_30

    # Compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
    data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
    data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
    data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
    data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    # Compute the stereo baselines in meters by projecting the origin of
    # each camera frame into the velodyne frame and computing the distances
    # between them
    p_cam = np.array([0, 0, 0, 1])
    p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
    p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
    p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
    p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

    data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
    data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

    #self.calib = namedtuple('CalibData', data.keys())(*data.values())
    return data
    

def load_velodyne_scan( abs_file_path ):
    """Load and parse a velodye binary file"""
    if not os.path.exists( abs_file_path ):
        print('input file does not exist: {}'.format( abs_file_path ) )
        raise ValueError('file does not exist: {}'.format( abs_file_path ) )
    scan = np.fromfile( abs_file_path, dtype = np.float32 )
    scan = scan.reshape((-1, 4))

    return scan

def load_label( label_filename ):
    if not os.path.exists( label_filename ):
        print('label file does not exists: {}'.format( label_filename ) )
        raise ValueError('file does not exist: {}'.format( label_filename ) )
    label = np.fromfile( label_filename, dtype = np.int32 )
    label = label.reshape((-1))

    return label

def proj_pcd( calib_dict, point ):
    """project a 3D point to an image plane
    P: camera internal param matrix, 3x4
    Tr: camera external param matrix, 4x4
    the projection is achieved by P*Tr*point
    """
    point[:,-1] = 1
    #x = calib_dict['P_rect_30'].dot( calib_dict['T_cam3_velo'] ).dot( point.T )
    x = calib_dict['P_rect_20'].dot( calib_dict['T_cam3_velo'] ).dot( point.T )
    x = x.T
    divisor = np.tile( np.expand_dims( x[:,-1], -1 ), ( 1, x.shape[-1] ) )
    x /= divisor

    return x

def get_pcd_2D_loc( velo_scan, calib_file_name ):
    calib_dict = load_calib( calib_file_name )
    loc_2D = proj_pcd( calib_dict, velo_scan )

    loc_2D = np.rint( loc_2D )

    return loc_2D[:, 0:2]



#project a potential 3D point cloud to a 2D image plane by apply a couple of filters
def get_pcd_2D_loc_check_0405( velo_scan, calib_file_name, max_depth, img_h, img_w ):
    calib_dict = load_calib( calib_file_name )
    pcd_ID = np.arange( velo_scan.shape[0] ).astype( np.int32 )
    depth_val = np.rint( velo_scan[:,0] ).astype( np.int32 )

    pcd_ID_depth = np.concatenate( (pcd_ID.reshape((-1,1)), depth_val.reshape(-1,1)), axis = -1 )

    #filter out the 3D point cloud according to the depth value

    pcd_ID_depth = pcd_ID_depth[ velo_scan[:,0] <= max_depth ]
    velo_scan = velo_scan[ velo_scan[:,0] <= max_depth ]
    pcd_ID_depth = pcd_ID_depth[ velo_scan[:,0] > 0 ]
    velo_scan = velo_scan[ velo_scan[:,0] > 0 ]


    loc_2D = proj_pcd( calib_dict, velo_scan )
    loc_2D = loc_2D[:,0:2]
    loc_2D = np.rint( loc_2D ).astype( np.int32 )

    #filter out the 2D loc that exceeds the image plane.
    pcd_ID_depth = pcd_ID_depth[ loc_2D[:,0] > 0 ]
    loc_2D = loc_2D[ loc_2D[:,0 ] > 0 ]
    pcd_ID_depth = pcd_ID_depth[ loc_2D[:, 0] < img_w ]
    loc_2D = loc_2D[ loc_2D[:,0 ] < img_w ]
    pcd_ID_depth = pcd_ID_depth[loc_2D[:, 1] > 0 ]
    loc_2D = loc_2D[ loc_2D[:,1] > 0 ]
    pcd_ID_depth = pcd_ID_depth[loc_2D[:, 1] < img_h ]
    loc_2D = loc_2D[ loc_2D[:,1] < img_h ]

    valid = np.zeros( (loc_2D.shape[0]), np.int32 )

    depth_map_dict = dict()
    valid_row_list = list()
    for row_tmp in range( loc_2D.shape[0] ):
        depth_val_tmp = velo_scan[ row_tmp, 0 ]
        row_idx = loc_2D[row_tmp, 1]
        col_idx = loc_2D[row_tmp, 0]

        key = (row_idx, col_idx)
        if not key in depth_map_dict:
            depth_map_dict[ key ] = dict()
            val = velo_scan[ row_tmp, : ]
            depth_map_dict[ key ]['pcd'] = val
            depth_map_dict[ key ]['row_id'] = row_tmp
            valid_row_list.append( row_tmp )
        else:
            depth_val_old = depth_map_dict[ key ]['pcd'][0]
            if depth_val_tmp < depth_val_old:
                depth_map_dict[ key ]['pcd'] = velo_scan[ row_tmp, : ]
                old_row_id = depth_map_dict[ key ]['row_id']
                valid_row_list.remove( old_row_id )
                depth_map_dict[ key ]['row_id'] = row_tmp
                valid_row_list.append( row_tmp )

    valid_row = np.asarray( valid_row_list ).astype( np.int32 )
    valid[ valid_row ] = 1

    loc_2D = loc_2D[valid == 1]
    pcd_ID_depth = pcd_ID_depth[valid == 1 ]

    #concate to obtain [pcd_ID, depth, loc_2D[0], loc_2D[1]]
    pcd_ID_depth_loc_2D = np.concatenate( (pcd_ID_depth, loc_2D), axis = -1 )

    return pcd_ID_depth_loc_2D

def get_pcd_2D_loc_check( velo_scan, calib_file_name, max_depth ):
    calib_dict = load_calib( calib_file_name )
    loc_2D = proj_pcd( calib_dict, velo_scan )

    valid = np.zeros( (loc_2D.shape[0]), np.int32 )

    img_h, img_w = 376, 1241

    depth_map_dict = dict()
    valid_row_list = list()
    for row_tmp in range( velo_scan.shape[0] ):
        depth_val_tmp = velo_scan[ row_tmp, 0 ]
        if depth_val_tmp < 0:
            continue

        if depth_val_tmp > max_depth:
            continue

        row_idx = int( loc_2D[row_tmp, 1] + 0.5 )
        col_idx = int( loc_2D[row_tmp, 0] + 0.5 )

        if not ( row_idx >= 0 and row_idx < img_h and col_idx >= 0 and col_idx < img_w ):
            continue

        key = (row_idx, col_idx)
        if not key in depth_map_dict:
            depth_map_dict[ key ] = dict()
            val = velo_scan[ row_tmp, : ]
            depth_map_dict[ key ]['pcd'] = val
            depth_map_dict[ key ]['row_id'] = row_tmp
            valid_row_list.append( row_tmp )
        else:
            depth_val_old = depth_map_dict[ key ]['pcd'][0]
            if depth_val_tmp < depth_val_old:
                depth_map_dict[ key ]['pcd'] = velo_scan[ row_tmp, : ]
                old_row_id = depth_map_dict[ key ]['row_id']
                valid_row_list.remove( old_row_id )
                depth_map_dict[ key ]['row_id'] = row_tmp
                valid_row_list.append( row_tmp )

    valid_row = np.asarray( valid_row_list ).astype( np.int32 )
    valid[ valid_row ] = 1

    return loc_2D[:,0:2], valid

def get_pcd_2D_loc_check_new( velo_scan, calib_file_name ):
    calib_dict = load_calib( calib_file_name )
    loc_2D = proj_pcd( calib_dict, velo_scan )

    loc_2D = np.rint( loc_2D ).astype( np.int32 )

    img_h, img_w = 376, 1241

    loc_2D = loc_2D[ loc_2D[:,1] > 0 ]
    loc_2D = loc_2D[ loc_2D[:,1] < img_h ]
    loc_2D = loc_2D[ loc_2D[:,0] > 0 ]
    loc_2D = loc_2D[ loc_2D[:,0] < img_w ]

    return loc_2D[:,0:2]


def proj_pcd_2_image( veloscan_file_name, calib_file_name ):
    pcd = load_velodyne_scan( veloscan_file_name )
    calib_dict = load_calib( calib_file_name )

    x = proj_pcd( calib_dict, pcd )
    img_h, img_w = 376, 1241

    max_depth = 0
    depth_map = np.zeros( ( 376, 1241 ), dtype = np.uint8 )

    for row_tmp in range( pcd.shape[0] ):
        depth_val_tmp = int( pcd[row_tmp, 0] + 0.5 )
        if depth_val_tmp < 0:
            continue
        if depth_val_tmp > max_depth:
            max_depth = depth_val_tmp
        try:
            row_idx = int( x[row_tmp, 1] + 0.5 )
        except:
            import pdb
            pdb.set_trace()
        col_idx = int( x[row_tmp, 0] + 0.5 )

        if not ( row_idx >= 0 and row_idx < img_h and col_idx >= 0 and col_idx < img_w ):
            continue

        depth_map[ row_idx, col_idx ] = depth_val_tmp

    depth_map = depth_map * (255./max_depth )
    depth_map = np.asarray( depth_map, dtype = np.uint8 )

    return depth_map

def get_label_from_2d( scan, x, pred ):
    point_num = scan.shape[0]
    label = np.zeros( shape = [ point_num ], dtype = np.uint32 )
    img_h, img_w = 376, 1241

    for row_tmp in range( scan.shape[0] ):
        row_idx = int( x[row_tmp, 1] + 0.5 )
        col_idx = int( x[row_tmp, 0] + 0.5 )
        if not ( row_idx >= 0 and row_idx < img_h and col_idx >= 0 and col_idx < img_w ):
            continue
        label_tmp = pred[ row_idx, col_idx ]
        label[ row_tmp ] = label_tmp

    return label

def merge_depth_and_rgb( depth_map, rgb_img_name ):
    rgb_img = cv2.imread( rgb_img_name, 1 )
    if rgb_img is None:
        print('failed to load the image {}'.format( rgb_img_name ) )
        return depth_map
    depth_map = cv2.cvtColor( depth_map, cv2.COLOR_GRAY2BGR )
    """
    img_h, img_w, _ = rgb_img.shape
    for row_tmp in range( img_h ):
        for col_tmp in range( img_w ):
            if depth_map[ row_tmp, col_tmp ] == 0:
                continue
            depth_val = depth_map[ row_tmp, col_tmp ]
            depth_float = depth_val/255.
            (depth_r, depth_g, depth_b) = float_to_rgb( depth_float )
            rgb_img[ row_tmp, col_tmp, 0 ] += depth_b
            rgb_img[ row_tmp, col_tmp, 1 ] += depth_g
            rgb_img[ row_tmp, col_tmp, 2 ] += depth_r

    """
    rgb_img = cv2.add( rgb_img, depth_map )
    return rgb_img


def test():
    calib_file = '/home/yuhang/KITTI/odometry/dataset/sequences/00/calib.txt'
    velo_file = '/home/yuhang/KITTI/odometry/dataset/sequences/00/velodyne/000001.bin'
    label_file = '/home/yuhang/KITTI/odometry/dataset/sequences/00/labels/000001.label'
    img_name = '/home/yuhang/KITTI/odometry/dataset/sequences/00/image_3/000000.png'
    label = load_label( label_file )
    scan = load_velodyne_scan( velo_file )
    depth_map = proj_pcd_2_image( velo_file, calib_file )
    rgb_depth_img = merge_depth_and_rgb( depth_map, img_name )
    cv2.imwrite('rgb_depth_img1_2side_100_0.png', rgb_depth_img )
    print('Done!')

# test()
