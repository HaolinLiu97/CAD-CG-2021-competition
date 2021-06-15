import json
import cv2
import numpy as np
import open3d as o3d

def quaternion_rotation_matrix(Q,pos):
    # Extract the values from Q
    x = Q[0]
    y = Q[1]
    z = Q[2]
    w = Q[3]

    x2=x+x
    y2=y+y
    z2=z+z
    xx=x*x2
    xy=x*y2
    xz=x*z2
    yy=y*y2
    yz=y*z2
    zz=z*z2
    wx=w*x2
    wy=w*y2
    wz=w*z2

    # First row of the rotation matrix
    r00 = (1-(yy+zz))
    r01 = xy+wz
    r02 = xz-wy
    r03 = 0

    # Second row of the rotation matrix
    r10 = xy-wz
    r11 = (1-(xx+zz))
    r12 = yz+wx
    r13 = 0

    # Third row of the rotation matrix
    r20 = xz+wy
    r21 = yz-wx
    r22 = 1-(xx+yy)
    r23 = 0

    r30=pos[0]
    r31=pos[1]
    r32=pos[2]
    r33=1

    # 3x3 rotation matrix
    pos = np.array([[r00, r10, r20,r30],
                    [r01, r11, r21,r31],
                    [r02, r12, r22,r32],
                    [r03, r13, r23,r33]])
    return pos

def quaternion2rot(Q):
    # Extract the values from Q
    x = Q[0]
    y = Q[1]
    z = Q[2]
    w = Q[3]

    x2=x+x
    y2=y+y
    z2=z+z
    xx=x*x2
    xy=x*y2
    xz=x*z2
    yy=y*y2
    yz=y*z2
    zz=z*z2
    wx=w*x2
    wy=w*y2
    wz=w*z2

    # First row of the rotation matrix
    r00 = (1-(yy+zz))
    r01 = xy+wz
    r02 = xz-wy

    # Second row of the rotation matrix
    r10 = xy-wz
    r11 = (1-(xx+zz))
    r12 = yz+wx

    # Third row of the rotation matrix
    r20 = xz+wy
    r21 = yz-wx
    r22 = 1-(xx+yy)


    # 3x3 rotation matrix
    rot = np.array([[r00, r10, r20],
                    [r01, r11, r21],
                    [r02, r12, r22],])
    return rot

def reconstruct_pcd(color,depth,depth_intrinsic,cam_pose):
    '''
    :param color: rgb image
    :param depth: depth image in meter unit
    :param depth_intrinsic: intrinsic matrix
    :param cam_pose: camera to world extrinsic matrix
    :return:
    '''
    valid_Y, valid_X = np.where(depth > 0)
    unprojected_Y = valid_Y * depth[valid_Y, valid_X]
    unprojected_X = valid_X * depth[valid_Y, valid_X]
    unprojected_Z = depth[valid_Y, valid_X]
    point_cloud_xyz = np.concatenate(
        [unprojected_X[:, np.newaxis], unprojected_Y[:, np.newaxis], unprojected_Z[:, np.newaxis]], axis=1)
    intrinsic_inv = np.linalg.inv(depth_intrinsic)
    '''
    minus sign is needed to convert to camera coordinate
    '''
    point_cloud_xyz = -np.dot(intrinsic_inv, point_cloud_xyz.T).T
    project_on_color = np.dot(depth_intrinsic, point_cloud_xyz.T).T

    Y_project = project_on_color[:, 1] / project_on_color[:, 2]
    X_project = project_on_color[:, 0] / project_on_color[:, 2]
    X_project = X_project.astype(np.int)
    Y_project = Y_project.astype(np.int)
    X_project = np.clip(X_project, a_min=0, a_max=color.shape[1] - 1)
    Y_project = np.clip(Y_project, a_min=0, a_max=color.shape[0] - 1)
    point_cloud_colors = color[Y_project, X_project, ::-1]

    point_cloud_xyz = np.concatenate([point_cloud_xyz, np.ones([point_cloud_xyz.shape[0], 1])], axis=1)
    point_cloud_xyz = np.dot(cam_pose, point_cloud_xyz.T).T
    point_cloud = np.concatenate([point_cloud_xyz[:, 0:3], point_cloud_colors / 255.0],axis=1)
    return point_cloud

def get_bbox(center_pred,size_pred,rot_matrix,color=[1,0,0]):
    c_x,c_y,c_z=center_pred[0],center_pred[1],center_pred[2]
    s_x,s_y,s_z=size_pred[0],size_pred[1],size_pred[2]
    verts=[[c_x-s_x/2,c_y-s_y/2,c_z-s_z/2],
           [c_x-s_x/2,c_y-s_y/2,c_z+s_z/2],
           [c_x-s_x/2,c_y+s_y/2,c_z-s_z/2],
           [c_x-s_x/2,c_y+s_y/2,c_z+s_z/2],
           [c_x+s_x/2,c_y-s_y/2,c_z-s_z/2],
           [c_x+s_x/2,c_y-s_y/2,c_z+s_z/2],
           [c_x+s_x/2,c_y+s_y/2,c_z-s_z/2],
           [c_x+s_x/2,c_y+s_y/2,c_z+s_z/2]]

    verts=verts-center_pred[np.newaxis,:]
    verts=np.dot(rot_matrix,verts.T).T
    verts=verts+center_pred[np.newaxis,:]

    lines=[[0,1],[0,2],[0,4],[1,3],
           [1,5],[2,3],[2,6],[3,7],
           [4,5],[4,6],[5,7],[6,7]]
    colors=[color for i in range(len(lines))]
    line_set=o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(verts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors=o3d.utility.Vector3dVector(colors)
    return line_set

color=cv2.imread("color.jpg",cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
depth=cv2.imread("depth.png",cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
depth=depth[:,:,0]
depth=depth/255.0
depth=(1-depth)*10
with open('desc.json','r') as f:
    info=json.load(f)

bbox_infos=info['bbox_infos']
camera_K=bbox_infos['camera']['K']
translation=bbox_infos['camera']['pos']
Q=bbox_infos['camera']['rot']
wrd2cam_matrix=quaternion_rotation_matrix(Q,translation)
cam2wrd_matrix=np.linalg.inv(wrd2cam_matrix)
K=np.array(camera_K)
object_infos=bbox_infos['object_infos']
object=object_infos[3]
object_bbox=object["bbox"]
bbox_center=object_bbox["center"]
bbox_size=object_bbox["size"]
bbox_Q=object['6dpose']['rot']
rot=quaternion2rot(bbox_Q)
'''
project_center=np.dot(wrd2cam_matrix,np.array(bbox_center+[1]))
img_coor=np.dot(K,project_center[0:3])
project_x=int(img_coor[0]/img_coor[2])
project_y=int(img_coor[1]/img_coor[2])


inverse_K=np.linalg.inv(K)
xy_depth=depth[project_y,project_x]
img_center=np.array([project_x*xy_depth,project_y*xy_depth,xy_depth])
cam_center=-np.dot(inverse_K,np.array([project_x*xy_depth,project_y*xy_depth,xy_depth]))
recover_center=np.dot(cam2wrd_matrix,np.concatenate([cam_center,np.ones((1))],axis=0))
'''

point_cloud=reconstruct_pcd(color,depth,K,cam2wrd_matrix)
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(point_cloud[:,0:3])
pcd.colors=o3d.utility.Vector3dVector(point_cloud[:,3:6])

bbox_center=np.array(bbox_center)
#bbox_center[1]-=1
bbox_line=get_bbox(bbox_center,np.array(bbox_size),rot)

vis=o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.add_geometry(bbox_line)
opt = vis.get_render_option()
opt.show_coordinate_frame = True
vis.run()




