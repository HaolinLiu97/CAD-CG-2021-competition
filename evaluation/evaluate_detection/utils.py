import numpy as np
import open3d as o3d

def quaternion_2rot(Q):
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

def quterion2euler(Q):
    q0=Q[0]
    q1=Q[1]
    q2=Q[2]
    q3=Q[3]

    roll=np.arctan2(2*(q0*q1+q2*q3),(1-2*(q1**2+q2**2)))
    yaw=np.arcsin(2*(q0*q2-q3*q1))
    pitch=np.arctan2(2*(q0*q3+q1*q2),(1-2*(q2**2+q3**2)))

    return [roll,pitch,yaw]

def euler2Quaterion(euler):
    yaw=euler[2]
    pitch=euler[1]
    roll=euler[0]
    cy=np.cos(yaw*0.5)
    sy=np.sin(yaw*0.5)
    cp=np.cos(pitch*0.5)
    sp=np.sin(pitch*0.5)
    cr=np.cos(roll*0.5)
    sr=np.sin(roll*0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    Q=[w,x,z,y]

    return Q
def euler2rot(euler):
    Q=euler2Quaterion(euler)
    rot=quaternion_2rot(Q)
    return rot

def get_bbox_verts(box_center,box_size,rot,color=(0,1,0)):
    s_x, s_y, s_z = box_size[0], box_size[1], box_size[2]
    verts = [[- s_x / 2, - s_y / 2, - s_z / 2],
             [- s_x / 2, - s_y / 2, s_z / 2],
             [- s_x / 2, s_y / 2, - s_z / 2],
             [- s_x / 2, s_y / 2, s_z / 2],
             [s_x / 2, - s_y / 2, - s_z / 2],
             [s_x / 2, - s_y / 2, s_z / 2],
             [s_x / 2, s_y / 2, - s_z / 2],
             [s_x / 2, s_y / 2,  s_z / 2]]
    verts=np.array(verts)
    verts=np.dot(rot,verts.T).T+box_center[np.newaxis,:]

    lines = [[0, 1], [0, 2], [0, 4], [1, 3],
             [1, 5], [2, 3], [2, 6], [3, 7],
             [4, 5], [4, 6], [5, 7], [6, 7]]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(verts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return verts,line_set