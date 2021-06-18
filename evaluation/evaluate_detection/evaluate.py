# coding=utf-8
import json
import sys
import os
import numpy as np
import time
from utils import quaternion_2rot,euler2rot, get_bbox_verts
import open3d as o3d

# 错误字典，这里只是示例
error_msg={
    1: "Bad input file",
    2: "Wrong input file format",
}

def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file,indent=1)

def parse_gt_result(gt_result):
    gt_dict={}
    length=len(gt_result)
    for i in range(length):
        image_id=gt_result[i]['img_id']
        object_info=gt_result[i]["bbox_infos"]["object_infos"]
        for j in range(len(object_info)):
            bbox_center=object_info[j]["bbox"]["center"]
            bbox_size=object_info[j]["bbox"]["size"]
            bbox_Q=object_info[j]["6dpose"]["rot"]
            label=object_info[j]["label"]
            rot=quaternion_2rot(bbox_Q)
            bbox=bbox_center+bbox_size+list(np.reshape(rot,[9]))
            if label not in gt_dict:
                gt_dict[label]={}
            if image_id not in gt_dict[label]:
                gt_dict[label][image_id]=[]
            gt_dict[label][image_id].append(bbox)
    return gt_dict

def parse_user_result(user_result):
    user_dict={}
    length=len(user_result)
    for i in range(length):
        image_id=user_result[i]['img_id']
        object_info = user_result[i]["object_infos"]
        for j in range(len(object_info)):
            #print(object_info[i])
            bbox_center = object_info[j]["bbox"]["center"]
            bbox_size = object_info[j]["bbox"]["size"]
            euler = object_info[j]["bbox"]["rot"]
            score = object_info[j]["objectness"]
            pred_class=object_info[j]["pred_cls"]
            rot=euler2rot(euler)
            bbox = bbox_center + bbox_size + list(np.reshape(rot,[9])) + [score]
            if pred_class not in user_dict:
                user_dict[pred_class]={}
            if image_id not in user_dict[pred_class]:
                user_dict[pred_class][image_id]=[]
            user_dict[pred_class][image_id].append(bbox)
    return user_dict

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_cls(user_dict_cls,gt_dict_cls,ovthresh=0.5):

    npos=0
    class_recs={}
    for image_id in gt_dict_cls.keys():
        bbox=np.array(gt_dict_cls[image_id])
        det=[False]*len(bbox)
        npos +=len(bbox)
        class_recs[image_id]={"bbox":bbox,"det":det}
    for image_id in user_dict_cls.keys():
        if image_id not in gt_dict_cls:
            class_recs[image_id] = {'bbox': np.array([]), 'det': []}

    image_ids=[]
    objectness=[]
    BB=[]
    for image_id in user_dict_cls.keys():
        for i in range(len(user_dict_cls[image_id])):
            BB.append(user_dict_cls[image_id][i][0:-1])
            image_ids.append(image_id)
            objectness.append(user_dict_cls[image_id][i][-1])
    objectness=np.array(objectness)
    BB=np.array(BB)

    sorted_ind=np.argsort(-objectness)
    BB=BB[sorted_ind,...]
    image_ids=[image_ids[x] for x in sorted_ind]
    nd=len(image_ids)
    tp=np.zeros(nd)
    fp=np.zeros(nd)

    for i in range(nd):
        R=class_recs[image_ids[i]]
        bb=BB[i,...].astype(float)
        X=np.linspace(-bb[3]/2,bb[3]/2,12,endpoint=True)[1:11]
        Y=np.linspace(-bb[4]/2,bb[4]/2,12,endpoint=True)[1:11]
        Z=np.linspace(- bb[5] / 2, bb[5] / 2, 12, endpoint=True)[1:11]
        X,Y,Z=np.meshgrid(X,Y,Z)
        sample_point=np.stack([X,Y,Z],axis=-1)
        sample_point=np.reshape(sample_point,[10*10*10,3])
        user_rot=np.reshape(bb[6:15],(3,3))
        user_center=bb[0:3]
        user_bbox_vol=bb[3]*bb[4]*bb[5]
        ovmax=-np.inf

        BBGT=R['bbox'].astype(float)
        if BBGT.size>0:
            for j in range(BBGT.shape[0]):
                gt_rot = np.reshape(BBGT[j, -9:], (3, 3))
                gt_center = BBGT[j, 0:3]
                gt_bbox_vol = BBGT[j, 3] * BBGT[j, 4] * BBGT[j, 5]
                relative_rot = np.dot(np.linalg.inv(gt_rot), user_rot)
                relative_trans = user_center - gt_center
                sample_point_in_gt_cano = np.dot(relative_rot, sample_point.T).T + relative_trans[np.newaxis, :]
                inside_gt_bbox = (sample_point_in_gt_cano[:, 0] <= BBGT[j, 3] / 2) & (
                            sample_point_in_gt_cano[:, 0] >= -BBGT[j, 3] / 2) \
                                 & ((sample_point_in_gt_cano[:, 1] <= BBGT[j, 4] / 2)) & (
                                             sample_point_in_gt_cano[:, 1] >= -BBGT[j, 4] / 2) \
                                 & ((sample_point_in_gt_cano[:, 2] <= BBGT[j, 5] / 2)) & (
                                             sample_point_in_gt_cano[:, 2] >= -BBGT[j, 5] / 2)
                num_inside = np.sum(inside_gt_bbox)
                inter_vol = num_inside / 1000 * user_bbox_vol
                IoU = inter_vol / (user_bbox_vol + gt_bbox_vol - inter_vol)


                if IoU>ovmax:
                    ovmax=IoU
                    jmax=j
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[i]=1
                R['det'][jmax]=1
            else:
                fp[i]=1
        else:
            fp[i]=1
    fp=np.cumsum(fp)
    tp=np.cumsum(tp)
    rec=tp/npos

    prec=tp/np.maximum(tp+fp,np.finfo(np.float64).eps)
    ap=voc_ap(rec,prec,True)
    return ap

def report_error_msg(detail, showMsg, out_p):
    error_dict=dict()
    error_dict['errorDetail']=detail
    error_dict['errorMsg']=showMsg
    error_dict['score']=0
    error_dict['scoreJson']={}
    error_dict['success']=False
    dump_2_json(error_dict,out_p)

def report_score(score,score_dict, out_p):
    result = dict()
    result['success']=True
    result['score'] = score

    # 这里{}里面的score注意保留，但可以增加其他key，比如这样：
    # result['scoreJson'] = {'score': score, 'aaaa': 0.1}

    result['class ap'] = score_dict

    dump_2_json(result,out_p)

if __name__=="__main__":
    '''
      online evaluation
    '''
    in_param_path = sys.argv[1]
    out_path = sys.argv[2]

    try:
        # read submit and answer file from first parameter
        with open(in_param_path, 'r') as load_f:
            input_params = json.load(load_f)

        # 标准答案路径
        standard_path=input_params["fileData"]["standardFilePath"]
        print("Read standard from %s" % standard_path)

        # 选手提交的结果文件路径
        submit_path=input_params["fileData"]["userFilePath"]
        print("Read user submit file from %s" % submit_path)

        with open(submit_path,'r') as f:
            user_result=json.load(f)
        with open(standard_path,'r') as f:
            standard_result=json.load(f)

        user_dict=parse_user_result(user_result)
        gt_dict=parse_gt_result(standard_result)

        class_list=set(user_dict.keys()).intersection(set(gt_dict.keys()))
        mAP=0
        score_dict={}
        for i,class_name in enumerate(class_list):
            ap=evaluate_cls(user_dict[class_name],gt_dict[class_name],ovthresh=0.25)
            print("ap of class %s is %f"%(class_name,ap))
            score_dict[class_name]=ap
            mAP+=ap
            # NOTICE: 这个是示例
        mAP=mAP/(i+1)
        score = mAP
        report_score(score,score_dict, out_path)
    except Exception as e:
        # NOTICE: 这个只是示例
        check_code = 1
        report_error_msg(error_msg[check_code],error_msg[check_code], out_path)
