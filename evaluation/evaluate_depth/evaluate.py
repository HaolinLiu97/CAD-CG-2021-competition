# coding=utf-8
import json
import sys
import os
import cv2
import numpy as np

# 错误字典，这里只是示例
error_msg={
    1: "Bad input file",
    2: "Wrong input file format",
}

def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file,indent=1)

def report_error_msg(detail, showMsg, out_p):
    error_dict=dict()
    error_dict['errorDetail']=detail
    error_dict['errorMsg']=showMsg
    error_dict['score']=0
    error_dict['scoreJson']={}
    error_dict['success']=False
    dump_2_json(error_dict,out_p)

def report_score(score, out_p):
    result = dict()
    result['success']=True
    result['rmse'] = score

    # 这里{}里面的score注意保留，但可以增加其他key，比如这样：
    # result['scoreJson'] = {'score': score, 'aaaa': 0.1}

    dump_2_json(result,out_p)

def compute_rmse(result_image,gt_image):
    error=(result_image-gt_image)**2
    mse=np.mean(error)
    rmse=np.sqrt(mse)
    return rmse

def evaluate_rmse(submit_path,standard_path):
    image_list=os.listdir(standard_path)
    total_mmse=0
    total_count=0
    for image_name in image_list:
        gt_path=os.path.join(standard_path,image_name,"depth.png")
        print(gt_path)
        gt_image=cv2.imread(gt_path,cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
        gt_image=(1-gt_image/255.0)*10
        result_path=os.path.join(submit_path,image_name,"depth.exr")
        result_image=cv2.imread(result_path,cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
        mmse=compute_rmse(result_image,gt_image)
        total_count+=1
        total_mmse+=mmse
    mean_mmse=total_mmse/total_count
    return mean_mmse


if __name__=="__main__":
    '''
      online evaluation
    '''
    in_param_path = sys.argv[1]
    out_path = sys.argv[2]

    # read submit and answer file from first parameter
    with open(in_param_path, 'r') as load_f:
        input_params = json.load(load_f)

    # 标准答案路径
    standard_path=input_params["fileData"]["standardFileDir"]

    # 选手提交的结果文件路径
    submit_path=input_params["fileData"]["userFileDir"]

    mean_rmse=evaluate_rmse(submit_path,standard_path)
    report_score(mean_rmse,out_path)