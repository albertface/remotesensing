import netCDF4 as nc
import numpy as np
from xml.dom import minidom
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
from osgeo import gdal, gdal_array
import os
import cv2
from collections import Counter
import operator
from xml.dom.minidom import Document

def readnc(nc_file):
    dataset = nc.Dataset(nc_file)
    label_name_to_value = eval(dataset.getncattr("class"))

    samples = []
    positions = []
    groupnames = []
    shapes = []
    for groupname,datagroup in dataset.groups.items():
        sample = []
        position = []
        for key,value in label_name_to_value.items():
            if "background" not in key:
                split = np.transpose(datagroup.variables[key][:])
                point = split[:,[0,1]]
                image = split[:,2:]
                if len(sample)==0:
                    sample=image
                else:
                    sample=np.vstack((sample,image))

                if len(position)==0:
                    position=point
                else:
                    position=np.vstack((position,point))
        shape = eval(datagroup.getncattr("shape"))
        shapes.append(shape)
        samples.append(sample)
        positions.append(position)
        groupnames.append(groupname)

    return samples,positions,label_name_to_value,shapes,groupnames

def rf_evaluate(xml_path):
    settings = readxml(xml_path)
    bands_use = False
    for key in settings.keys():
        if "model_path" == key:
            model_path = settings[key]
        if "samples" == key:
            nc_file = settings[key]
        if "eval_result_path" == key:
            eval_result_path = settings[key]

    rf = joblib.load(model_path)

    samples,positions,label_name_to_value,shapes,groupnames = readnc(nc_file)

    label_name_to_value.pop("_background_")
    label_value_to_name = {v: k for k, v in label_name_to_value.items()}

    wrong_all_class = {k: 0 for k, v in label_name_to_value.items()}
    classes_all_nums = {k: 0 for k, v in label_name_to_value.items()}
    wrong_percent_total = []
    for i in range(len(groupnames)):
        labels = samples[i][:,0]
        images = samples[i][:,1:]
        position = positions[i]
        shape = shapes[i]
        groupname = groupnames[i]
        # Now predict for each pixel
        class_prediction = rf.predict(images)

        jug_array = np.zeros([shape[0],shape[1]])

        #统计数量
        # a = dict(Counter(labels))
        # classes_nums = sorted(a.items(), key=operator.itemgetter(0), reverse=False)

        classes_nums = {k: 0 for k, v in label_name_to_value.items()}
        wrong_class = {k:0 for k,v in label_name_to_value.items()}
        for j in range(len(labels)):
            classes_nums[label_value_to_name[labels[j]]] +=1
            classes_all_nums[label_value_to_name[labels[j]]]+=1
            jug_array[position[j][0]][position[j][1]]=1
            if class_prediction[j] != labels[j]:
                wrong_class[label_value_to_name[labels[j]]]+=1
                wrong_all_class[label_value_to_name[labels[j]]]+=1
                jug_array[position[j][0]][position[j][1]] = 2
        wrong_percent = {k:v/c for k,v in wrong_class.items() for k,c in classes_nums.items()}
        wrong_percent_total.append(wrong_percent)
        jug_img_path = os.path.join(eval_result_path, groupname+"_eval.tif")
        cv2.imwrite(jug_img_path,jug_array)

    wrong_all_percent = {k:v/c for k,v in wrong_all_class.items() for k,c in classes_all_nums.items()}
    percent_xml_path = os.path.join(eval_result_path, "wrong_classify_percentage.xml")
    writexml(wrong_percent_total,wrong_all_percent,label_name_to_value,groupnames,percent_xml_path)
    return

def writexml(wrong_percent_total,wrong_all_percent,label_name_to_value,groupnames,xml_path):
    # label_value_to_name = {v: k for k, v in label_name_to_value.items()}
    doc = Document()

    percentage = doc.createElement('percentage')
    doc.appendChild(percentage)

    each = doc.createElement('each')
    percentage.appendChild(each)

    for i in range(len(groupnames)):
        each_image = doc.createElement(groupnames[i])
        wrong_percent = wrong_percent_total[i]
        for k,v in wrong_percent.items():
            each_class = doc.createElement(str(k))
            each_class_percent = doc.createTextNode(str(v))
            each_class.appendChild(each_class_percent)
            each_image.appendChild(each_class)
        each.appendChild(each_image)

    all = doc.createElement('all')
    percentage.appendChild(all)

    for k, v in wrong_all_percent.items():
        all_class = doc.createElement(str(k))
        all_class_percent = doc.createTextNode(str(v))
        all_class.appendChild(all_class_percent)
        all.appendChild(all_class)

    # 写入本地xml文件
    with open(xml_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return

def readxml(xml_path):
    with open(xml_path, 'r', encoding='utf8') as xml_file:
        dom = minidom.parse(xml_file)
        settings = dom.documentElement
        results = {}
        for setting in settings.childNodes:
            if setting.nodeType == minidom.Node.ELEMENT_NODE:
                results[setting.nodeName]=setting.childNodes[0].nodeValue
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--xml", help="Path to xml file", required=True)
    args = parser.parse_args()
    # return check_args(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    xml = args.xml
    rf_evaluate(xml)