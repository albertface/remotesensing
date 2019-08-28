import netCDF4 as nc
import numpy as np
from xml.dom import minidom
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
from osgeo import gdal, gdal_array
import os
import cv2
from xml.dom.minidom import Document
from collections import Counter
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
import gc

def loaddata(imgpath,use_bands):
    gdal.UseExceptions()
    gdal.AllRegister()

    # for b in range(img.shape[2]):
    img_ds = gdal.Open(imgpath, gdal.GA_ReadOnly)
    if use_bands:
        img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, len(use_bands)),
                       gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
        usebandnum = 0
        for b in use_bands:
            img[:, :, usebandnum] = img_ds.GetRasterBand(b + 1).ReadAsArray()
            usebandnum += 1
    else:
        img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                       gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
        for b in range(img.shape[2]):
            img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    # Take our full image, ignore the Fmask band, and reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])

    img_as_array = img.reshape(new_shape)
    # print('Reshaped from {o} to {n}'.format(o=img.shape,
    #                                         n=img_as_array.shape))
    img_shape = img[:, :, 0].shape

    return img_as_array,img_shape

def readxml(xml_path):
    with open(xml_path, 'r', encoding='utf8') as xml_file:
        dom = minidom.parse(xml_file)
        settings = dom.documentElement
        results = {}
        for setting in settings.childNodes:
            if setting.nodeType == minidom.Node.ELEMENT_NODE:
                results[setting.nodeName]=setting.childNodes[0].nodeValue
        return results

def readcolorxml(xml_path):
    with open(xml_path, 'r', encoding='utf8') as xml_file:
        dom = minidom.parse(xml_file)
        colors = dom.documentElement
        results = {}
        for color in colors.childNodes:
            if color.nodeType == minidom.Node.ELEMENT_NODE:
                num = int(color.nodeName.replace("color",""))
                colorvalue = color.childNodes[0].nodeValue.split(",")
                colortuple = (int(colorvalue[0]),int(colorvalue[1]),int(colorvalue[2]),int(colorvalue[3]))
                results[num]=colortuple
        return results

def readnc(nc_file):
    dataset = nc.Dataset(nc_file)
    label_name_to_value = eval(dataset.getncattr("class"))
    return label_name_to_value

def saveShowPic(show_img_path,colors,class_prediction_reshape):
    shape = class_prediction_reshape.shape
    showpicture = np.zeros(shape=(shape[0],shape[1],4))
    for y in range(shape[0]):
        for x in range(shape[1]):
            color = colors[class_prediction_reshape[y][x]]
            showpicture[y][x][:]=[color[2],color[1],color[0],color[3]]
    cv2.imwrite(show_img_path,showpicture)

def saveLegend(filename,colors,label_value_to_name):
    handles = []
    try:
        handle = mpl.patches.Patch(color=(colors[-1][0] / 255, colors[-1][1] / 255, colors[-1][2] / 255, colors[-1][3] / 255),
                                   label=label_value_to_name[-1].replace("_", ""))
        handles.append(handle)
        for i in range(len(label_value_to_name)-1):
            i=i+1
            handle = mpl.patches.Patch(color=(colors[i][0]/255,colors[i][1]/255,colors[i][2]/255,colors[i][3]/255), label=label_value_to_name[i].replace("_",""))
            handles.append(handle)
    except:
        for i in range(len(label_value_to_name)):
            i=i+1
            handle = mpl.patches.Patch(color=(colors[i][0]/255,colors[i][1]/255,colors[i][2]/255,colors[i][3]/255), label=label_value_to_name[i].replace("_",""))
            handles.append(handle)
    legend = plt.legend(handles=handles, fontsize="xx-large", edgecolor=(1, 1, 1))
    fig = legend.figure

    fig.canvas.draw()
    #if got error,try this
    # renderer = fig.canvas.get_renderer()
    # fig.draw(renderer=renderer)

    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def rf_predict(xml_path):
    settings = readxml(xml_path)
    bands_use = False
    for key in settings.keys():
        if "root_dir" == key:
            root_dir = settings[key]
        if "model_path" == key:
            model_path = settings[key]
        if "bands" == key:
            bands = str(settings[key]).split(",")
            bands_use = [int(x) for x in bands]
        if "image_dir" == key:
            image_dir = settings[key]
        if "samples" == key:
            nc_file = settings[key]
        if "rf_result_path" == key:
            result_path = settings[key]
        if "color_table_path" == key:
            color_table_path = settings[key]

    colors = readcolorxml(color_table_path)

    label_name_to_value=readnc(nc_file)
    label_value_to_name = {v: k for k, v in label_name_to_value.items()}

    print("loading model......")
    rf = joblib.load(model_path)

    print("predicting......")
    image_list = os.listdir(image_dir)
    for i in range(0, len(image_list)):
        if ".hdr" not in image_list[i]:
            img_path = os.path.join(image_dir, image_list[i])
            img_as_array,img_shape = loaddata(img_path,bands_use)
            # Now predict for each pixel
            class_prediction = rf.predict(img_as_array)

            # Reshape our classification map
            class_prediction_reshape = class_prediction.reshape(img_shape)
            result_img_path = os.path.join(result_path, image_list[i].split(".")[0]+".tif")
            cv2.imwrite(result_img_path,class_prediction_reshape)

            saveShowPic(result_img_path.replace(".tif","_show.png"),colors,class_prediction_reshape)

            percentxml_path = result_img_path.replace("tif","xml")
            writexml(label_name_to_value,percentxml_path,class_prediction)
            print("Image {x} classification finish".format(x=image_list[i]))
            # del(img_as_array)
            # del(class_prediction_reshape)
            # del(class_prediction)
            # gc.collect()

    saveLegend(root_dir+"\\legend.png",colors,label_value_to_name)

    return

def writexml(label_name_to_value,xml_path,class_prediction):
    label_value_to_name = {v: k for k, v in label_name_to_value.items()}

    doc = Document()
    percentage = doc.createElement('classes_percentage')
    doc.appendChild(percentage)

    a = dict(Counter(list(class_prediction)))
    #进行排序
    b= sorted(a.items(), key=operator.itemgetter(0),reverse=False)

    for i in range(len(b)):
        percent = b[i][1]/len(class_prediction)
        setting = doc.createElement(label_value_to_name[b[i][0]])
        percent_text = doc.createTextNode(str(percent))
        setting.appendChild(percent_text)
        percentage.appendChild(setting)

    with open(xml_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--xml" , help="Path to xml file", required=True)
    args = parser.parse_args()
    # return check_args(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    xml = args.xml
    rf_predict(xml)
