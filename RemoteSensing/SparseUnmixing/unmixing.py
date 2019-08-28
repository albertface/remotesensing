import numpy as np
import netCDF4 as nc
from osgeo import gdal, gdal_array
import methods
from xml.dom import minidom
from xml.dom.minidom import Document
import argparse
import os
import operator
import cv2

def readnc(nc_file):
    dataset = nc.Dataset(nc_file)
    label_name_to_value = eval(dataset.getncattr("class"))
    samples = []
    for groupname,datagroup in dataset.groups.items():
        for key in list(datagroup.variables.keys()):
                split = np.transpose(datagroup.variables[key][:].filled())
                split = np.delete(split,[0,1],axis=1)
                split = np.sum(split,axis=0)/(split.shape[0])
                if len(samples)==0:
                    samples=split
                else:
                    samples=np.vstack((samples,split))

    labels = samples[:,0]
    images = np.delete(samples,0,axis=1)
    return images,labels,label_name_to_value

def loaddata(imgpath,use_bands):
    gdal.UseExceptions()
    gdal.AllRegister()

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

    new_shape = (img.shape[0] * img.shape[1], img.shape[2])

    img_as_array = img.reshape(new_shape)
    img_shape = img[:, :, 0].shape

    return img_as_array,img_shape

def writetiff(img_path,image):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(img_path, image.shape[1], image.shape[0], image.shape[2], gdal.GDT_Float64)
    for i in range(image.shape[2]):
        band = dataset.GetRasterBand(i+1)
        band.WriteArray(image[:,:,i])
        # band.FlushCache()
    del dataset

def readxml(xml_path):
    with open(xml_path, 'r', encoding='utf8') as xml_file:
        dom = minidom.parse(xml_file)
        settings = dom.documentElement
        results = {}
        for setting in settings.childNodes:
            if setting.nodeType == minidom.Node.ELEMENT_NODE:
                results[setting.nodeName]=setting.childNodes[0].nodeValue
        return results

def writexml(label_name_to_value,xml_path,classes_percentage):
    label_value_to_name = {v: k for k, v in label_name_to_value.items()}

    doc = Document()
    classes = doc.createElement('classes')
    doc.appendChild(classes)

    discription = doc.createElement('classes_discription')
    classes.appendChild(discription)

    setting = doc.createElement(str(0))
    percent_text = doc.createTextNode('mask')
    setting.appendChild(percent_text)
    discription.appendChild(setting)
    for i in range(len(classes_percentage)):
        setting = doc.createElement(str(i+1))
        percent_text = doc.createTextNode(label_value_to_name[i+1])
        setting.appendChild(percent_text)
        discription.appendChild(setting)

    percentage = doc.createElement('classes_percentage')
    classes.appendChild(percentage)
    for i in range(len(classes_percentage)):
        setting = doc.createElement(label_value_to_name[i+1])
        percent_text = doc.createTextNode(str(classes_percentage[i]))
        setting.appendChild(percent_text)
        percentage.appendChild(setting)

    with open(xml_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return

def unmixing(xml_path):
    settings = readxml(xml_path)
    bands = False
    no_value_member = []
    for key in settings.keys():
        if "samples" == key:
            sample_file = settings[key]
        if "bands" == key:
            bands = str(settings[key]).split(",")
            bands_use = [int(x) for x in bands]
        if "image_dir" == key:
            image_dir = settings[key]
        if "unmixing_result_path" == key:
            result_path = settings[key]

    endmembers, labels,label_name_to_value = readnc(sample_file)
    for i in range(len(labels)):
        if labels[i]==-1:
            no_value_member = endmembers[0,:]
            endmembers = np.delete(endmembers,i,axis=0)

    if not bands:
        bandnum = endmembers.shape[1]
        bands_use = np.arange(0, bandnum).tolist()

    endmembers_part = endmembers[:, bands_use]

    if len(no_value_member)!=0:
        no_value_member_part = no_value_member[bands_use]

    print("unmixing......")
    image_list = os.listdir(image_dir)
    for i in range(0, len(image_list)):
        if ".hdr" not in image_list[i]:
            img_path = os.path.join(image_dir, image_list[i])
            img_as_array, img_shape = loaddata(img_path, bands_use)

            # if len(no_value_member)!=0:
            #     null_index = np.where((img_as_array == no_value_member_part).all(1))[0]
            #     value_index = np.where((img_as_array != no_value_member_part).all(1))[0]
            #     img_as_array_part = np.delete(img_as_array,null_index,axis=0)
            #     # Now unmixing each pixel
            #     x,res_p,res_d,ite = methods.sunsal(np.transpose(endmembers_part),np.transpose(img_as_array_part),positivity=True,addone=True)
            #     x1 = np.zeros((endmembers_part.shape[0],img_as_array.shape[0]))
            #     num=0
            #     for j in value_index:
            #         x1[:,j] = x[:,num]
            #         num+=1
            #     # no_value_result = np.zeros(endmembers.shape[0])
            #     # for i in index:
            #     #     x = np.insert(x,i,values=no_value_result,axis=1)
            #     abundance = np.transpose(x1).reshape((img_shape[0], img_shape[1], x1.shape[0]))
            # else:
            #     x, res_p, res_d, ite = methods.sunsal(np.transpose(endmembers_part), np.transpose(img_as_array),
            #                                           positivity=True, addone=True)
            #
            #     abundance = np.transpose(x).reshape((img_shape[0],img_shape[1],x.shape[0]))
            l = np.ones(endmembers.shape[0])
            x, res_p, res_d, ite = methods.sunsal(np.transpose(endmembers_part), np.transpose(img_as_array),lambda_0=l, positivity=True, addone=True)

            mask = np.ones(img_as_array.shape[0])
            if len(no_value_member) != 0:
                null_index = np.where((img_as_array == no_value_member_part).all(1))[0]
                no_value_result = np.zeros(endmembers.shape[0])
                for j in null_index:
                    x[:,j] = no_value_result
                    mask[j]=0
                x_part = np.delete(x, null_index, axis=1)
                percentage = np.sum(x_part, axis=1) / (x_part.shape[1])
            else:
                percentage = np.sum(x, axis=1) / (x.shape[1])
            x = np.vstack((mask,x))

            abundance = np.transpose(x).reshape((img_shape[0], img_shape[1], x.shape[0]))

            result_img_path = os.path.join(result_path, image_list[i].split(".")[0] + ".tif")
            if os.path.exists(result_img_path):
                os.remove(result_img_path)
            writetiff(result_img_path,abundance)

            writexml(label_name_to_value,result_img_path.replace("tif","xml"),percentage)

            print("Image {x} unmixing finish".format(x=image_list[i]))

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
    unmixing(xml)