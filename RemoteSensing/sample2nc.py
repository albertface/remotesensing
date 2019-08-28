import json
import os
from osgeo import gdal, gdal_array
import numpy as np
from labelme import utils
import netCDF4 as nc
from xml.dom import minidom
import argparse

def sample2nc(json_dir,nullname,nc_file,use_bands=False):
    gdal.UseExceptions()
    gdal.AllRegister()

    # label_name_to_value = {'_background_': 0}
    label_name_to_value = {}
    samples = {}
    shapes = {}
    num = 1

    if not nc_file:
        nc_file=os.path.join(json_dir, "samples.nc")
    dataset = nc.Dataset(nc_file, 'w', format='NETCDF4')

    print("reading json test......")
    list_file = os.listdir(json_dir)
    for i in range(0, len(list_file)):
        if ".json" in list_file[i]:
            tifpath = os.path.join(json_dir, list_file[i].replace("json","tif"))
            geotifpath = os.path.join(json_dir, list_file[i].replace("json","geotiff"))
            imgpath = False
            if os.path.exists(tifpath):
                imgpath = tifpath
            if os.path.exists(geotifpath):
                imgpath = geotifpath

            if not imgpath:
                print("Can't find sample image")
            jsonpath = os.path.join(json_dir, list_file[i])
            jsondata = json.load(open(jsonpath))

            #读取图像
            img_ds = gdal.Open(imgpath,gdal.GA_ReadOnly)
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

            # 读取图像对应的样本
            for shape in sorted(jsondata['shapes'], key=lambda x: x['label']):
                label_name = shape['label']
                # if label_name in label_name_to_value:
                #     label_value = label_name_to_value[label_name]
                # else:
                if label_name not in label_name_to_value:
                    if nullname == label_name:
                        label_value=-1
                    else:
                        label_value = num
                        num = num+1
                    label_name_to_value[label_name] = label_value
            lbl = utils.shapes_to_label([img.shape[0],img.shape[1]], jsondata['shapes'], label_name_to_value)

            [x,y]=np.where(lbl)
            image = img[lbl != 0, :]
            label = lbl[lbl != 0]

            sample = np.column_stack((x,y,label,image))
            samples[list_file[i].replace(".json","")]=sample
            shapes[list_file[i].replace(".json", "")] = img.shape

    # print('We have {n} samples'.format(n=len(label_name_to_value)))
    # print('Our imagedata matrix is sized: {sz}'.format(sz=X.shape))
    # print('Our label array is sized: {sz}'.format(sz=y.shape))
    # samples = samples[samples[:,2].argsort()]

    print("writing netcdf file......")
    #写入nc文件
    label_value_to_name = {v : k for k, v in label_name_to_value.items()}
    dataset.setncattr_string("class",str(label_name_to_value))
    for key in samples.keys():
        datagroup = dataset.createGroup(key)
        datagroup.setncattr_string("shape", str(shapes[key]))

        # sample = samples[key]
        # split = np.zeros((1,sample.shape[1]))
        # x_dim = label_value_to_name[0] + '_x'
        # y_dim = label_value_to_name[0] + '_y'
        # datagroup.createDimension(x_dim, sample.shape[1])  # 创建坐标点
        # datagroup.createDimension(y_dim, 1)  # 创建坐标点
        # datagroup.createVariable(label_value_to_name[0], 'i8', (x_dim, y_dim))
        # datagroup.variables[label_value_to_name[0]][:] = split

        sample = samples[key]
        split = sample[sample[:, 2] == -1, :]
        specialnum = 0
        if len(split) != 0:
            specialnum = 1
            x_dim = label_value_to_name[-1] + '_x'
            y_dim = label_value_to_name[-1] + '_y'
            datagroup.createDimension(x_dim, split.shape[1])  # 创建坐标点
            datagroup.createDimension(y_dim, split.shape[0])  # 创建坐标点
            datagroup.createVariable(label_value_to_name[-1], 'i8', (x_dim, y_dim))
            datagroup.variables[label_value_to_name[-1]][:] = split.transpose()

        for i in range(len(label_name_to_value)-specialnum):
            sample = samples[key]
            split = sample[sample[:,2]==i+1,:]
            if len(split) != 0:
                x_dim = label_value_to_name[i+1]+'_x'
                y_dim = label_value_to_name[i+1]+'_y'
                datagroup.createDimension(x_dim, split.shape[1])  # 创建坐标点
                datagroup.createDimension(y_dim, split.shape[0])  # 创建坐标点
                datagroup.createVariable(label_value_to_name[i+1], 'i8', (x_dim,y_dim))
                datagroup.variables[label_value_to_name[i+1]][:] = split.transpose()
    dataset.close()

    print("writing netcdf file done")
    return None

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
    parser.add_argument("-x", "--xml" , help="Path to xml file", required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    xml = args.xml
    settings = readxml(xml)
    bands = False
    for key in settings.keys():
        if "image_json_dir" == key:
            json_dir = settings[key]
        if "no_value_name" == key:
            nullname = settings[key]
        if "samples" == key:
            sample = settings[key]
        if "bands" == key:
            bands = settings[key]
    sample2nc(json_dir,nullname,sample,bands)