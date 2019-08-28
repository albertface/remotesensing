from xml.dom.minidom import Document
import argparse
import os

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-j", "--image_json_dir", help="Path to image&json dir", required=True)
#     parser.add_argument("-i", "--image_dir", help="Path to image dir", required=True)
#     parser.add_argument("-s", "--sample" , help="Path to sample netcdf dir", required=True)
#     # parser.add_argument("-c", "--classes", help="Sample classes", required=True)
#     parser.add_argument("-b", "--bands", help="Bands to use", required=False)
#     parser.add_argument("-m", "--model_path" , help="Path to trained model", required=True)
#     parser.add_argument("-r", "--result_path", help="Path to classification results", required=True)
#     parser.add_argument("-er", "--eval_result_path", help="Path to evaluation results", required=True)
#     parser.add_argument("-e", "--estimator_num", help="Set RF estimator_num", required=False)
#     parser.add_argument("-n", "--job_num", help="Set RF job_num", required=False)
#     parser.add_argument("-x", "--xml_path", help="Path to save output xml file", required=True)
#     args = parser.parse_args()
#     # return check_args(args)
#     return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--root_dir", help="Path to root dir", required=True)
    args = parser.parse_args()
    return args

# def check_args(args):
#     if not os.path.exists(args.samples):
#         raise ValueError("Sample file does not exist")
#     if not os.path.exists(args.outputxml):
#         raise ValueError("Wrong bands format")
#     if not os.path.exists(args.bands):
#         raise ValueError("Wrong bands format")
#     return args

def writeInfoToXml(root_dir,sample,image_json_dir,image_dir,model_path,rf_result_path,unmixing_result_path,eval_result_path,xml_path,color_table_path,model_settings_path,no_value_name,bands=False,estimator_num=False,job_num=False):
    doc = Document()

    settings = doc.createElement('settings')
    doc.appendChild(settings)

    setting11 = doc.createElement('root_dir')
    root_path = doc.createTextNode(root_dir)
    setting11.appendChild(root_path)
    settings.appendChild(setting11)

    setting5 = doc.createElement('image_json_dir')
    json_path = doc.createTextNode(image_json_dir)
    setting5.appendChild(json_path)
    settings.appendChild(setting5)

    setting6 = doc.createElement('image_dir')
    image_path = doc.createTextNode(image_dir)
    setting6.appendChild(image_path)
    settings.appendChild(setting6)

    setting1 = doc.createElement('samples')
    sample_path = doc.createTextNode(sample)
    setting1.appendChild(sample_path)
    settings.appendChild(setting1)

    # setting9 = doc.createElement('classes')
    # classes_use = doc.createTextNode(classes)
    # setting9.appendChild(classes_use)
    # settings.appendChild(setting9)


    if not estimator_num:
        estimator_num=500
    setting3 = doc.createElement('estimator_num')
    estimator_num_use = doc.createTextNode(str(estimator_num))
    setting3.appendChild(estimator_num_use)
    settings.appendChild(setting3)

    if not job_num:
        job_num=1
    setting4 = doc.createElement('job_num')
    jobnum_use = doc.createTextNode(str(job_num))
    setting4.appendChild(jobnum_use)
    settings.appendChild(setting4)

    setting7 = doc.createElement('model_path')
    model_1_path = doc.createTextNode(model_path)
    setting7.appendChild(model_1_path)
    settings.appendChild(setting7)

    setting8 = doc.createElement('rf_result_path')
    rf_result_1_path = doc.createTextNode(rf_result_path)
    setting8.appendChild(rf_result_1_path)
    settings.appendChild(setting8)

    setting14 = doc.createElement('unmixing_result_path')
    unmixing_result_1_path = doc.createTextNode(unmixing_result_path)
    setting14.appendChild(unmixing_result_1_path)
    settings.appendChild(setting14)


    setting9 = doc.createElement('eval_result_path')
    eval_result_1_path = doc.createTextNode(eval_result_path)
    setting9.appendChild(eval_result_1_path)
    settings.appendChild(setting9)

    setting10 = doc.createElement('color_table_path')
    color_table_1_path = doc.createTextNode(color_table_path)
    setting10.appendChild(color_table_1_path)
    settings.appendChild(setting10)

    setting13 = doc.createElement('model_settings_path')
    model_settings_path_1 = doc.createTextNode(model_settings_path)
    setting13.appendChild(model_settings_path_1)
    settings.appendChild(setting13)

    if bands:
        setting2 = doc.createElement('bands')
        bands_use = doc.createTextNode(bands)
        setting2.appendChild(bands_use)
        settings.appendChild(setting2)

    setting12 = doc.createElement('no_value_name')
    no_value_name_1 = doc.createTextNode(no_value_name)
    setting12.appendChild(no_value_name_1)
    settings.appendChild(setting12)

    # 写入本地xml文件
    with open(xml_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return

def writeColorsToXml(input,color_table_path):
    doc = Document()

    colors = doc.createElement('colors')
    doc.appendChild(colors)

    for key,value in input.items():
        color = doc.createElement("color"+str(key))
        colorstring = str(value)
        colorstring = colorstring[1:len(colorstring)-1]
        color_1 = doc.createTextNode(colorstring)
        color.appendChild(color_1)
        colors.appendChild(color)

    with open(color_table_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return

if __name__ == '__main__':
    args = parse_args()
    root_dir = args.root_dir
    # image_json_dir = args.image_json_dir
    # image_dir = args.image_dir
    # sample = args.sample
    # bands = args.bands
    # estimator_num = args.estimator_num
    # job_num = args.job_num
    # model_path = args.model_path
    # xml_path = args.xml_path
    # result_path = args.result_path
    # eval_result_path = args.eval_result_path
    # classes = args.classes

    image_json_dir = root_dir+"\\image_json"
    image_dir = root_dir+"\\image"
    samples = root_dir+"\\netcdf\\samples.nc"
    model_path = root_dir+"\\model\\train.m"
    xml_path = root_dir+"\\settings.xml"
    rf_result_path = root_dir+"\\rf_result"
    unmixing_result_path = root_dir + "\\unmixing_result"
    eval_result_path = root_dir+"\\eval_result"
    color_table_path = root_dir+"\\colors.xml"
    model_settings_path = root_dir +"\\model_settings.xml"
    bands = False
    no_value_name = "_null_"
    estimator_num = 500
    job_num = 8

    folder = os.path.exists(image_json_dir)
    if not folder:
        os.makedirs(image_json_dir)

    folder = os.path.exists(image_dir)
    if not folder:
        os.makedirs(image_dir)

    folder = os.path.exists(root_dir+"\\netcdf")
    if not folder:
        os.makedirs(root_dir+"\\netcdf")

    folder = os.path.exists(root_dir+"\\model")
    if not folder:
        os.makedirs(root_dir+"\\model")

    folder = os.path.exists(rf_result_path)
    if not folder:
        os.makedirs(rf_result_path)

    folder = os.path.exists(unmixing_result_path)
    if not folder:
        os.makedirs(unmixing_result_path)

    folder = os.path.exists(eval_result_path)
    if not folder:
        os.makedirs(eval_result_path)

    writeInfoToXml(root_dir,samples,image_json_dir,image_dir,model_path,rf_result_path,unmixing_result_path,eval_result_path,xml_path,color_table_path,model_settings_path,no_value_name,bands,estimator_num,job_num)

    colors = dict((
        (-1, (0, 0, 0, 255)),  # Nodata
        (1, (255, 255, 0, 255)),
        (2, (0, 255, 255, 255)),
        (3, (0, 255, 0, 255)),
        (4, (46, 139, 87, 255)),
        (5, (176, 48, 96, 255)),
        (6, (127, 180, 235, 255)),
        (7, (255, 0, 255, 255)),
        (8, (160, 32, 240, 255)),
        (9, (160, 82, 45, 255)),
        (10, (218, 112, 214, 255)),
        (11, (127, 255, 212, 255)),
        (12, (216, 191, 216, 255)),
        (13, (255, 127, 80, 255)),
        (14, (160, 255, 45, 255)),
        (15, (80, 225, 132, 255))
    ))

    writeColorsToXml(colors,color_table_path)



    # writeModelSettingsXml(model_settings_path,)

