import netCDF4 as nc
import numpy as np
from xml.dom import minidom
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
import pandas as pd

def readnc(nc_file):
    dataset = nc.Dataset(nc_file)
    # label_name_to_value = eval(dataset.getncattr("class"))

    samples = []
    for groupname,datagroup in dataset.groups.items():
        # for key,value in label_name_to_value.items():
            # if "background" not in key:
        for key in list(datagroup.variables.keys()):
                split = np.transpose(datagroup.variables[key][:])
                split = np.delete(split,[0,1],axis=1)
                if len(samples)==0:
                    samples=split
                else:
                    samples=np.vstack((samples,split))

    labels = samples[:,0]
    images = np.delete(samples,0,axis=1)
    return images,labels

def readxml(xml_path):
    with open(xml_path, 'r', encoding='utf8') as xml_file:
        dom = minidom.parse(xml_file)
        settings = dom.documentElement
        results = {}
        for setting in settings.childNodes:
            if setting.nodeType == minidom.Node.ELEMENT_NODE:
                results[setting.nodeName]=setting.childNodes[0].nodeValue
        return results

def rf_train(xml_path):
    settings = readxml(xml_path)
    bands = False
    for key in settings.keys():
        if "samples" == key:
            sample_file = settings[key]
        if "model_path" == key:
            model_path = settings[key]
        if "bands" == key:
            bands = str(settings[key]).split(",")
            bands_use = [int(x) for x in bands]
        if "estimator_num" == key:
            estimator_num = int(settings[key])
        if "job_num" == key:
            job_num = int(settings[key])

    print("reading sample test......")
    images,labels = readnc(sample_file)
    if not bands:
        bandnum = images.shape[1]
        bands_use = np.arange(0, bandnum).tolist()

    images_part = images[:, bands_use]

    print("training......")
    rf = RandomForestClassifier(n_estimators=estimator_num, n_jobs=job_num, oob_score=True)
    rf = rf.fit(images_part, labels)
    print("done.")

    joblib.dump(rf, model_path)

    print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))
    
    for b, imp in zip(bands_use, rf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))

    df = pd.DataFrame()
    df['truth'] = labels
    df['predict'] = rf.predict(images_part)
    # Cross-tabulate predictions
    print(pd.crosstab(df['truth'], df['predict'], margins=True))

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
    rf_train(xml)