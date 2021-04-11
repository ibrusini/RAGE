import os
import numpy as np
import json
import xlrd
import nibabel as nib


def load_config_file(config_file):
    with open(config_file) as data:
        config = json.loads(data.read())
    return config


def load_dataset(metadata_path, nifti_path, tissue_suffix):
    list_age = []
    list_images = []
    list_subj = []
    nifti_path = os.path.join(nifti_path, '')

    wb = xlrd.open_workbook(metadata_path)
    sheet = wb.sheet_by_index(0)
    count = 0

    for i in range(1, sheet.nrows):
        curr_age = np.float(sheet.cell_value(i, 4))

        name_column = sheet.cell_value(0, 4)
        if 'weeks' not in name_column:
            curr_age = round(curr_age / 7)

        curr_subj = sheet.cell_value(i, 1)
        if sheet.cell_value(i, 2) == '':
            curr_subj_stem = curr_subj
        else:
            curr_subj_stem = sheet.cell_value(i, 1) + '_ses-' + str(int(sheet.cell_value(i, 2)))

        curr_subj_path = nifti_path + curr_subj_stem + tissue_suffix

        try:
            curr_image_object = nib.load(curr_subj_path)
            curr_data = curr_image_object.get_data()
            curr_data_reshaped = curr_data.reshape(np.prod(curr_data.shape), )
            list_images.append(curr_data_reshaped)
            list_age.append([np.float(curr_age)])
            list_subj.append(curr_subj)

        except:
            if count == 0:
                print('Samples not included in the dataset because image files were not found:')
            count += 1
            print(curr_subj_stem)

    age = np.stack(list_age)
    images = np.stack(list_images)

    return images, age, list_subj