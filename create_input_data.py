import numpy as np
import os
from utils import load_config_file, load_dataset
import sys


if __name__ == '__main__':
    config_file_name = sys.argv[1]
    config = load_config_file(config_file_name)

    metadata_path = config["Metadata Path"]
    input_folder = config["Input Nifti Folder"]
    output_folder = config["Output Folder"]
    output_X = config["Output Feature File"]
    output_Y = config["Output Age File"]
    output_IDs = config["Output IDs File"]
    tissues_used = config["Tissues Used"]
    all_templates = config["All Templates"]

    # any additional templates to be concatenated have different suffixes
    if all_templates:
        templates = ['03motpl', '05motpl', '17motpl']

    # only GM
    if tissues_used == 'gm':
        images, ages, subjects = load_dataset(metadata_path, input_folder, tissue_suffix='_desc-modulated_01.nii.gz')
        if all_templates:
            for tpl in templates:
                curr_tissue_suffix = '_desc-modulated-' + tpl + '_01.nii.gz'
                images_tpl, _, _ = load_dataset(metadata_path, input_folder, tissue_suffix=curr_tissue_suffix)
                images = np.concatenate((images, images_tpl), axis=1)

    # only WM
    elif tissues_used == 'wm':
        images, ages, subjects = load_dataset(metadata_path, input_folder, tissue_suffix='_desc-modulated_02.nii.gz')
        if all_templates:
            for tpl in templates:
                curr_tissue_suffix = '_desc-modulated-' + tpl + '_02.nii.gz'
                images_tpl, _, _ = load_dataset(metadata_path, input_folder, tissue_suffix=curr_tissue_suffix)
                images = np.concatenate((images, images_tpl), axis=1)

    # GM + WM
    elif tissues_used == 'both':
        images_gm, _, _ = load_dataset(metadata_path, input_folder, tissue_suffix='_desc-modulated_01.nii.gz')
        images_wm, ages, subjects = load_dataset(metadata_path, input_folder, tissue_suffix='_desc-modulated_02.nii.gz')
        images = np.concatenate((images_gm, images_wm), axis=1)

        if all_templates:
            tissues = ['01', '02']
            for tpl in templates:
                for tissue in tissues:
                    curr_tissue_suffix = '_desc-modulated-' + tpl + '_' + tissue + '.nii.gz'
                    images_tpl, _, _ = load_dataset(metadata_path, input_folder, tissue_suffix=curr_tissue_suffix)
                    images = np.concatenate((images, images_tpl), axis=1)

    # GM + WM + CSF
    else:
        images_gm, _, _ = load_dataset(metadata_path, input_folder, tissue_suffix='_desc-modulated_01.nii.gz')
        images_wm, _, _ = load_dataset(metadata_path, input_folder, tissue_suffix='_desc-modulated_02.nii.gz')
        images_csf, ages, subjects = load_dataset(metadata_path, input_folder, tissue_suffix='_desc-modulated_03.nii.gz')
        images = np.concatenate((images_gm, images_wm, images_csf), axis=1)

        if all_templates:
            tissues = ['01', '02', '03']
            for tpl in templates:
                for tissue in tissues:
                    curr_tissue_suffix = '_desc-modulated-' + tpl + '_' + tissue + '.nii.gz'
                    images_tpl, _, _  = load_dataset(metadata_path, input_folder, tissue_suffix=curr_tissue_suffix)
                    images = np.concatenate((images, images_tpl), axis=1)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    np.save(os.path.join(output_folder, output_X), images)
    np.save(os.path.join(output_folder, output_Y), ages)
    np.save(os.path.join(output_folder, output_IDs), subjects)
