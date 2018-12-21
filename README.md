### Readme File
https://docs.google.com/document/d/1ExURc_Yi76-FEBz1e7iuzYeHCG_ENSAjmml8wyb8le0/edit?usp=sharing

"# torch-phoc" 
1. go to `src/train.py` to run training code.
2. to change data please check `./src/dataset/maps_alt.py`
3. the path to data should be given in `line 140` of `train.py`
4. requires `tqdm`, `pytorch 0.4.0`, `python 2.7`
5. to create new dataset, run `make_dataset/make_data_for_test.py`. Please fix the path variables to direct to the correct files.
6. Check the padded and aspect options for generating the dataset.

Make sure: you have maps, alignment ids, annotations and the bounding boxes for the next four steps.

For generating regions and labels (before extension):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ change path files in make_testdataset/get_ims_and_labels.py

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ run run_get_ims_lab.sh as a job. (If needed parallelize this. Look at generating regions and labels after extension).

This additionally uses files from image_to_extend/ folder (like image_dir_<map_name>.npy and image_labels_<map_name>.npy)

For producing new bound boxes: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ change path files in src/extend_boundary.py

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ run extend_boundary.sh (This is pretty fast. No need of running in parallel).

For generating regions and labels (after extension):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ change path files in make_testdataset/generate_ims_labels.py

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ run run_generate_ims_lab_all.sh in the shell normally. It'll spawn multiple short jobs using un_generate_ims_lab.sh

To generate ground truth boxes and ground truth labels (This is not needed. Since it will already be provided)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ change path files in make_testdataset/get_truthbox_ims_and_labels.py
