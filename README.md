"# torch-phoc" 
1. go to `src/train.py` to run training code.
2. to change data please check `./src/dataset/maps_alt.py`
3. the path to data should be given in `line 140` of `train.py`
4. requires `tqdm`, `pytorch 0.4.0`, `python 2.7`
5. to create new dataset, run `make_dataset/make_data_for_test.py`. Please fix the path variables to direct to the correct files.
6. Check the padded and aspect options for generating the dataset.
