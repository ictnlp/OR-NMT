# wenmt

### Runtime Environment
This system has been tested in the following environment.
+ 64bit-Ubuntu
+ Python 2.7
+ \> Pytorch 0.4

### Data Preparation
Name the file names of the datasets according to the variables in the ``wargs.py`` file  

#### Training Dataset

+ **Source side**: ``dir_data + train_prefix + '.' + train_src_suffix``  
+ **Target side**: ``dir_data + train_prefix + '.' + train_trg_suffix``  

#### Validation Set

+ **Source side**: ``val_tst_dir + val_prefix + '.' + val_src_suffix``    
+ **Target side**:  
	+ One reference  
``val_tst_dir + val_prefix + '.' + val_ref_suffix``  
	+ multiple references  
``val_tst_dir + val_prefix + '.' + val_ref_suffix + '0'``  
``val_tst_dir + val_prefix + '.' + val_ref_suffix + '1'``  
``......``

#### Test Dataset
+ **Source side**: ``val_tst_dir + test_prefix + '.' + val_src_suffix``  
+ **Target side**:  
``for test_prefix in tests_prefix``
	+ One reference  
``val_tst_dir + test_prefix + '.' + val_ref_suffix``  
	+ multiple references  
``val_tst_dir + test_prefix + '.' + val_ref_suffix + '0'``  
``val_tst_dir + test_prefix + '.' + val_ref_suffix + '1'``  
``......``
 
### Training
Before training, parameters about training in the file ``wargs.py`` should be configured  
then, run ``sh train.sh``

### Inference
Assume that the trained model is named ``best.model.pt``  
Before decoding, parameters about inference in the file ``wargs.py`` should be configured  
+ translate one sentence  
run ``python bin/wtrans.py -m best.model.pt``
+ translate one file  
	+ put the test file to be translated into the path ``val_tst_dir + '/'``  
	+ run ``sh trans.sh filename``

# evaluate alignment

score-alignments.py -d path/900 -s zh -t en -g wa -i force_decoding_alignment


