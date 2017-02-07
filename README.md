# kaggle_nature_conserv

steps

0. prepare_files to begin building dataframe with file, label. 
1. run anokas to find boat ID's (clustering ID) for each image. return
	a dataframe with file, boatID. i should be able to merge this dataframe
	with the prepare_files df, merging on file_path

2. use tensorflow inception to get features for each image. create a pd data
	frame here. convert between np arrays and pd dfs as needed

3. split train, validation set
w
3. run xgboost on 2048 features, plus boat id. return prediction on validation
4. run tensorflow classification. return prediction on validation
5. create ensemble. combine weighted xgboost and tensorflow.