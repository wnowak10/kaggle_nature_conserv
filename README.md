# kaggle_nature_conserv

steps

0. prepare_files to begin building dataframe with file, label. 
1. run anokas to find boat ID's (clustering ID) for each image
2. use tensorflow inception to get features
3. split train, validation set
3. run xgboost on 2048 features, plus boat id. return prediction on validation
4. run tensorflow classification. return prediction on validation
5. create ensemble. combine weighted xgboost and tensorflow.