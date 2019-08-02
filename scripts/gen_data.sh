cat ./data/MAFA/train_masks/imglist.txt ./data/celebA/faces/imglist.txt \
    ./data/WiderFace_modified/imglist.txt ./data/mask_classifier/imglist_train.txt > ./data/train.txt

cat ./data/MAFA/test_masks/imglist.txt ./data/mask_classifier/imglist_test.txt > ./data/test.txt

shuf ./data/train.txt -o ./data/train.txt
shuf ./data/test.txt -o ./data/test.txt
