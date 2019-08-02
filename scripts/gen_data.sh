cat ./data/MAFA_Dataset/train_masks/imglist.txt ./data/celebA/faces/imglist.txt \
    ./data/WiderFace_modified/imglist.txt ./data/cropped_img_from_vid/imglist_train.txt > ./data/train.txt

cat ./data/MAFA_Dataset/test_masks/imglist.txt ./data/cropped_img_from_vid/imglist_test.txt > ./data/test.txt

shuf ./data/train.txt -o ./data/train.txt
shuf ./data/test.txt -o ./data/test.txt