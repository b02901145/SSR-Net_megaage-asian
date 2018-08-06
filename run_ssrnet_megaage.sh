max1=4
max2=4
for i in `seq 4 $max1`
do
    #echo "$i"
    for j in `seq 4 $max2`
    do
        #echo "$j" 50
        KERAS_BACKEND=tensorflow python ./SSRNET/SSRNET_train.py --input1 ./data/megaage_train.npz --input2 ./data/megaage_test.npz --db megaage --netType1 $i --netType2 $j --batch_size 50
    done
done
