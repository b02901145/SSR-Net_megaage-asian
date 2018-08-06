import numpy as np
import sys
import logging
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from SSRNET_model import SSR_net
from TYY_utils import mk_dir, load_data_npz

def MAE(a,b):
  mae = np.sum(np.absolute(a-b))
  mae/=len(a)
  return mae

'''''''''''''''''''''''''''''''''''''''''''''
  file name
'''''''''''''''''''''''''''''''''''''''''''''
test_file = sys.argv[1]
netType1 = int(sys.argv[2])
netType2 = int(sys.argv[3])
stage_num = [3,3,3]

lambda_local = 0.25*(netType1%5)
lambda_d = 0.25*(netType2%5)


logging.debug("Loading testing data...")
image2, age2, image_size = load_data_npz(test_file)

mk_dir('Results_csv')

model_file = 'megaage_models/batch_size_50/ssrnet_%d_%d_%d_%d_%s_%s/ssrnet_%d_%d_%d_%d_%s_%s.h5' % (stage_num[0],stage_num[1],stage_num[2], image_size, lambda_local, lambda_d, stage_num[0],stage_num[1],stage_num[2], image_size, lambda_local, lambda_d)
save_name = 'Results_csv/ssrnet_%d_%d_%d_%d_%s_%s_age.csv' % (stage_num[0],stage_num[1],stage_num[2], image_size, lambda_local, lambda_d)

'''''''''''''''''''''''''''''''''''''''''''''
  load data
'''''''''''''''''''''''''''''''''''''''''''''
model = SSR_net(image_size,stage_num, lambda_local, lambda_d)()

logging.debug("Loading model file...")
model.load_weights(model_file)

age_p=model.predict(image2)

'''''''''''''''''''''''''''''''''''''''''''''
  prediction
'''''''''''''''''''''''''''''''''''''''''''''
age_p2=age_p

pred=[['MAE'],[str(MAE(age2[age2>=-1],age_p2[age2>=-1]))],['CA3','CA5'],['0','0'],['ID','age','age_p','error']]
CA3=0
CA5=0
for i in range(0,len(image2)):
  error=np.absolute(age2[i]-age_p2[i])
  if age2[i]>=-1:
    if error<=3:
      CA3+=1
    if error<=5:
      CA5+=1
    temp = [str(i), str(age2[i]), str(age_p2[i]), str(error)]
    pred.append(temp)

CA3/=len(age2[age2>=-1])
CA5/=len(age2[age2>=-1])
pred[3]=[str(CA3),str(CA5)]

print('CA3: ',CA3,'\nCA5: ',CA5)

f=open(save_name,'w')
w=csv.writer(f)
w.writerows(pred)
f.close
