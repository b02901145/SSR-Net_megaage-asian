import numpy as np
import sys
import logging
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from TYY_utils import mk_dir, load_data_npz
from TYY_model import TYY_MobileNet_reg

def MAE(a,b):
  mae = np.sum(np.absolute(a-b))
  mae/=len(b)
  return mae

'''''''''''''''''''''''''''''''''''''''''''''
  file name
'''''''''''''''''''''''''''''''''''''''''''''
test_file = sys.argv[1]
netType = int(sys.argv[2])

logging.debug("Loading testing data...")
image2, age2, image_size = load_data_npz(test_file)

if netType == 1:
    alpha = 0.25
    model_file = 'megaage_models/MobileNet/batch_size_50/mobilenet_reg_0.25_64/mobilenet_reg_0.25_64.h5'
    model = TYY_MobileNet_reg(image_size,alpha)()
    mk_dir('Results_csv')
    save_name = 'Results_csv/mobilenet_reg_%s_%d.csv' % (alpha, image_size)

elif netType == 2:
    alpha = 0.5
    model_file = 'megaage_models/MobileNet/batch_size_50/mobilenet_reg_0.5_64/mobilenet_reg_0.5_64.h5'
    model = TYY_MobileNet_reg(image_size,alpha)()
    mk_dir('Results_csv')
    save_name = 'Results_csv/mobilenet_reg_%s_%d.csv' % (alpha, image_size)


'''''''''''''''''''''''''''''''''''''''''''''
  load data
'''''''''''''''''''''''''''''''''''''''''''''
logging.debug("Loading model file...")
model.load_weights(model_file)

age_p=model.predict(image2,batch_size=len(image2))

'''''''''''''''''''''''''''''''''''''''''''''
  prediction
'''''''''''''''''''''''''''''''''''''''''''''
pred=[['MAE'],[str(MAE(age2,age_p[:,0]))],['ID','age','age_p','error']]
pred.append(['CA3','CA5'])
pred.append(['0','0'])
CA3=0
CA5=0
for i in range(0,len(image2)):
  error=np.absolute(age2[i]-age_p[i,0])
  if error<=3:
    CA3+=1
  if error<=5:
    CA5+=1
  temp = [str(i), str(age2[i]), str(age_p[i,0]), str(error)]
  pred.append(temp)

CA3/=len(image2)
CA5/=len(image2)
pred[4]=[str(CA3),str(CA5)]

print('CA3: ',CA3,'\nCA5: ',CA5)

f=open(save_name,'w')
w=csv.writer(f)
w.writerows(pred)
f.close
