import numpy as np
import sys
import logging
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from TYY_utils import mk_dir, load_data_npz
from TYY_model import TYY_DenseNet_reg

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

if netType == 3:
    N_densenet = 3
    depth_densenet = 3*N_densenet+4
    model_file = 'megaage_models/DenseNet/batch_size_50/densenet_reg_%d_64/densenet_reg_%d_64.h5'%(depth_densenet, depth_densenet)
    model = TYY_DenseNet_reg(image_size,depth_densenet)()
    mk_dir('Results_csv')
    save_name = 'Results_csv/densenet_reg_%d_%d.csv' % (depth_densenet, image_size)

elif netType == 4:
    N_densenet = 5
    depth_densenet = 3*N_densenet+4
    model_file = 'megaage_models/DenseNet/batch_size_50/densenet_reg_%d_64/densenet_reg_%d_64.h5'%(depth_densenet, depth_densenet)
    model = TYY_DenseNet_reg(image_size,depth_densenet)()
    mk_dir('Results_csv')
    save_name = 'Results_csv/densenet_reg_%d_%d.csv' % (depth_densenet, image_size)


'''''''''''''''''''''''''''''''''''''''''''''
  load data
'''''''''''''''''''''''''''''''''''''''''''''
logging.debug("Loading model file...")
model.load_weights(model_file)

age_p=model.predict(image2)

'''''''''''''''''''''''''''''''''''''''''''''
  prediction
'''''''''''''''''''''''''''''''''''''''''''''
pred=[['MAE'],[str(MAE(age2,age_p[:,0]))],['CA3','CA5'],['0','0'],['ID','age','age_p','error']]
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
pred[3]=[str(CA3),str(CA5)]

print('CA3: ',CA3,'\nCA5: ',CA5)

f=open(save_name,'w')
w=csv.writer(f)
w.writerows(pred)
f.close
