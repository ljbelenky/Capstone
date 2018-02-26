import os
import shutil
import pandas as pd
from random import sample
import os.path
from PIL import Image

'''Utility program to create holdout data set from images'''
'''Do not run more than once, or else training set will be decimated'''

def move_holdouts():

    holdout_images = []
    styles = []

    for style, number in zip(['Abstract','Cubism','Expressionism','Pointillism'],[0,0,0,1]):
        file_list = []
        for root, dirs, files in os.walk('static/images/{}'.format(style)):
            for name in files:
                file_list.append(os.path.join(root,name))

        holdouts = sample(file_list, number)

        for holdout in holdouts:
            # print(holdout)
            source = holdout
            destination = holdout.replace('static/images/','static/holdouts/')
            dest_dir = '/'.join(destination.split('/')[:-1])+'/'
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                shutil.move(source, destination)
                holdout_images.append(destination)
                styles.append(style)


# holdout_df = pd.DataFrame({'files':holdout_images,'actuals':styles})
# holdout_df.to_csv('holdouts.csv')
actuals = []
file_list = []
# height = []
# width = []
for style in ['Abstract','Cubism','Expressionism','Pointillism']:
    for root,dirs,files in os.walk('static/holdouts/{}'.format(style)):
        for name in files:
            file_list.append(os.path.join(root,name))
            actuals.append(style)
            # w, h = Image.open(os.path.join(root,name)).size
            # width.append(w)
            # height.append(h)

holdouts_df = pd.DataFrame({'files':file_list,'actuals':actuals})#,'width':width,'height':height})
counts = holdouts_df.groupby('actuals').count()
print(counts)
