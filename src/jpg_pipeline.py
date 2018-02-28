from PIL import Image
import numpy as np

'''For use in Art Guesser
This class loads a jpg and prepares it by cropping and sizing  in the manner used to train the neural network'''

class jpgPipeline():
    def __init__(self,filename, image_mode = 'RGB'):
        image = Image.open(filename).convert(mode = image_mode)
        box = image.getbbox()
        image = image.crop(box)
        w = box[2]-box[0]
        h = box[3]-box[1]
        a = min(w,h)*.9
        left = (w-a)/2
        right = (w+a)/2
        top = (h-a)/2
        bottom = (h+a)/2
        cropped_image = image.crop((left,top,right,bottom))
        large_thumbnail=cropped_image.resize((200,200))
        large = np.expand_dims(np.asarray(large_thumbnail)/255,axis = 0)
        small_thumbnail=cropped_image.resize((150,150))
        small = np.expand_dims(np.asarray(small_thumbnail)/255, axis = 0)
        self.X = {'large':large,'small':small}
