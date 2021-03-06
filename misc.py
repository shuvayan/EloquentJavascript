import subprocess as sp
tmp = sp.call('cls',shell=True)


''' Saving a graph '''

#import matplotlib.pyplot as plt
#
#plt.plot([1,2,3,4])
#plt.ylabel('[1,2,3,4] are considered on ordinate')
#plt.xlabel('With no values for abscissa, py starts from 0 ')
#plt.savefig('files/fig.pdf')


''' 3d scatter plot (rotatable)'''

#from mpl_toolkits.mplot3d import axes3d
#import matplotlib.pyplot as plt
#import numpy as np
#
#fig = plt.figure()
#ax = fig.add_subplot(111, 
#                     projection='3d')   
#for color in ['red', 'black', 'yellow']:
#    n = 50
#    x, y, z = np.random.rand(3,n)
#    ax.scatter(x, y, z, c = color, alpha = 0.5)
#    
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#plt.show()
#
### rotate the axes and update
##for angle in range(0, 360):
##    ax.view_init(30, angle)
##    plt.draw()
##    plt.pause(.001)

''' Animated plots '''

#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import animation
#   
#n = 30
#x = np.arange(0,1,0.001)
#y = np.ones( (1000,n) )
#
#for i in range(0,n):
#    y[:,i] = np.sin(2 * np.pi * x) * i+1
#
#
#def func(arg):
#     plt.cla()   #Clear axis
#     plt.plot(y[:,arg])
#     plt.ylim(-30,30)
#
#fig = plt.figure(figsize=(5,4))
#      
#to_save = animation.FuncAnimation(fig, func, frames=30)
##to_save.save('demoanimation.gif', writer='imagemagick', fps=4);


''' Reading and Writing Pandas DataFrame '''

## http://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection
#
#import pandas as pd
#
#df = pd.read_csv('files/Youtube04-Eminem.csv', 
#                 names = ['COMMENT_ID', 'AUTHOR'])
#
#print(df.iloc[:,1])
#
#
#df.iloc[:,1].to_csv('files/result.csv')


''' Cropping image region '''

#import numpy as np
#import matplotlib.pyplot as plt
#
#from skimage import data
#from skimage import transform as tf
#
#text = data.text()
#
#inp = np.array([[155, 15], [65, 40], [260, 130], [360, 95]])
#out = np.array([[0, 0], [0, 50], [300, 50], [300, 0]])
#
## Applying projective transform
#tform3 = tf.ProjectiveTransform()
#tform3.estimate(out, inp)
#
## Warping the selected image portion to given shape
#warped = tf.warp(text, tform3, 
#                 output_shape=(50, 300))
#
#fig, ax = plt.subplots(nrows=2, figsize=(8, 3))
#
#ax[0].imshow(text, cmap=plt.cm.gray)
#
## marking red dots on original image
#ax[0].plot(inp[:, 0], inp[:, 1], '.r')
#
#ax[1].imshow(warped, cmap=plt.cm.gray)
#
#for a in ax:
#    a.axis('off')
#
#plt.tight_layout()


''' Edge Detection '''

#import matplotlib.pyplot as plt
#from skimage.data import camera
#from skimage.filters import roberts, sobel #, scharr, prewitt
#
#image = camera()
#
#fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
#                       figsize=(8, 4))
#
#ax[0].imshow(roberts(image), cmap=plt.cm.gray)
#ax[0].set_title('Roberts Edge Detection')
#
#ax[1].imshow(sobel(image), cmap=plt.cm.gray)
#ax[1].set_title('Sobel Edge Detection')
#
#for a in ax:
#    a.axis('off')
#
#plt.tight_layout()
#plt.show()



''' Image selection and removal '''

#from skimage import data, draw
#from skimage import transform, util
#import numpy as np
#from skimage import filters, color
#from matplotlib import pyplot as plt
#
#
#hl_color = np.array([0, 1, 0])  # used for coloring
#
#img = data.rocket()
#img = util.img_as_float(img)
## Sobel detector : to identify imp. pixels
#eimg = filters.sobel(color.rgb2gray(img))

#plt.title('Original Image')
#plt.imshow(img)


### Image resizing
#out = transform.seam_carve(img, eimg, 'vertical', 
#                           200) # number of conscutive similar pixels to be removed
#plt.figure()
#plt.title('Resized using Seam Carving')
#plt.imshow(out)


## Object Marking
#
#masked_img = img.copy()
#
#poly = [(404, 281), (404, 360), (359, 364), (338, 337), (145, 337), (120, 322),
#        (145, 304), (340, 306), (362, 284)]
#pr = np.array([p[0] for p in poly])
#pc = np.array([p[1] for p in poly])
#rr, cc = draw.polygon(pr, pc)
#
#masked_img[rr, cc, :] = masked_img[rr, cc, :]*0.5 + hl_color*.5   # transparency + color
#plt.figure()
#plt.title('Object Marked')
#
#plt.imshow(masked_img)


# Object removal

## Edge of selected image
#eimg[rr, cc] -= 1000
#
#plt.figure()
#plt.title('Object Removed')
#out = transform.seam_carve(img, eimg, 'vertical', 90)
#plt.imshow(out)
#plt.show()