import subprocess as sp
tmp = sp.call('cls',shell=True)

import numpy as np
import matplotlib.pyplot as plt

''' Basic Example '''

#plt.plot([1,2,3,4])
#plt.ylabel('[1,2,3,4] are considered on ordinate')
#plt.xlabel('With no values for abscissa, py starts from 0 ')
#plt.show()


''' Controlling Line and Axis properties '''

#t = np.arange(0., 5., 0.2)
## red dashes, blue squares and green triangles
#plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^',
#         linewidth = 1.0)
#plt.axis([0, 6, 0, 200])
#
### Alternative to axis function
##plt.xlim(-2,10)
##plt.ylim(-20,200)
#
#plt.grid(True)
#plt.show()


''' Multiple figures and subplots '''

#x1 = np.arange(0.0, 5.0, 0.1)
#y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
#x2 = np.arange(0.0, 5.0, 0.02)
#y2 = np.cos(2 * np.pi * x2)

## Subplots
## subplot(n_rows, n_cols, index)
#
#plt.subplot(2, 1, 1)
#plt.plot(x1, y1, 'bo')
#
#plt.subplot(212)
#plt.plot(x2, y2, 'r--')
#
#plt.show()


## Multiple figures
#
#plt.figure(1)
##plt.subplot(3,3,1)
#plt.plot(x1, y1, 'bo')
#
#plt.figure(2)
##plt.subplot(3,2,2)
#plt.plot(x2, y2, 'k')
#
#plt.show()



''' Labels and Text '''

#x1 = np.arange(0.0, 5.0, 0.1)
#y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
#
#plt.plot(x1, y1, 'bs', x1, y1, 'k-')    # k for black
#plt.xlabel('Abscissa', fontsize = 10, color = 'b')
#plt.ylabel('Ordinate')
#plt.title('Figure Title')
#plt.suptitle("Main heading",
#             fontsize = 14,
#             fontweight = 'bold')
#
## TeX symbols available at
## https://matplotlib.org/1.5.3/users/mathtext.html
#plt.text(1, .4, r'$\theta=60 \degree$') # TeX
#
#plt.annotate('2nd crest', xy=(2, 0.17), 
#             xytext=(2.5, 0.3),
#            arrowprops=dict(facecolor='green', 
#                            shrink=0.02),)
#
#plt.grid(True)
#plt.show()


# Misc Labelling
#x = np.array([1,1,1,1,2,3,4,4,4,4,3,2,1])
#y = np.array([1,2,3,4,4,4,4,3,2,1,1,1,1])
#
#plt.plot(x,y)
#
## ha (align X-value) -> [center | right | left]
## va (align Y-value) -> [center | top | bottom]
#plt.text(3.9,1.05,'Text1',
#         ha = 'right',
#         verticalalignment = 'bottom',
#         rotation = 'horizontal'
#         )
#
#plt.text(4,2.5,'Text2',
#         horizontalalignment = 'right',
#         va = 'center',
#         rotation = 'vertical'
#         )
#
#plt.text(1,4,'Text3 \n next line',
#         horizontalalignment = 'left',
#         verticalalignment = 'top',
#         rotation = 45
#         )


# Text box

#x1 = np.arange(0.0, 5.0, 0.1)
#y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
#
#fig, ax = plt.subplots(1)
#ax.plot(x1, y1)
#plt.text(0.65, 0.85, r'$\lambda = 2$, $\alpha = 5$',
#         transform = ax.transAxes,  # makes width and height in percentage
#         va = 'top',
#         bbox = dict(
#                 boxstyle = 'round',
#                 facecolor = 'wheat',
#                 alpha = 0.2))  # alpha -> transparency
#
#fig.show()



''' Non-uniform scale/axis '''

#x1 = np.arange(0.0, 1000.0, 0.5)
#y1 = np.log10(x1)
#
#plt.subplot(211)
#plt.plot(x1, y1)
#plt.yscale('linear')
#plt.title('Linear Scale')
#
#plt.subplot(212)
#plt.plot(x1, y1)
#plt.yscale('log')
#plt.title('Log Scale')


''' Customizing plots background'''
#
#print(plt.style.available)
#
#plt.style.use('classic')
#plt.plot([1,12,3],[20,23,12])



''' Image Operations '''

#import matplotlib.image as img
#
#I = img.imread('files/stinkbug.png')
#
#plt.imshow(I)
#
## colormap: e.g. gist_earth, ocean, jet, etc.
#plt.imshow(I[:,:,0], cmap = 'hot')
#plt.colorbar()



''' Legend '''

#x1 = np.arange(0.0, 5.0, 0.1)
#y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
#x2 = np.arange(0.0, 5.0, 0.02)
#y2 = np.cos(2 * np.pi * x2)
#
#plt.plot(x1, y1, 'o-', label = "Legend I")
#plt.plot(x2, y2, '.-', label = "Legend II")
##plt.axis([0, 5.5, -1, 1.5])
#
#plt.legend(loc = 'best',    # upper left
#           shadow = True,
#           fontsize = 'medium',
#           framealpha = 0.5)    # transperancy

            

''' Ticks '''

#x1 = np.arange(0.0, 5.0, 0.1)
#y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(x1, y1)

#for tick in ax.yaxis.get_major_ticks():
#    tick.label1On = False
#    tick.label2On = True
#    tick.label2.set_color('green')
#
#for tick in ax.xaxis.get_major_ticks():
#    tick.label1On = False
#    tick.label2On = True
#    tick.label2.set_color('red')
    
    
    
''' Dynamic Subplots '''

#x1 = np.arange(0.0, 5.0, 0.1)
#y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
#
#plt.figure(0)
## similar to subplot(3,3,1)
## making 3 * 3 canvas space and assigning subplots
## at values like (0,0)
#ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
#ax1.plot(x1,y1)
#ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
#ax2.plot(x1,y1)
#ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
#ax3.plot(x1,y1)
#ax4 = plt.subplot2grid((3,3), (2, 0))
#ax4.plot(x1,y1)
#ax5 = plt.subplot2grid((3,3), (2, 1))
#ax5.plot(x1,y1)
#
#plt.show()
#
## Adjust spaces between subplots
#plt.tight_layout()



''' Highlighting areas '''

#import matplotlib.patches as patches
#import matplotlib.transforms as transforms
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#x1 = np.arange(0.0, 5.0, 0.1)
#y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
#
#ax.plot(x1, y1)


## Highlighting Vertically
#trans = transforms.blended_transform_factory(
#    ax.transData, ax.transAxes)
#
## height in percentage
## width per unit
#rect = patches.Rectangle((0,0), width=1, height=1,
#                         transform=trans, color='yellow',
#                         alpha=1)
#
#ax.add_patch(rect)
#plt.show()


## Highlighting Horizontally
#trans = transforms.blended_transform_factory(
#    ax.transAxes, ax.transData)
#
#rect = patches.Rectangle((0,-.2), width=1, height=0.6,
#                         transform=trans, color='yellow',
#                         alpha=1)
#ax.add_patch(rect)
#plt.show()



''' Shadow '''

#import matplotlib.patheffects as pe
#
#x1 = np.arange(0.0, 5.0, 0.1)
#y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
#
#plt.plot(x1, y1, linewidth = 5,
#         path_effects = [pe.SimpleLineShadow(),
#                         pe.Normal()])
#    
#plt.text(3, 0.8, "Hello World!",
#         path_effects = [pe.withSimplePatchShadow()])
#    
#plt.show()



''' Plots '''

''' Histogram '''

#plt.hist(np.random.randn(1000),
#         100,   # Number of bins, by default 10
#         facecolor = 'green',
#         alpha = 0.5)


''' Bar charts '''

## Vertical bar charts
#
#n_grps = np.arange(5)
#bar_width = 0.4
#
#var1 = [20, 30, 10, 50, 90]
#err1 = [2, 3, 4, 5, 4]
#
#var2 = [10, 123, 19, 60, 40]
#err2 = [1, 6, 2, 8, 7]
#
#plt.bar(n_grps, var1, bar_width,
#                 alpha = 0.5,
#                 color = 'b',
#                 yerr = err1,
#                 error_kw = {'ecolor': '0.5'},
#                 label='Var1')
#
## bar_width is imp. to pass else both bars overlap
#plt.bar(n_grps + bar_width, var2, bar_width,
#                 alpha = 0.5,
#                 color = 'r',
#                 yerr = err2,
#                 error_kw = {'ecolor': '0.5'},
#                 label='Var2')
#
#plt.legend()
#plt.xticks(n_grps + bar_width, 
#           ('A', 'B', 'C', 'D', 'E'))



## Horizontal bar charts
#
#n_grps = np.arange(5)
#bar_width = 0.4
#
#var1 = [20, 30, 10, 50, 90]
#err1 = [2, 3, 4, 5, 4]
#
#plt.barh(n_grps, var1, bar_width,
#         alpha = 0.5,
#         color = 'b',
#         xerr = err1,
#         label = 'Var1')
#
#plt.legend(loc = 'best')
#plt.yticks(n_grps + (bar_width)/2, 
#           ('A', 'B', 'C', 'D', 'E'))



''' Pie Chart '''

#names = ['Summer', 'Autumn', 'Spring', 'Winter']
#percent = [15, 30, 45, 10]
#colors = ['gold', 'lightcoral', 
#          'yellowgreen', 'lightskyblue']
#explode = (0, 0, 0.1, 0)  # takes out only the 3rd slice 
#
#plt.pie(percent, explode = explode,
#        labels = names, colors = colors,
#        autopct='%.2f%%',   # display value
#        shadow=True,
#        startangle=90)
## Set aspect ratio to be equal so that 
## pie is drawn as a circle.
#plt.axis('equal')



''' Scatter plot '''

## Single type of value
#
#x = np.random.randn(100)
#y = np.random.randn(100) * 1.7
#
#plt.scatter(x, y, c = (0.2, 0.5, 0.3))
#plt.show()


## Multiple value
#
#fig, ax = plt.subplots()
#for color in ['red', 'black', 'yellow']:
#    n = 50
#    x, y = np.random.rand(2,n)
#    ax.scatter(x, y, c = color, alpha = 0.5)
#    
#plt.show()



''' Box plot '''

#plt.boxplot([np.linspace(0,60,30)])
#plt.show()


## With outliers
#plt.boxplot(np.concatenate((
#                np.random.rand(50)*100,
#                np.ones(25)*50,
#                np.random.rand(10)*100+100,
#                np.random.rand(10)*-100),
#                axis = 0))
#    
#plt.show()
