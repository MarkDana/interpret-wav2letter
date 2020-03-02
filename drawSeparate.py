import os
import drawSeparate_26
import drawSeparate_31

rawdir='/Users/markdana/Desktop/data'
for folder in os.listdir(rawdir):
    if folder in ['.DS_Store', ]: continue

    print(folder)

    files = os.listdir(os.path.join(rawdir, folder))
    drawSeparate_31.draw_api(os.path.join(rawdir, folder, 'preOutput.txt'))
    drawSeparate_31.draw_api(os.path.join(rawdir, folder, 'lastOutput.txt'))

    drawSeparate_26.draw_api(os.path.join(rawdir, folder, 'preOutput_0.txt'))
    drawSeparate_26.draw_api(os.path.join(rawdir, folder, 'lastOutput_0.txt'))
