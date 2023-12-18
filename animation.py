import sys
import os
import glob
import imageio
import aux
# path = os.path.dirname(os.path.realpath('__file__'))
# sys.path.append(path)




def animate(state:str, d:int, noise_types:list, FPS:int=16, AddInitialSecond:bool=True, AddFinalSecond:bool=True):
    for noise in noise_types:
        print(noise)
        files = {}
        files[noise] = {}
        files[noise]['real'] = glob.glob('plots' + os.sep + state + os.sep + 'd=' + str(d) + os.sep + noise + os.sep + 'rho' + os.sep + '*re.png')
        # files['imag'] = glob.glob('plots' + os.sep + state + os.sep + 'd=' + str(d) + os.sep + noise + os.sep + 'rho' + os.sep + '*im.png')
        
        for component in files[noise].keys():
            files[noise][component] = sorted(files[noise][component])
        
        
        images = []
        
        path = 'animations'
        aux.check_dir(path)
        
        for component in files[noise].keys():
            
            if AddInitialSecond:
                for i in range(FPS):
                    images.append(imageio.imread(files[noise][component][0]))
                    
            for file in files[noise][component]:
                images.append(imageio.imread(file))
        
            if AddFinalSecond:
                for i in range(FPS):
                    images.append(imageio.imread(file))
        
            
            imageio.mimsave(path + os.sep + 'rho_%s_d=%d_%s.mp4'%(component, d, noise), images, fps=FPS)



state = 'GHZplus'
noise_types = ['Amplitude Damping', 'Depolarising Noise', 'Phase Damping']
d = 3
FPS = 16    # frames per second
AddInitialSecond = True
AddFinalSecond = True   # to add a second at the end

animate(state, d, noise_types, FPS)