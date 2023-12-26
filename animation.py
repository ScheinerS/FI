import sys
import os
import glob
import imageio.v3 as iio
import aux
# path = os.path.dirname(os.path.realpath('__file__'))
# sys.path.append(path)




def animate(state:str, d:int, var:str, noise_types:list, FPS:int=16, initial_seconds:int=2, final_seconds:int=2, verbose:bool=True, alpha:float=0):
    if verbose:
        print('Animation:\t\tstate=%s\td=%d\tvar=%s'%(state, d, var))
    for noise in noise_types:
        print(noise)
        files = {}
        files[noise] = {}
        files[noise]['real'] = glob.glob('plots' + os.sep + state + os.sep + 'd=' + str(d) + os.sep + noise + os.sep + var + os.sep + '*alpha=%.2f_re.png'%alpha)
        files[noise]['imag'] = glob.glob('plots' + os.sep + state + os.sep + 'd=' + str(d) + os.sep + noise + os.sep + var + os.sep + '*alpha=%.2f_im.png'%alpha)
        
        for component in files[noise].keys():
            files[noise][component] = sorted(files[noise][component])
        
        
        
        path = 'animations' + os.sep + state + os.sep + 'd=%d'%d
        aux.check_dir(path)
        
        for component in files[noise].keys():
            
            images = []
            for i in range(FPS*initial_seconds):
                images.append(iio.imread(files[noise][component][0]))
                    
            for file in files[noise][component]:
                images.append(iio.imread(file))
        
            for i in range(FPS*final_seconds):
                images.append(iio.imread(file))
        
            kargs = { 'fps': FPS,
                     # 'quality': 10,
                     'macro_block_size': None,
                     # 'ffmpeg_params': ['-s','600x450']
                     }
            # imageio.mimsave(gifOutputPath, images, 'FFMPEG', **kargs)
            if alpha==None:
                iio.imwrite(path + os.sep + '%s_%s_d=%d_%s.mp4'%(var, component, d, noise), images, **kargs)
            else:
                iio.imwrite(path + os.sep + '%s_%s_d=%d_alpha=%.2f_%s.mp4'%(var, component, d, alpha, noise), images, **kargs)



if __name__=='__main__':
    
    state = 'GHZplus'
    noise_types = ['Amplitude Damping', 'Depolarising Noise', 'Dephasing Noise']#['Phase Damping'] 
    d = 2
    variables = ['rho']#, 'C_0_1']
    alpha = 0
    FPS = 16
    initial_seconds = 2
    final_seconds = 2
    
    
    print('\nalpha=%.2f:'%alpha)
    for var in variables:
        print('\n%s:'%var)
        animate(state = state,
                d = d,
                var = var,
                noise_types = noise_types,
                FPS = FPS)