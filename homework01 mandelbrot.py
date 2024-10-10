import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        #define mandelbrot_set
        def mandelbrot_set(c,times):
            z = 0
            for n in range(times):
                if abs(z) > 2.0:
                    return n
                z = z * z + c
            return times
        
        x = np.linspace(-2.0, 1.0, 1200)
        y = np.linspace(-1.5, 1.5, 1200)
        
        img = np.zeros((1200, 1200))
        
        times = 100
        for x in range(1200):
            for y in range(1200):
                a = -2.0 + x * 3 / 1200
                b = -1.5 + y * 3 / 1200
                c = complex(a, b)
                color =  np.log(mandelbrot_set(c, times) + 1)
                img[y, x] = color
        
        plt.tick_params( direction='in', labelsize=10, 
                        labeltop='on', labelbottom='on', labelleft='on', labelright='off', 
                        top='on', bottom='on', left='on', right='on')  
        font_self = { 'fontsize': 15}
        plt.xlabel('Real',fontdict = font_self )
        plt.ylabel('Imaginary',fontdict = font_self )
        plt.title('Mandelbrot Set',fontdict = font_self )
        
        plt.imshow(img, cmap='magma', extent=(-2.0, 1.0, -1.5, 1.5))
        plt.savefig('mandelbrot_set.pdf')
        plt.show()