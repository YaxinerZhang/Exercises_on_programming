import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Mandelbrot:
    def __init__(self, max_iters: int, radius: (float, int), real_axis: np.ndarray, imag_axis: np.ndarray):
        """
        Initialize the Mandelbrot set parameters.

        Arguments:
        ----------
        max_iters: int
            Maximum number of iterations.
        radius: float or int
            Boundary for iterations.
        real_axis: A real numpy array representing x-axis.
        imag_axis: An imaginary numpy array representing y-axis.
        """

        # Check if 'max_iters' is an integer and greater than zero
        if not isinstance(max_iters, int):
            raise TypeError("Argument 'max_iters' must be an integer.")
        if max_iters <= 0:
            raise ValueError("Argument 'max_iters' must be a positive number.")
    
        # Check if 'radius' is either an integer or a float and greater than zero
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise TypeError("Argument 'radius' must be a float number or an int number.")
        if radius <= 0:
            raise ValueError("Argument 'radius' must be a positive number.")

        #check if 'real_axis' is a numpy array
        if not isinstance(real_axis, np.ndarray):
            raise TypeError("Argument 'real_axis' must be a numpy array.")
    
        #check if 'imag_axis' is a numpy array
        if not isinstance(imag_axis, np.ndarray):
            raise TypeError("Argument 'imag_axis' must be a numpy array.")
     
        self.max_iters = max_iters 
        self.radius = radius
        self.real_axis = real_axis
        self.imag_axis = imag_axis

    def mandelbrot(self, x0: float, y0: float) -> int:
        """
        Mandelbrot function.
    
        Arguments:
        ----------
        x0: float
            The real part of the complex number.
        y0: float
            The imaginary part of the complex number.

        Returns:
        --------
        int
            The number of iterations before reaching the radius or max_iters.
        """
    
        #Type Checking
    
        # Check if 'x0' is of type complex
        if not isinstance(x0, float):
            raise TypeError("Argument 'x0' must be a float number.")
    
        # Check if 'y0' is of type complex
        if not isinstance(y0, float):
            raise TypeError("Argument 'y0' must be a float number.")
             
        x, y = 0.0, 0.0
        for n in range(self.max_iters):
            if x * x + y * y > self.radius:
                return n
            xtemp = x * x - y * y + x0
            y = 2 * x * y + y0
            x = xtemp
        return self.max_iters

    def generate(self) -> np.ndarray:
        """
        Generate the Mandelbrot set.

        Returns:
        --------
        np.ndarray
            A 2D array representing the Mandelbrot set.
        """
        
        res = np.zeros((len(self.imag_axis), len(self.real_axis)),dtype = int)
        hue = np.zeros((len(self.imag_axis), len(self.real_axis)))
        NumIterationsPerPixel = np.zeros(self.max_iters + 1)
        
        for j, imag in enumerate(self.imag_axis):
            for i, real in enumerate(self.real_axis):
                iteration = self.mandelbrot(real, imag)
                res[j, i] = iteration
                NumIterationsPerPixel[iteration] += 1
    
        total = 0
        for i in range(self.max_iters):
            total += NumIterationsPerPixel[i]
        
        for j, imag in enumerate(self.imag_axis):
            for i, real in enumerate(self.real_axis):
                iter = res[j, i]
                for k in range(iter):
                    hue[j, i] += NumIterationsPerPixel[k] /total
        return hue
            
    def plot(self, cmap, output_file):
        """
        Plot the Mandelbrot set.

        Arguments:
        ----------
        output_file: str
            The filename for saving the plot.
        """
        mandelbrot_img = self.generate()
        
         # Plotting the Mandelbrot set
        # Update global fontsize and frame linewidth.
        plt.rcParams.update({
            "font.size": 16, "axes.linewidth": 2,
            "xtick.major.width": 2, "ytick.major.width": 2})
        fig = plt.figure(figsize=(9, 8))  # Create figure.
    
        # Axes is very useful tool, always isolate axes from figure.
        ax = fig.add_subplot()  # Create axes on top of figure.
        ax.set_aspect("equal")  # Set x to y ratio to equal.
        pcm_data = np.log10(mandelbrot_img)
        pcm = ax.pcolormesh(self.real_axis, self.imag_axis, pcm_data, cmap=cmap)
    
        # Draw colorbar
        cb = plt.colorbar(mappable=pcm, ax=ax, shrink=0.9, pad=0.02)
        cb.ax.tick_params(direction="in")
        cb.ax.set_ylabel(r"log$_{10}(N_{iters})$", loc="center")
        cb.ax.set_yticks(np.arange(0, 2 + 0.5, 0.5))
    
        # Make all ticks inward.
        ax.tick_params(direction="in")
    
        # Polish x-axis
        ax.set_xlabel("Real Axis")
        ax.set_xlim([np.min(self.real_axis), np.max(self.real_axis)])
        ax.set_xticks(np.arange(-1.5, 0.5 + 0.5, 0.5))
        # Polish y-axis
        ax.set_ylabel("Imaginary Axis")
        ax.set_ylim([np.min(self.imag_axis), np.max(self.imag_axis)])
    
        # The star '*' means Non-Keyword Arguments
        ax.set_title(r"Mandelbrot Set: $z_{n+1}=z_n^2 + c$", y=1.01)
        plt.tight_layout()  # Let matplotlib to auto polish your plot.
        plt.savefig(output_file)

class Mandelbrot_EscapeTimeAlgorithms(Mandelbrot):
    """
    a subclass of Mandelbrot
    """
    def generate(self) -> np.ndarray:
        """
        Generate the Mandelbrot set.

        Returns:
        --------
        np.ndarray
            A 2D array representing the Mandelbrot set.
        """
    
        res = np.zeros((len(self.imag_axis), len(self.real_axis))) 
        for j, imag in enumerate(self.imag_axis):
            for i, real in enumerate(self.real_axis):
                res[j, i] = self.mandelbrot(real, imag)
        return res

class Mandelbrot_ColoringAlgorithms(Mandelbrot):
    """
    a subclass of Mandelbrot
    """
    def generate(self) -> np.ndarray:
        """
        Generate the Mandelbrot set.

        Returns:
        --------
        np.ndarray
            A 2D array representing the Mandelbrot set.
        """
        
        res = np.zeros((len(self.imag_axis), len(self.real_axis)),dtype = int)
        hue = np.zeros((len(self.imag_axis), len(self.real_axis)))
        NumIterationsPerPixel = np.zeros(self.max_iters + 1)
        
        for j, imag in enumerate(self.imag_axis):
            for i, real in enumerate(self.real_axis):
                iteration = self.mandelbrot(real, imag)
                res[j, i] = iteration
                NumIterationsPerPixel[iteration] += 1
    
        total = 0
        for i in range(self.max_iters):
            total += NumIterationsPerPixel[i]
        
        for j, imag in enumerate(self.imag_axis):
            for i, real in enumerate(self.real_axis):
                iter = res[j, i]
                for k in range(iter):
                    hue[j, i] += NumIterationsPerPixel[k] /total
        return hue

if __name__ == "__main__":  
    #parameters
    lower, upper = (-2.0, -1.5), (0.75, 1.5)
    num_grids = (1000, 1000)
    max_iters = 100
    radius = 4 
    output_file = "mandelbrot_ColoringAlgorithms.pdf"

    # Construct real and imag axis.
    axes = [np.linspace(l, u, n) for l, u, n in zip(lower, upper, num_grids)]
    # Compute Mandelbrot set.

    
    mandelbrot = Mandelbrot_EscapeTimeAlgorithms(max_iters, radius, *axes)
    mandelbrot.plot("coolwarm", output_file)
