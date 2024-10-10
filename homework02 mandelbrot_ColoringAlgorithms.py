import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def mandelbrot(x0: float, y0: float, max_iters: int, radius: (float, int)) -> int:
    """
    Mandelbrot function.

    Arguments:
    ----------
    c: complex
        A complex number.
    Max_iters: int
        Max number of interations.
    Radius: float or int
        A boundary for interations.

    Returns:
    --------
    int
        The number of interations before reaching the radius or max_iters.   
    """

    #Type Checking

    # Check if 'x0' is of type complex
    if not isinstance(x0, float):
        raise TypeError("Argument 'x0' must be a float number.")

    # Check if 'y0' is of type complex
    if not isinstance(y0, float):
        raise TypeError("Argument 'y0' must be a float number.")

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
         
    x = 0.0
    y = 0.0
    for n in range(max_iters):
        if x * x + y * y > radius:
            return n
        xtemp = x * x - y * y + x0
        y = 2 * x * y + y0
        x = xtemp
    return max_iters

def get_mandelbrot(real_axis: np.ndarray, imag_axis: np.ndarray, max_iters:int, radius: (float, int) ) -> np.ndarray:
    """
    Mandelbrot set

    Arguments
    ---------
    real_axis: A real numpy array representing x-axis.
    imag_axis: An imaginary numpy array representing y-axis.
     Max_iters: int
        Max number of interations.
    Radius: float or int
        A boundary for interations. 
    """

    #check if 'real_axis' is a numpy array
    if not isinstance(real_axis, np.ndarray):
        raise TypeError("Argument 'real_axis' must be a numpy array.")

    #check if 'imag_axis' is a numpy array
    if not isinstance(imag_axis, np.ndarray):
        raise TypeError("Argument 'imag_axis' must be a numpy array.")
         
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
 
    res = np.zeros((len(imag_axis), len(real_axis)),dtype = int)
    hue = np.zeros((len(imag_axis), len(real_axis)))
    NumIterationsPerPixel = np.zeros(max_iters + 1)
    
    for j, imag in enumerate(imag_axis):
        for i, real in enumerate(real_axis):
            iteration = mandelbrot(real, imag, max_iters, radius)
            res[j, i] = iteration
            NumIterationsPerPixel[iteration] += 1

    total = 0
    for i in range(max_iters):
        total += NumIterationsPerPixel[i]
    
    for j, imag in enumerate(imag_axis):
        for i, real in enumerate(real_axis):
            iter = res[j, i]
            for k in range(iter):
                hue[j, i] += NumIterationsPerPixel[k] /total
    return hue
            
def plot_mandelbrot(real_axis: np.ndarray, imag_axis: np.ndarray, mandelbrot_img: np.ndarray, output_file):
    """
    Plot mandelbrot set.

    Arguments
    ---------
    real_axis: A real numpy array representing x-axis.
    imag_axis: A real numpy array representing y-axis.
    mandelbrot_img: 2D numpy array represeinting the Mandelbrot set.
    output_file: Output filename.
    """

    #check if 'real_axis' is a numpy array
    if not isinstance(real_axis, np.ndarray):
        raise TypeError("Argument 'real_axis' must be a numpy array.")

    #check if 'real_axis' is a numpy array
    if not isinstance(imag_axis, np.ndarray):
        raise TypeError("Argument 'imag_axis' must be a numpy array.")

    #check if 'mandelbrot_img' is a numpy array
    if not isinstance(imag_axis, np.ndarray):
        raise TypeError("Argument 'mandelbrot_img' must be a numpy array.")

 
    # Plotting the Mandelbrot set
    plt.rcParams.update({
        "font.size": 16, "axes.linewidth": 2,
        "xtick.major.width": 2, "ytick.major.width": 2})
    fig = plt.figure(figsize=(9, 8))  # Create figure.

    # Axes is very useful tool, always isolate axes from figure.
    ax = fig.add_subplot()  # Create axes on top of figure.
    ax.set_aspect("equal")  # Set x to y ratio to equal.
    pcm = ax.pcolormesh(real_axis, imag_axis, mandelbrot_img, cmap="coolwarm")

    # Draw colorbar
    cb = plt.colorbar(mappable=pcm, ax=ax, shrink=0.9, pad=0.02)
    cb.ax.tick_params(direction="in")
    cb.ax.set_yticks(np.arange(0, 1 + 0.2, 0.2))

    # Make all ticks inward.
    ax.tick_params(direction="in")

    # Polish x-axis
    ax.set_xlabel("Real Axis")
    ax.set_xlim([np.min(real_axis), np.max(real_axis)])
    ax.set_xticks(np.arange(-1.5, 0.5 + 0.5, 0.5))
    # Polish y-axis
    ax.set_ylabel("Imaginary Axis")
    ax.set_ylim([np.min(imag_axis), np.max(imag_axis)])

    # The star '*' means Non-Keyword Arguments
    ax.set_title(r"Mandelbrot Set: $z_{n+1}=z_n^2 + c$", y=1.01)
    plt.tight_layout()  # Let matplotlib to auto polish your plot.
    plt.savefig(output_file)

if __name__ == "__main__":  
    #parameters
    lower, upper = (-2.0, -1.5), (0.75, 1.5)
    num_grids = (1000, 1000)
    max_iters = 1000
    radius = 4 
    output_file = "mandelbrot_ColoringAlgorithms.pdf"

    # Construct real and imag axis.
    axes = [np.linspace(l, u, n) for l, u, n in zip(lower, upper, num_grids)]
    # Compute Mandelbrot set.
    mandelbrot_img = get_mandelbrot(*axes, max_iters, radius)
    # Plot Mandelbrot set.
    plot_mandelbrot(*axes, mandelbrot_img, output_file)



