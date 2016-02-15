import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as op 

def runge_kutta_step_class(f_prime, h, y_cur):
    """
    Here y_cur is the current value of y. 
    """
    k1 = f_prime(h, y_cur)
    k2 = k1 + (h/2.)* f_prime(h,(h/2.)*k1) 
    k3 = k1 + (h/2.) * f_prime(h, (h/2.)*k2) 
    k4 = k1 + h*f_prime(h, k3) 
    return k1 + (1./6.)*(y_prime(k1) + 2*y_prime(k2) + 2*y_prime(k3)+y_prime(k4))

def runge_kutta_step_wiki(f_prime, h, y_cur, *args): 
    """
    Here y_cur is the current value of y in the simulation.
    Assuming that f_prime has no direct dependence on t. 
    h is the step size in the simulation. 
    """
    k1 = f_prime(y_cur, *args) 
    k2 = f_prime(y_cur + (h/2.)*k1, *args) 
    k3 = f_prime(y_cur + (h/2.)*k2, *args) 
    k4 = f_prime(y_cur + h*k3, *args) 
    return y_cur + (h/6.)*(k1 + 2*k2+ 2*k3 + k4) 

def f_prime(y, v_wind):
    """
    v_wind is a velocity vector of the following form:
    v_wind = (v_wind_x, 0) 
    """
    y_prime = np.zeros(4) #dealing with 2D problem here
    x, vx, y, vy = y 
    v = np.sqrt((vx-v_wind)**2 + vy**2) 
    vd = 35. #This is some constant. 
    delta = 5 #Another constant. 
    g = 9.8 # a constant 
    b2 = 0.0039 + 0.0058/(1. + np.exp((v - vd)/delta)) 
    y_prime[0] = vx 
    y_prime[1] = -(b2*v*(vx-v_wind))
    y_prime[2] = vy 
    y_prime[3] = -(g + (b2*v*vy))
    return y_prime 

def run_simulation(theta, *args):
    """
    Assumes that you give the starting velocity and height of player in 
    imperial (ick) units
    """
    try:
        v_init, height_init = args
        v_init *= 1609.34/3600. #converting to m/s
        height_init *= 0.3048 # converting from feet to meters          
    except:
        v_init, height_init = 110., 6. #110 mph, 6 ft tall
        v_init *= 1609.34/3600. #converting to m/s
        height_init *= 0.3048 # converting from feet to meters    
    y = np.array([0.0, np.cos(theta)*v_init, height_init, np.sin(theta)*v_init])  
    h = 0.001 
    v_wind = 0
    ys = [y] 
    while (y[2] > 0):
        y = runge_kutta_step_wiki(f_prime, h, y, v_wind) 
        ys.append(y) 
    
    return ys[-1][0] 
 
def main():
    #try to find minimimum of simulation 
    fun = lambda x: -run_simulation(x) 
    res = op.minimize(fun, np.pi/4., method='Nelder-Mead') 
    print(res) 

    #ys = run_simulation(np.pi/4.) 
    #
    #fig = plt.figure()
    #ax = fig.add_subplot(111) 
    #ax.set_title("Batted Ball. Max X: {}".format(ys[-1][0]))
    #x = [y[0] for y in ys] 
    #y = [y[2] for y in ys] 
    #ax.plot(x,y) 
    #plt.show()
main() 
