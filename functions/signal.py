import numpy as np
import datetime
from random import randrange


def generate_fourier_signal(n_fourier, period_fourier, signal_T, dt):

    a = np.random.uniform(-0.5,0.5,n_fourier)
    b = np.random.uniform(-0.5,0.5,n_fourier)

    t = np.linspace(0, signal_T, int(signal_T/dt))

    f = fourier_series(t, a, b, period_fourier)

    scale = 4.5/(abs(max(f))+abs(min(f)))
    f = f * scale

    return [f, [a,b]]


def fourier_series(t, a, b, period_fourier):

    f = 0
    for i in range(0, len(a)):
        f = f + a[i]*np.cos(2*np.pi*(i+1)*t / period_fourier) + b[i]*np.sin(2*np.pi*(i+1)*t / period_fourier)

    return f


def save_initial_conditions(n_fourier, n_population):

    conditions = np.empty(n_population, dtype=list)

    for i in range(n_population):
        a = np.random.uniform(-0.5,0.5,n_fourier)
        b = np.random.uniform(-0.5,0.5,n_fourier)
        conditions[i] = [a,b]

    print(conditions)

    filename = str(n_population)+"-initial_conditions_" + datetime.datetime.now().strftime("%H%M")

    np.save(filename, conditions)


def fourier_series_from_coeffs(coeffs, signal_T, dt, period_fourier):

    a = coeffs[0]
    b = coeffs[1]

    t = np.linspace(0,signal_T,int(signal_T/dt))

    f = fourier_series(t, a, b, period_fourier)
    scale = 4.5/(abs(max(f))+abs(min(f)))
    f = f * scale

    return [f, [a, b]]


# noise functions
def gen_white_noise(amplitude, length):
    return np.random.normal(0, amplitude, length)


def gen_fourier_noise(amplitude, signal_T, dt, n_fourier, period_fourier):
    """
    Generates Fourier-signal with coefficients sampled from N(0,amplitude)
    This will be added to signals as copying noise.

    Parameters
    ----------
    amplitude: float
        sd of the Gaussian distribution the coefficients are sampled from

    Returns
        noise: array
    -------

    """
    a = np.random.normal(0, amplitude, n_fourier)
    b = np.random.normal(0, amplitude, n_fourier)

    t = np.linspace(0, signal_T, int(signal_T/dt))

    noise = fourier_series(t, a, b, period_fourier)

    return noise


def noise_power(noise_function):
    return sum(np.power(noise_function,2))


def generate_signal(reservoir, signal_T, dt):

    Q = reservoir[0]
    w_feedback = np.array(reservoir[1]).reshape([Q.shape[0],1])
    w_output = np.array(reservoir[2]).reshape([Q.shape[0],1])
    x = reservoir[3].reshape([Q.shape[0],1])
    r = np.tanh(x)

    z = np.empty(int(signal_T / dt), dtype='float64')
    z[0] = 0.5*np.random.randn(1)

    # print("x:",x.shape,"r:",r.shape,"w_o:",w_output.shape,"w_f:",w_feedback.shape)

    for t in range(1, int(signal_T / dt)):
        x = (1.0 - dt) * x + np.dot(Q, r * dt) + w_feedback * (z[t - 1] * dt)
        r = np.tanh(x)
        z[t] = np.dot(w_output.T,r)

    return z, x


# signal evaluation
def align_signal(signal, eval_length, timebin):

    '''
     Cuts an eval_length long signal (should be one period if working with Fourier-signals) that will be evaluated.
     We find for the maximum of the signal, and from here, we are looking for an index,
     where the signal values turn from non-positive to positive. If we find such index = i,
     then aligned_signal = signal[i : i+eval_length]

     :param signal: input signal - array
     :param eval_length: evaluation length - int
     :param timebin: if alignment is not successful (i.e. cannot find a point after the maximum of the signal
                     where the signal values turn from non-positive to positive)
                     signal[timebin : timebin+eval_length] will be the output - int
     :param add_noise: 'gaussian', 'uniform' or None
     :return: aligned: aligned signal - array
    '''

    signal_cut = signal[int(len(signal) / 3):int(2 * len(signal) / 3)]
    i = np.argmax(signal_cut, axis=0)
    i = i + int(len(signal) / 3)

    while not (signal[i] <= 0 and signal[i + 1] > 0):
        if i == (len(signal) - (1 + eval_length)):
            i = timebin
            break
        i = i + 1

    # eval_length is 300 currently
    start = int(i)

    aligned = signal[start:start+eval_length]

    return aligned, start


# convert signal to bitsring
def signal_to_bitstring_singlevalue(signal, timebin, threshold=0):

    bitstring=''
    for x in range(int(len(signal)/timebin)):
        if signal[randrange(x*timebin,(x+1)*timebin)]<threshold:
            bitstring+='0'
        else:
            bitstring+='1'

    return bitstring


def signal_to_bitstring_center(signal, timebin, threshold=0):

    bitstring=''
    for x in range(int(len(signal)/timebin)):
        if signal[int((x*timebin+(x+1)*timebin)/2)]<threshold:
            bitstring+='0'
        else:
            bitstring+='1'

    return bitstring


def signal_to_bitstring_allpos(slist, time, threshold=0):

    bitstring=''
    for x in range(int(len(slist)/time)):
        for c in slist[x*time:(x+1)*time]:
            if c < threshold:
                bitstring+='0'
                break
            bitstring+='1'

    return bitstring

