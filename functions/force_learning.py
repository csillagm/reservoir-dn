import numpy as np

def force(reservoir, target, alpha, training_T, dt):


    Q = reservoir[0]
    w_feedback = reservoir[1]
    w_output = np.array(reservoir[2] / 10).reshape((Q.shape[0],1))
    #x = 0.5*np.random.normal(0,1,[N_neurons,1])
    x = reservoir[3].reshape((Q.shape[0],1))
    r = np.tanh(x)

    # print(Q.shape,x.shape,r.shape,w_output.shape)

    f = np.array(target[0:int(training_T/dt)])

    z = np.empty(int(training_T/dt))
    z[0] = 0.5*np.random.rand(1)

    error = np.empty(int(training_T/dt))
    w_out_len = np.empty(int(training_T/dt))

    P = np.identity(Q.shape[0]) / alpha

    for t in range(1, int(training_T/dt)):

        x = (1.0 - dt)*x + np.dot(Q,r*dt) + w_feedback *(z[t-1]*dt)
        r = np.tanh(x)
        z[t] = np.dot(w_output.T,r)


        if t % 10 == 0:
            k = np.dot(P,r)
            rPr = float(np.dot(r.T,k))
            c = 1.0 / (1.0 + rPr)
            P = P - np.outer(k, k)*c

            error[t] = z[t]-f[t]
            dw = -error[t]*k*c
            w_output = w_output + dw

        w_out_len[t] = np.linalg.norm(w_output)

    state = x

    return w_output, state

