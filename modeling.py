import pandas,numpy
import plotly.graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots
import torch
from torch.autograd import Variable

device = 'cpu'
torch.manual_seed(42)
epsilon=1e-10

# Lagrangian Fitting

def controlPath(time):
    c = pandas.Series(time,index=time)
    cdot = pandas.Series(1,index=time)
    cddot = pandas.Series(0,index=time)
    return c,cdot,cddot

def logLoss(x):
    return 1+torch.log((x+epsilon)**2)

def lossFunction(ND,ELC,ELD):
    loss = logLoss(ND)*logLoss(ELC)*((ELD+epsilon)/(ELC+epsilon))
    return loss

def parametrization(power_space):
    parameters = []
    for power in power_space:
        c = Variable(torch.rand(1),requires_grad=True)
        power['c'] = c
        parameters.append(c)
    return power_space,parameters

def trainPolynomial(power_space,q,qdot,qddot,lr=.00001,iterations=200):
    time = q.index
    c,cdot,cddot = controlPath(time)
    power_space,parameters = parametrization(power_space)
    PD = precomputeCoefficients(power_space,q,qdot,qddot,time)
    PC = precomputeCoefficients(power_space,c,cdot,cddot,time)
    optimizer = torch.optim.SGD(parameters,lr=lr)
    parameters = torch.cat(parameters)
    for iteration in range(iterations):
        ELC = timeIntegral(parameters,PC)
        ELD = timeIntegral(parameters,PD)
        ND = regularizer(power_space,q,qdot,qddot,time)
        loss = lossFunction(ND,ELC,ELD)
        print(loss.data.numpy())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    coefficients = [mono['c'].data.numpy()[0] for mono in power_space]
    return power_space,coefficients

# Lagrangian analysis

def precomputeCoefficients(power_space,q,qdot,qddot,time):
    P = []
    for t in time:
        EL = []
        for power in power_space:
            alpha,beta,gamma,c = power.values()
            el = (q[t]**alpha)*(qdot[t]**beta)*(t**gamma)
            el *= beta*gamma/(qdot[t]*t) + (beta-1)*alpha/q[t] + beta*(beta-1)*qddot[t]/(qdot[t]**2)
            EL.append(el)
        EL = Variable(torch.FloatTensor(EL))
        P.append(EL)
    return P

def regularizer(power_space,q,qdot,qddot,time):
    time_integral = Variable(torch.FloatTensor([0]))
    for t in time:
        EL = Variable(torch.FloatTensor([0]))
        for power in power_space:
            alpha,beta,gamma,c = power.values()
            el = c*(q[t]**alpha)*(qdot[t]**beta)*(t**gamma)
            A = el*beta
            A *= gamma/(qdot[t]*t) + alpha/q[t] + (beta-1)*qddot[t]/(qdot[t]**2)
            B = el*alpha/q[t]
            EL += torch.pow(A,2) + torch.pow(B,2)
        time_integral += EL

    return time_integral

def timeIntegral(coefficients,P):
    time_integral = 0#+Variable(torch.FloatTensor([0]))
    for p in P:
        el = coefficients*p
        el = torch.sum(el)
        time_integral += torch.pow(el,2)
    return time_integral

def EulerLagrangian(power_space,q,qdot,qddot,time):
    for t in time:
        EL = Variable(torch.FloatTensor([0]))
        for power in power_space:
            alpha,beta,gamma,c,P = power.values()
            el = (q[t]**alpha)*(qdot[t]**beta)*(t**gamma)
            el *= beta*gamma/(qdot[t]*t) + (beta-1)*alpha/q[t] + beta*(beta-1)*qddot[t]/(qdot[t]**2)
            el = c*el
            EL += el
        time_integral += torch.pow(EL,2)

    return time_integral

def polynomialLagrangian(power_space,q,qdot,t):
    L = 0
    for mono in power_space:
        L += mono['c']*(q**mono['alpha'])*(qdot**mono['beta'])*(t**mono['gamma'])
    return L

def evolutionLagrangian(power_space,q,qdot,time):
    lagMap = []
    for t in time:
        x_axis = []
        for position in q:
            y_axis = []
            for velocity in qdot:
                y_axis.append(polynomialLagrangian(power_space,position,velocity,t))
            x_axis.append(y_axis)
        lagMap.append(numpy.asarray(x_axis))
    return lagMap

## plotting methods

def plotTemporalHeatmap(data,x,y,time):
    data_slider = []
    for index,t in enumerate(time):
        frame = data[index]
        frame = {'type':'heatmap','z':frame,'x':x,'y':y}
        data_slider.append(frame)

    steps = []
    for i in range(len(data_slider)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(data_slider)],
                    label='{}'.format(i))
        step['args'][1][i] = True
        steps.append(step)

    sliders = [dict(active=0, pad={"t": 1}, steps=steps)]
    layout = dict(sliders=sliders)
    fig = {'data':data_slider,'layout':layout}
    py.iplot(fig)

def plotTimeIndicators(indicators,time,height=300,subplot=True):
    """
        plot Indicators as subplot time series
        cascading one after the otherc
    """
    N = len(indicators)
    if subplot:
        fig = make_subplots(shared_xaxes=True, rows=N, cols=1, vertical_spacing=.02)
        fig.update_layout(height=height*N,margin={'l':10,'r':10,'t':5,'b':5})
    else:
        fig = []

    for index,(name,series) in enumerate(indicators.items()):
        trace = go.Scatter(name=name,x=time,y=series)
        if subplot:
            fig.add_trace(trace,row=index+1,col=1)
        else:
            fig.append(trace)
    py.iplot(fig)

def plotJointIndicators(indicators,time):
    N = len(indicators)
    fig = []
    for index,(name,series) in enumerate(indicators.items()):
        if time is None:
            series = rescale(series)
            time = numpy.arange(0,1,(1/len(series)))
        trace = go.Scatter(name=name,x=time,y=series)
        fig.append(trace)

    py.iplot(fig)

# data manipulation

def rescale(series):
    return (series - series.min())/(series.max()-series.min()) * 100

def diff(series,lag):
    differe = series - numpy.roll(series,lag)
    differe[:lag] = numpy.nan
    return differe

def macd(series,slow,fast):
    return series.ewm(fast).mean() - series.ewm(slow).mean()

def countCrossovers(A,B):
    delta = A-B
    delta = numpy.sign(delta)
    delta = delta * delta.shift(1)
    crosses = delta[(delta==-1) | (delta==0)]

    return crosses.index.to_series()

def eventSegmentation(dM,events):
    segments = []
    begin = dM.index[0]
    for event in events:
        segments.append(dM.loc[begin:event])
        begin = event
    segments.append(dM.loc[begin:])
    return segments

def normalisation(series):
    series = series-series[0]
    series = series/abs(series)[-1]
    series = series.reset_index()
    series.index = series.index/series.index[-1]
    return series

def truncateTrend(series,trend=+1):
    series = series.copy()
    series.loc[series*trend < 0] = 0
    return series

def powerFitting(series,trend):
    series = series*trend
    series = series[series > 0]
    power = numpy.mean(numpy.log(series)/numpy.log(series.index))
    return trend,power#,mse

# fractal analysis

def HurstExponent(series,beta=2,lag=50):
    """
        close : price series of definite rolling size
        lag : range of lag to fit R/S statistic
        window >> lags
    """
    lags = numpy.arange(1,lag)
    tau = [numpy.mean(numpy.abs(numpy.power(diff(series,lag),beta))) for lag in lags]
    m = numpy.polyfit(numpy.log(lags), numpy.log(tau), 1)
    hurst = m[0]/beta
    return hurst

def hurst(series,beta=2,window=300,lag=50):
    return series.rolling(window=window).apply(lambda x:HurstExponent(x,beta,lag))
