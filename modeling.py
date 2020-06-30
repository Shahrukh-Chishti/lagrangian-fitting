import pandas,numpy
import plotly.graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots
import torch
device = 'cpu'

def controlPath(time):
    c = pandas.Series(time,index=time)
    cdot = pandas.Series(1,index=time)
    cddot = pandas.Series(0,index=time)
    return c,cdot,cddot

def logLoss(x,epsilon=e-10):
    return torch.log(x+epsilon)**2

def lossFunction(ND,ELC,ELD):
    loss = logLoss(ND)*logLoss(ELC)*((ELD+epsilon)/(ELC+epsilon))
    return loss


def parametrization(power_space):
    parameters = []
    for power in power_space:
        c = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
        power['c'] = c
        parameters.append(c)
    return power_space,parameters

def trainPolynomial(power_space,q,qdot,qddot,iterations=200):
    time = q.index
    c,cdot,cddot = controlPath(time)
    power_space,parameters = parametrization(power_space)
    ELD = eulerLagrangian(power_space,q,qdot,qddot)
    ELC = eulerLagrangian(power_space,c,cdot,cddot)
    ND = regularizer(power_space,q,qdot,qddot)
    loss = lossFunction(ND,ELC,ELD)
    for iteration in range(iterations):
        loss.backward()


#def regularizer(power_space,q,qdot,qddot):

def EulerLagrangian(power_space,q,qdot,qddot,time):
    parameters = []
    ELD = []
    for t in time:
        EL = []
        for power in power_space:
            alpha,beta,gamma,c = power
            parameters.append(c)
            el = c*(q[t]**alpha)*(qdot[t]**beta)*(qddot[t]*gamma)
            el *= beta*gamma/(qdot[t]*t) + (beta-1)*alpha/q[t] + beta*(beta-1)*qddot[t]/(qdot[t]**2)
            EL.append(el)
        ELD.append(torch.sum(EL))

    ELD = torch.sum(ELD)
    return EL,parameters


## plotting methods

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
