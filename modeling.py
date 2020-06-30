import pandas,numpy
import plotly.graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots

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
