# distutils: language=c++

# distutils: language=c++

from collections.abc import Iterable
from pkgutil import iter_modules
from functools import cached_property
from functools import lru_cache
import pandas as pd
import numpy as np
import importlib
import datetime
import string
import random
import base64
import socket
import html
import time
import math
import json
import gzip
import sys
import re
import os
import finlab
from finlab import get_token
from finlab import data
from finlab import ffn_core
from finlab.utils import logger, requests
from finlab.market import Market
from finlab.utils import get_tmp_dir
from finlab.core.dashboard import generate_html
from finlab.core.metrics import Metrics
cimport numpy as np
cimport cython
import pickle

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            # Module mapping
            module_mapping = {
            'finlab.market_info': 'finlab.markets.tw'
            }

            # Class mapping
            class_mapping = {
            'TWMarketInfo': 'TWMarket'
            }

            # Remap module if needed
            mapped_module = module_mapping.get(module, module)

            # Remap class name if needed
            mapped_name = class_mapping.get(name, name)

            return super().find_class(mapped_module, mapped_name)
        except Exception as e:
            print(f"Error finding class: module={module}, name={name}")
            print(f"Mapped module: {mapped_module}, Mapped name: {mapped_name}")
            raise e

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef query_data_int64(np.ndarray df, np.ndarray[np.int64_t, ndim=1] idx_d, np.ndarray[np.int64_t, ndim=1] idx_s):
    cdef int n = len(idx_d)
    cdef list result = [None] * n
    cdef int i
    cdef np.int64_t s, d


    cdef default_value

    if df.dtype == np.float64:
        default_value = np.nan
    elif df.dtype == np.int64:
        default_value = -1
    elif df.dtype == np.int8:
        default_value = -1
    elif df.dtype == np.bool_:
        default_value = False
    elif df.dtype == np.object_:  # Assuming str type
        default_value = None
    else:
        raise ValueError("Unsupported dtype")

    for i in range(n):
        s, d = idx_s[i], idx_d[i]
        result[i] = default_value if s == -1 else df[d, s]

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef query_data_int32(np.ndarray df, np.ndarray[np.int32_t, ndim=1] idx_d, np.ndarray[np.int32_t, ndim=1] idx_s):
    cdef int n = len(idx_d)
    cdef list result = [None] * n
    cdef int i
    cdef np.int64_t s, d


    cdef default_value

    if df.dtype == np.float64:
        default_value = np.nan
    elif df.dtype == np.int64:
        default_value = -1
    elif df.dtype == np.int8:
        default_value = -1
    elif df.dtype == np.bool_:
        default_value = False
    elif df.dtype == np.object_:  # Assuming str type
        default_value = None
    else:
        raise ValueError("Unsupported dtype")

    for i in range(n):
        s, d = idx_s[i], idx_d[i]
        result[i] = default_value if s == -1 else df[d, s]

    return result

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(CustomEncoder, self).default(obj)


def is_in_vscode():
    for k in os.environ.keys():
        if 'VSCODE' in k:
            return True
    return False

daily_return = lambda s: s.resample('1d').last().dropna().pct_change()

def safe_division(n, d):
    return n / d if d else 0

calc_cagr = (
    lambda s: (s.add(1).prod()) ** safe_division(365.25, (s.index[-1] - s.index[0]).days) - 1
    if len(s) > 1 else 0)


server_port = None

def find_latest_file(directory, pattern):
    candidates = [f for f in os.listdir(directory) if re.match(pattern, f)]
    if not candidates:
        raise FileNotFoundError(f"No file found matching pattern '{pattern}'")
    return max(candidates, key=lambda f: os.path.getctime(os.path.join(directory, f)))


def start_server():

    import http.server
    import socketserver
    import threading

    # get a random port
    port = 8000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    if server_port is not None:
        return

    Handler = http.server.SimpleHTTPRequestHandler

    def run_server():

        global server_port
        os.chdir(get_tmp_dir())
        with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
            server_port = port
            httpd.serve_forever()

    server_thread = threading.Thread(target=run_server)
    server_thread.start()


def check_environment():
    env = os.environ

    if 'COLAB_GPU' in env:
        return 'Google Colab'

    if 'VSCODE_PID' in env:
        return 'VSCode'

    if 'JUPYTERHUB_SERVICE_PREFIX' in env:
        return 'Jupyter Lab'

    if 'JPY_PARENT_PID' in env:
        return 'Jupyter Notebook'

    return 'Unknown'

class Report(object):

    def __init__(self, creturn, position, fee_ratio, tax_ratio, trade_at, next_trading_date, market):
        # cumulative return
        self.creturn = creturn
        self.daily_creturn = self.creturn.resample('1d').last().dropna().ffill().rebase()

        # benchmark return
        self.benchmark = market.get_benchmark()
        if isinstance(self.benchmark, pd.Series) and self.benchmark.index.tz is not None:
            self.benchmark.index = pd.Series(self.benchmark.index).dt.tz_convert(position.index.tz)
        if len(self.benchmark) == 0:
            self.benchmark = pd.Series(1, index=self.creturn.index)

        self.daily_benchmark = self.benchmark\
        .dropna().reindex(self.daily_creturn.index, method='ffill') \
        .ffill().rebase()

        # position
        self.position = position
        self.fee_ratio = fee_ratio
        self.tax_ratio = tax_ratio
        self.trade_at = trade_at
        self.update_date = position.index[-1] if len(position) > 0 else datetime.datetime.now()
        self.asset_type = 'tw_stock' if (
        position.columns.str.find('USDT') == -1).all() else 'crypto'

        position_changed = position.diff().abs().sum(axis=1)
        self.last_trading_date = position_changed[position_changed != 0].index[-1] \
        if (position_changed != 0).sum() != 0 else \
        position.index[0] if len(position) > 0 else datetime.datetime.now()

        self.next_trading_date = next_trading_date
        self.market = market
        self.weights = None
        self.next_weights = None
        self.actions = None

        # Initialize additional attributes that Metrics class needs
        self.trades = pd.DataFrame()
        self.live_performance_start = None
        self.stop_loss = None
        self.take_profit = None
        self.trail_stop = None

    def _repr_html_(self):
        return generate_html(self, with_iframe=True)

    @cached_property
    def metrics(self):
        """Get the Metrics instance for accessing individual metrics.


        Metrics: Metrics instance providing individual metric methods
        """
        return Metrics(self)

    def display(self, lagacy=False, save_report_path=None):

        """








        """

        if save_report_path is not None:
            self.to_html(save_report_path)
            # show deprecated warning

        if not lagacy:
            iframe_code = generate_html(self, with_iframe=True)

            from IPython.display import HTML, display
            display(HTML(iframe_code))

            environment = check_environment()

            if environment == 'Google Colab':
                # cancel cell scrolling
                try:
                    from google.colab import output
                    output.no_vertical_scroll()
                except:
                    logger.warning('Cannot cancel cell scrolling')
        else:
            if save_report_path is not None:
                logger.warning('save_report_path is not supported in lagacy mode.')

            logger.warning('lagacy mode will be deprecated')

            if self.benchmark is not None:
                performance = pd.DataFrame({
                    'strategy': self.daily_creturn,
                    'benchmark': self.daily_benchmark.reindex(self.daily_creturn.index, method='ffill')}).dropna().rebase()
            else:
                performance = pd.DataFrame({
                    'strategy': self.creturn}).dropna().rebase()

        fig = self.create_performance_figure(
        performance, (self.position != 0).sum(axis=1))

        stats = self.get_stats()
        sharpe = stats['daily_sharpe'] if stats['daily_sharpe'] == stats['daily_sharpe'] else stats['monthly_sharpe']
        imp_stats = pd.Series({
        'annualized_rate_of_return':str(round(stats['cagr']*100, 2))+'%',
        'sharpe': str(round(sharpe, 2)),
        'max_drawdown':str(round(stats['max_drawdown']*100, 2))+'%',
        'win_ratio':str(round(stats['win_ratio']*100, 2))+'%',
        }).to_frame().T
        imp_stats.index = ['']

        yearly_return_fig = self.create_yearly_return_figure(stats['return_table'])
        monthly_return_fig = self.create_monthly_return_figure(stats['return_table'])

        from IPython.display import display

        display(imp_stats)
        display(fig)
        display(yearly_return_fig)
        display(monthly_return_fig)

        if hasattr(self, 'current_trades'):
        display(self.current_trades)
        else:
        if len(self.position) > 0:
        p = self.position.iloc[-1]
        display(p[p != 0])

    @staticmethod
    def create_performance_figure(performance_detail, nstocks):

        from plotly.subplots import make_subplots
        import plotly.graph_objs as go
        # plot performance

    def diff(s, period):
        return (s / s.shift(period) - 1)

        drawdowns = performance_detail.to_drawdown_series()

        fig = go.Figure(make_subplots(
        rows=4, cols=1, shared_xaxes=True, row_heights=[2, 1, 1, 1]))
        fig.add_scatter(x=performance_detail.index, y=performance_detail.strategy / 100 - 1,
        name='strategy', row=1, col=1, legendgroup='performance', fill='tozeroy')
        fig.add_scatter(x=drawdowns.index, y=drawdowns.strategy, name='strategy - drawdown',
        row=2, col=1, legendgroup='drawdown', fill='tozeroy')
        fig.add_scatter(x=performance_detail.index, y=diff(performance_detail.strategy, 20),
        fill='tozeroy', name='strategy - month rolling return',
        row=3, col=1, legendgroup='rolling performance', )

        if 'benchmark' in performance_detail.columns:
        fig.add_scatter(x=performance_detail.index, y=performance_detail.benchmark / 100 - 1,
        name='benchmark', row=1, col=1, legendgroup='performance', line={'color': 'gray'})
        fig.add_scatter(x=drawdowns.index, y=drawdowns.benchmark, name='benchmark - drawdown',
        row=2, col=1, legendgroup='drawdown', line={'color': 'gray'})
        fig.add_scatter(x=performance_detail.index, y=diff(performance_detail.benchmark, 20),
        fill='tozeroy', name='benchmark - month rolling return',
        row=3, col=1, legendgroup='rolling performance', line={'color': 'rgba(0,0,0,0.2)'})

        fig.add_scatter(x=nstocks.index, y=nstocks, row=4,
        col=1, name='nstocks', fill='tozeroy')
        fig.update_layout(legend={'bgcolor': 'rgba(0,0,0,0)'},
        margin=dict(l=60, r=20, t=40, b=20),
        height=600,
        width=800,
        xaxis4=dict(
        rangeselector=dict(
        buttons=list([
        dict(count=1,
        label="1m",
        step="month",
        stepmode="backward"),
        dict(count=6,
        label="6m",
        step="month",
        stepmode="backward"),
        dict(count=1,
        label="YTD",
        step="year",
        stepmode="todate"),
        dict(count=1,
        label="1y",
        step="year",
        stepmode="backward"),
        dict(step="all")
        ]),
        x=0,
        y=1,
        ),
        rangeslider={'visible': True, 'thickness': 0.1},
        type="date",
        ),
        yaxis={'tickformat': ',.0%', },
        yaxis2={'tickformat': ',.0%', },
        yaxis3={'tickformat': ',.0%', },
        )
        return fig


    @staticmethod
    def create_yearly_return_figure(return_table):
        import plotly.express as px
        yearly_return = [round(v['YTD']*1000)/10 for v in return_table.values()]
        fig = px.imshow([yearly_return],
        labels=dict(color="return(%)"),
        x=list([str(k) for k in return_table.keys()]),
        text_auto=True,
        color_continuous_scale='RdBu_r',
        )

        fig.update_traces(
        hovertemplate="<br>".join([
        "year: %{x}",
        "return: %{z}%",

        )

        fig.update_layout(
        height = 120,
        width= 800,
        margin=dict(l=20, r=270, t=40, b=40),
        yaxis={
        'visible': False,
        },
        title={
        'text': 'yearly return',
        'x': 0.025,
        'yanchor': 'top',
        },
        coloraxis_showscale=False,
        coloraxis={'cmid':0}
        )

        return fig

    @staticmethod
    def create_monthly_return_figure(return_table):

        if len(return_table) == 0:
        return None

        import plotly.express as px
        monthly_table = pd.DataFrame(return_table).T
        monthly_table = round(monthly_table*100,1).drop(columns='YTD')

        fig = px.imshow(monthly_table.values,
        labels=dict(x="month", y='year', color="return(%)"),
        x=monthly_table.columns.astype(str),
        y=monthly_table.index.astype(str),
        text_auto=True,
        color_continuous_scale='RdBu_r',

        )

        fig.update_traces(
        hovertemplate="<br>".join([
        "year: %{y}",
        "month: %{x}",


        )

        fig.update_layout(
        height = 550,
        width= 800,
        margin=dict(l=20, r=270, t=40, b=40),
        title={
        'text': 'monthly return',
        'x': 0.025,
        'yanchor': 'top',
        },
        yaxis={
        'side': "right",
        },
        coloraxis_showscale=False,
        coloraxis={'cmid':0}
        )

        return fig

    @lru_cache(maxsize=1)
    def to_json(self):

        # Convert DataFrame to JSON
        json_str = self.trades.to_json(orient='records')

        # Encode JSON string into bytes
        json_bytes = json_str.encode('utf-8')

        # Compress JSON bytes with gzip
        gzip_bytes = gzip.compress(json_bytes)

        # Convert gzip bytes to base64 encoded string for easier storage and transmission
        gzip_b64_str = base64.b64encode(gzip_bytes).decode('utf-8')

        ret = {
        'timestamps': self.daily_creturn.index.astype(str).to_list(),
        'strategy': self.daily_creturn.values.tolist(),
        'benchmark': self.daily_benchmark.values.tolist(),
        'metrics': self.get_metrics(),
        'trades': gzip_b64_str
        }

        return ret

    @lru_cache(maxsize=1)
    def _to_json046(self):

        # Convert DataFrame to JSON
        json_str = self.trades.tail(500).to_json(orient='records')

        # Encode JSON string into bytes
        json_bytes = json_str.encode('utf-8')

        # Compress JSON bytes with gzip
        gzip_bytes = gzip.compress(json_bytes)

        # Convert gzip bytes to base64 encoded string for easier storage and transmission
        gzip_b64_str = base64.b64encode(gzip_bytes).decode('utf-8')

        ret = {
        'metrics': self.get_metrics(),
        'trades': gzip_b64_str
        }

        return ret


    def upload(self, name=None):

        name = os.environ.get(
        'FINLAB_STRATEGY_NAME', name)

        name = os.environ.get(
        'FINLAB_FORCED_STRATEGY_NAME', name) or ''

        head_is_eng = len(re.findall(
        r'[\u0041-\u005a|\u0061-\u007a]', name[0])) > 0
        has_cn = len(re.findall('[\u4e00-\u9fa5]', name[1:])) > 0
        if head_is_eng and has_cn:
        raise Exception('Strategy Name Error: ')
        for c in '()[]+-|!@#$~%^={}&*':
        name = name.replace(c, '_')

        # stats
        stats = self.get_stats()

        # creturn
        creturn = {'time': self.daily_creturn.index.astype(str).to_list(),
        'value': self.daily_creturn.values.tolist()}

        # ndays return
        ndays_return = {d: self.get_ndays_return(
        self.daily_creturn, d) for d in [1, 5, 10, 20, 60]}
        ndays_return_benchmark = {d: self.get_ndays_return(
        self.daily_benchmark, d) for d in [1, 5, 10, 20, 60]}

        d = {
        # backtest info
        'drawdown_details': stats['drawdown_details'],
        'stats': stats,
        'returns': creturn,
        'benchmark': self.daily_benchmark.values.tolist(),
        'ndays_return': ndays_return,
        'ndays_return_benchmark': ndays_return_benchmark,
        'return_table': stats['return_table'],
        'fee_ratio': self.fee_ratio,
        'tax_ratio': self.tax_ratio,
        'trade_at': self.trade_at if isinstance(self.trade_at, str) else 'open',
        'timestamp_name': self.market.get_name(),
        'freq': self.market.get_freq(),

        # dates
        'update_date': self.update_date.isoformat(),
        'next_trading_date': self.next_trading_date.isoformat(),

        # key data
        'position': self.position_info(),
        'position2': self.position_info2(),

        # live performance
        'live_performance_start': self.live_performance_start.isoformat() if self.live_performance_start else None,
        'stop_loss': self.stop_loss,
        'take_profit': self.take_profit,
        **self._to_json046()
        }

    def upload():

        payload = {'data': json.dumps(d, cls=CustomEncoder),
        'api_token': get_token(),
        'collection': 'strategies',
        'document_id': name}

        result = requests.post(
        'https://asia-east2-fdata-299302.cloudfunctions.net/write_database', data=payload).text

        # python is in website backtest
        if 'FINLAB_FORCED_STRATEGY_NAME' in os.environ:
        return {'status': 'success'}

        # create iframe
        try:
        result = json.loads(result)
        except:
        return {'status': 'error', 'message': 'cannot parse json object'}

        if 'status' in result and result['status'] == 'error':
        print('Fail to upload result to server')
        print('error message', result['message'])
        return {'status': 'error', 'message': result['status']}

        return True

        return upload()

    @lru_cache(maxsize=1)
    def calculate_current_trades(self):
        # calculate self.current_trades
        # find trade without end or end today
        maxday = max(self.trades.entry_sig_date.max(), self.trades.exit_sig_date.max())
        latest_entry_day = self.trades.entry_sig_date[self.trades.entry_date.notna()].max()
        self.current_trades = self.trades[
        (self.trades.entry_sig_date == maxday )
        | (self.trades.exit_sig_date == maxday)
        | (self.trades.exit_sig_date >= latest_entry_day) # for the case of sl_enter, tp_enter
        | (self.trades.entry_sig_date == latest_entry_day)
        | (self.trades.exit_sig_date.isnull())]\
        .set_index('stock_id')# \

        # cannot drop duplicates, because the same stock can be traded multiple times

        # .pipe(lambda df: df[~df.index.duplicated(keep='last')])

        self.current_trades.loc[self.current_trades['return'].isna(), 'trade_price@entry_date'] = np.nan
        self.current_trades.loc[self.current_trades['return'].isna(), 'trade_price@exit_date'] = np.nan

        # self.next_trading_date = max(self.current_trades.entry_sig_date.max(), self.current_trades.exit_sig_date.max())

        self.current_trades['weight'] = 0
        if len(self.weights) != 0:
        self.current_trades['weight'] = self.weights.reindex(self.current_trades.index).fillna(0)

        self.current_trades['next_weights'] = 0
        if len(self.next_weights) != 0:
        self.current_trades['next_weights'] = self.next_weights.reindex(self.current_trades.index).fillna(0)

    def format_resample(self):
        if isinstance(self.resample, str):
        return self.resample

        data = self.resample
        #  pd.DatetimeIndex 1e9
        if isinstance(data, pd.DatetimeIndex):
        ts_list = data.astype('int64') / 1e9
        return ts_list.tolist()
        #  datetime.datetime  list timestamp()
        elif isinstance(data, list) and all(isinstance(x, datetime) for x in data):
        return [x.timestamp() for x in data]
        raise TypeError(" pd.DatetimeIndex  datetime.datetime  list")


    @lru_cache(maxsize=1)
    def position_info2(self):

        if not hasattr(self, 'current_trades'):
        return {
        "positions": [],
        "positionConfig": {
        "isDailyStrategy": 1,
        "market": self.market.get_name(),
        "sl": self.stop_loss,
        "tp": self.take_profit,
        "ts": self.trail_stop,
        "resample": self.resample if isinstance(self.resample, str) else None,
        "entryTradePrice": self.trade_at,
        "exitTradePrice": self.trade_at,
        "dataFreq": self.market.get_freq(),
        'currentRebalanceDate': self.weights.name.timestamp() \
        if self.weights is not None and self.weights.name and isinstance(self.weights.name, pd.Timestamp) else None,
        'nextRebalanceDate': self.next_weights.name.timestamp() \
        if self.next_weights is not None and self.next_weights.name and isinstance(self.next_weights.name, pd.Timestamp) else None,
        }
        }
        actions = []

        for idx, row in self.current_trades.iterrows():
        is_entry = row['entry_sig_date'] == self.creturn.index[-1]
        is_future_entry = row['entry_sig_date'] > self.creturn.index[-1]

        # exit today or before today
        is_exit = row['exit_sig_date'] == self.creturn.index[-1]
        is_past_exit = row['exit_sig_date'] < self.creturn.index[-1]

        # exit in futures (hold)
        is_hold = (row['exit_sig_date'] > self.creturn.index[-1]) | (row['exit_sig_date'] != row['exit_sig_date'])

        type_ = 'hold'

        if is_entry:
        type_ = 'entry'
        elif is_future_entry:
        type_ = 'entry_f'
        elif is_exit:
        type_ = 'exit'
        elif is_past_exit:
        type_ = 'exit_p'
        elif is_hold:
        type_ = 'hold'
        else:
        print(f'There is a strange type for stock {idx} {row}')

        if type_ == 'exit' or type_ == 'exit_p':
        reason = self.actions.loc[idx] if idx in self.actions.index else '_'

        # handle the case of exit but still hold
        # that means the exit signal only for the record
        # of the periodical balance, not for the real exit
        if reason == 'hold':
        reason = '_'
        else:
        reason = '_'

        date = row['exit_sig_date']
        if date != date:
        date = row['entry_sig_date']
        profit = row['return']
        if profit != profit:
        profit = 0
        actions.append({
        'type': type_,
        'reason': reason,
        'date': date.isoformat(),
        'profit': profit
        })

        adj_close = self.market.get_price('close', adj=True)
        close = self.market.get_price('close', adj=False)
        cmin = adj_close.iloc[-20:].min()
        cmax = adj_close.iloc[-20:].max()
        rsv20 = (adj_close.iloc[-1] - cmin) / (cmax - cmin)

    def get_adj_close(row):
        if pd.isnull(row.entry_date):
        return np.nan
        return adj_close.loc[row.entry_date, row.get('stock_id', row.name).split(' ')[0]]

        to_date_string = lambda d: d.isoformat() if d == d else None

        trades = self.current_trades.copy()
        positions = pd.DataFrame({
        'assetName': trades.index.str.split(' ').str[1].fillna(''),
        'assetId':trades.index.str.split(' ').str[0],
        'entryDate':trades['entry_date'].apply(to_date_string),
        'entrySigDate':trades['entry_sig_date'].apply(to_date_string),
        'exitDate':trades['exit_date'].apply(to_date_string),
        'exitSigDate':trades['exit_sig_date'].apply(to_date_string),
        'entryPrice':trades['trade_price@entry_date'],
        'entryAdjPrice':trades.apply(get_adj_close, axis=1),
        'exitPrice':trades['trade_price@exit_date'],
        'currentPrice':trades.index.str.split(' ').str[0].map(close.iloc[-1]),
        'profit':trades['return'],
        'currentWeight':trades['weight'],
        'nextWeight':trades['next_weights'],
        'rsv20': trades.index.str.split(' ').str[0].map(rsv20),
        'action': actions,
        'industry': trades['industry'],
        }).to_dict(orient='records')

        position_config = {
        # isDailyStrategy: False,
        "isDailyStrategy": 1 if ((self.position.index.hour == 0) & (self.position.index.minute == 0) & (self.position.index.second == 0)).all() else 0,
        "sl": self.stop_loss,
        "tp": self.take_profit,
        "resample": self.resample if isinstance(self.resample, str) else None,
        "entryTradePrice": self.trade_at,
        "exitTradePrice": self.trade_at,
        'currentRebalanceDate': self.weights.name.timestamp() \
        if self.weights is not None and self.weights.name and isinstance(self.weights.name, pd.Timestamp) else None,
        'nextRebalanceDate': self.next_weights.name.timestamp() \
        if self.next_weights is not None and self.next_weights.name and isinstance(self.next_weights.name, pd.Timestamp) else None,
        "scheduled": self.next_trading_date.isoformat(),
        "dataFreq": self.market.get_freq(),
        "created": datetime.datetime.now().isoformat(),
        "lastTimestamp": self.creturn.index[-1].isoformat() if len(self.creturn) > 0 else None,
        }

        return {
        "positions": positions,
        "positionConfig": position_config,
        }

    @lru_cache(maxsize=1)
    def position_info(self):

        if not hasattr(self, 'current_trades'):
        return pd.DataFrame(columns=['status', 'weight', 'next_weight',
        'entry_date', 'exit_date', 'return', 'entry_price'])\
        .to_dict('index')

        current_trades = self.current_trades

        # default_status = pd.Series('hold', index=current_trades.index)


        #     default_status.loc[current_trades.exit_sig_date.isnull()] = 'hold'
        #     default_status.loc[current_trades.exit_sig_date.notna()] = 'exit'
        default_status = pd.Series('hold', index=current_trades.index)
        default_status = default_status.where(current_trades.exit_sig_date.isnull(), 'exit')

        if self.resample is None:
        default_status = default_status.where(current_trades.exit_sig_date.isnull(), 'exit')

        trade_at = self.trade_at if isinstance(self.trade_at, str) else 'close'
        status = self.actions.reindex(current_trades.index).fillna(default_status)

        entry_date = current_trades.entry_sig_date.apply(lambda d: d.isoformat() if d else '')
        exit_date = current_trades.exit_sig_date.apply(lambda d: d.isoformat() if d else '')
        ret = pd.DataFrame({
        'status': status,
        'weight': current_trades.weight,
        'next_weight': current_trades.next_weights,
        'entry_date': entry_date,
        'exit_date': exit_date,
        'latest_sig_date': pd.DataFrame({'entry': entry_date, 'exit': exit_date}).max(axis=1).values,
        'return': current_trades['return'].fillna(0),
        'entry_price': current_trades['trade_price@entry_date'].fillna(0),
        }, index=current_trades.index)

        # ret['latest_sig_date'] = pd.DataFrame({'entry': ret.entry_date, 'exit': ret.exit_date}).max(axis=1)
        ret = ret.sort_values('latest_sig_date').groupby(level=0).last()
        ret = ret.drop(columns='latest_sig_date')

        ret = ret.to_dict('index')

        ret['update_date'] = self.update_date.isoformat()
        ret['next_trading_date'] = self.next_trading_date.isoformat()
        ret['trade_at'] = trade_at
        ret['freq'] = self.market.get_freq()
        ret['market'] = self.market.get_name()
        ret['stop_loss'] = self.stop_loss
        ret['take_profit'] = self.take_profit

        return ret

    @staticmethod
    def get_ndays_return(creturn, n):
        last_date_eq = creturn.iloc[-1]
        ref_date_eq = creturn.iloc[max(-1 - n, -len(creturn))]
        return last_date_eq / ref_date_eq - 1

    def add_trade_info(self, name, df, date_col='entry_sig_date'):

        """





        """

        if isinstance(date_col, str):
        date_col = [date_col]

        combined_dates = set().union(*[set(self.trades[d]) for d in date_col])
        df_temp = df.reindex(df.index.union(combined_dates), method='ffill')

        for date_name in date_col:
        dates = self.trades[date_name]
        stocks = self.trades['stock_id'].str.split(' ').str[0]

        idx_d = df_temp.index.get_indexer_for(dates)
        idx_s = df_temp.columns.get_indexer_for(stocks)

        # self.trades[f"{name}@{date_name}"] = [

        #     for s, d in zip(idx_s, idx_d)]

        if idx_d.dtype == np.int64:
        self.trades[f"{name}@{date_name}"] = query_data_int64(df_temp.values, idx_d, idx_s)
        elif idx_d.dtype == np.int32:
        self.trades[f"{name}@{date_name}"] = query_data_int32(df_temp.values, idx_d, idx_s)
        else:
        raise Exception(f'Unknown dtype {idx_d.dtype} for query_date')


    def remove_trade_info(self, name):
        cs = [c for c in self.columns if c != name]
        self.trades = self.trades[cs]

    def get_mae_mfe(self):
        return self.mae_mfe

    def get_trades(self):
        return self.trades

    def is_rebalance_due(self):

        market = self.market

        next_trading_time = market.market_close_at_timestamp(
        self.next_weights.name)

        # check if next_trading_time is tz-aware
        now = datetime.datetime.now() if next_trading_time.tzinfo is None \
        else datetime.datetime.now().astimezone()

        if isinstance(self.resample, str):
        return now >= next_trading_time

        # TODO: remove this when 1.2.23 is released
        old_cloud_report = self.weights.name == 'weight'
        if old_cloud_report:
        last_market_timestamp = market.market_close_at_timestamp(market.get_price('close').index[-1])
        return (now > next_trading_time) and next_trading_time == last_market_timestamp

        # when resample==None, there is no future next_trading_date

        # when next weight is not the same as current weight
        # that means the rebalance is due
        return self.weights.name != self.next_weights.name

    def is_stop_triggered(self):
        return self.actions.isin(['sl', 'tp']).any()

    @lru_cache(maxsize=1)
    def get_stats(self, resample='1d', riskfree_rate=0.02):

        stats = self.daily_creturn.calc_stats()
        stats.set_riskfree_rate(riskfree_rate)

        # calculate win ratio
        ret = stats.stats.to_dict()
        ret['start'] = ret['start'].strftime('%Y-%m-%d')
        ret['end'] = ret['end'].strftime('%Y-%m-%d')
        ret['version'] = finlab.__version__

        trades = self.trades.dropna()
        ret['win_ratio'] = sum(trades['return'] > 0) / len(trades) if len(trades) != 0 else 0
        ret['return_table'] = stats.return_table.transpose().to_dict()
        # ret['mae_mfe'] = self.run_analysis("MaeMfe", display=False)
        # ret['liquidity'] = self.run_analysis("Liquidity", display=False)



        # todo old remove
        drawdown = self.run_analysis("Drawdown", display=False)
        ret['drawdown_details'] = drawdown['strategy']['largest_drawdown']
        return ret

    @lru_cache(maxsize=1)
    def get_metrics(self, stats_=None, riskfree_rate=0.02):

        """Get the metrics of the backtest result.



















































        # Use the new Metrics class for calculations
        metrics = self.metrics

        # Add market value to trade info if available
        mv = self.market.get_market_value()
        if len(mv) != 0:
        self.add_trade_info('market_value', mv, ['entry_date'])

        return {
        "backtest": {
        "startDate": metrics.start_date(),
        "endDate": metrics.end_date(),
        "version": metrics.version(),
        'feeRatio': metrics.fee_ratio(),
        'taxRatio': metrics.tax_ratio(),
        'tradeAt': metrics.trade_at(),
        'market': metrics.market(),
        'freq': metrics.freq(),
        'expired': metrics.expired(),

        # dates
        'updateDate': metrics.update_date(),
        'nextTradingDate': metrics.next_trading_date(),
        'currentRebalanceDate': metrics.current_rebalance_date(),
        'nextRebalanceDate': metrics.next_rebalance_date(),

        # live performance
        'livePerformanceStart': metrics.live_performance_start(),
        'stopLoss': metrics.stop_loss(),
        'takeProfit': metrics.take_profit(),
        'trailStop': metrics.trail_stop(),
        },

        "profitability": {
        "annualReturn": metrics.annual_return(riskfree_rate),
        "alpha": metrics.alpha(),
        "beta": metrics.beta(),
        "avgNStock": metrics.avg_n_stock(),
        "maxNStock": metrics.max_n_stock(),
        },

        "risk": {
        "maxDrawdown": metrics.max_drawdown(riskfree_rate),
        "avgDrawdown": metrics.avg_drawdown(riskfree_rate),
        "avgDrawdownDays": metrics.avg_drawdown_days(riskfree_rate),
        "valueAtRisk": metrics.value_at_risk(),
        "cvalueAtRisk": metrics.cvalue_at_risk()
        },

        "ratio": {
        "sharpeRatio": metrics.sharpe_ratio(riskfree_rate),
        "sortinoRatio": metrics.sortino_ratio(riskfree_rate),
        "calmarRatio": metrics.calmar_ratio(riskfree_rate),
        "volatility": metrics.volatility(riskfree_rate),
        "profitFactor": metrics.profit_factor(),
        "tailRatio": metrics.tail_ratio(),
        },

        "winrate": {
        "winRate": metrics.win_rate(),
        "m12WinRate": metrics.m12_win_rate(riskfree_rate),
        "expectancy": metrics.expectancy(),
        "mae": metrics.mae(),
        "mfe": metrics.mfe(),
        },

        "liquidity": {
        "capacity": metrics.capacity(),
        "disposalStockRatio": metrics.disposal_stock_ratio(),
        "warningStockRatio": metrics.warning_stock_ratio(),
        "fullDeliveryStockRatio": metrics.full_delivery_stock_ratio(),
        "buyHigh": metrics.buy_high(),
        "sellLow": metrics.sell_low(),
        }
        }

    @cached_property
    def liquidity(self):
        ret = self.run_analysis('liquidity', display=False)
        return ret


    def run_analysis(self, analysis, display=True, **kwargs):

        # get the instance of analysis
        if isinstance(analysis, str):

        if analysis[-8:] != 'Analysis':
        analysis += 'Analysis'

        # get module
        module_name = 'finlab.analysis.' + analysis[0].lower() + analysis[1:]

        if importlib.util.find_spec(module_name) is None:
        import finlab.analysis as module
        submodules = []
        for submodule in iter_modules(module.__path__):
        if '_' not in submodule.name:
        submodules.append(submodule.name[:-8:])

        error = f"Cannot find {module_name}. Possible candidates are " + str(submodules)[1:-1]
        raise Exception(error)

        analysis_module = importlib.import_module(module_name)

        # create an instance from module
        analysis_class = analysis[0].upper() + analysis[1:]

        analysis = getattr(analysis_module, analysis_class)(**kwargs)

        # calculate additional trade info for analysis
        additional_trade_info = analysis.calculate_trade_info(self)
        for v in additional_trade_info:
        self.add_trade_info(*v)

        # analysis and return figure or data as result
        result = analysis.analyze(self)

        if display:
        return analysis.display()

        return result

    def display_mae_mfe_analysis(self, **kwargs):
        return self.run_analysis("MaeMfeAnalysis", **kwargs)

    def to_pickle(self, file_path):
        """Serialize the object to a pickle file."""
        with open(file_path, 'wb') as file:
        pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, file_path):
        """Deserialize the object from a pickle file."""
        with open(file_path, 'rb') as file:
        unpickler = CustomUnpickler(file)
        return unpickler.load()

    def to_html(self, path):
        html_text = generate_html(self, with_iframe=False)
        with open(path, 'w', encoding='utf-8') as f:
        f.write(html_text)

    def to_text(self, name=None):

        # Extract stock symbols
        w_return = self.creturn.iloc[-1] / self.creturn.iloc[-6] - 1
        m_return = self.creturn.iloc[-1] / self.creturn.iloc[-21] - 1
        date = self.next_trading_date

        datestr = date.strftime('%Y-%m-%d')

        if self.actions and self.actions.isin(['sl', 'tp']).any():
        removed_stocks = self.actions[self.actions.isin(['sl', 'tp'])].index.to_list()
        new_stocks = []
        if self.weights:
        hold_stocks = [s for s in self.weights.index.to_list() if s not in removed_stocks]
        else:
        hold_stocks = []
        elif self.actions:
        removed_stocks = self.actions[self.actions.isin(['exit'])].index.to_list()
        new_stocks = self.actions[self.actions.isin(['enter'])].index.to_list()
        hold_stocks = self.actions[self.actions.isin(['hold'])].index.to_list()

        icons = {
        "rise":'',
        "fall":'',
        }

        # Format text
        self = " : " + name + "\n" + \
        f" : {datestr} " + "\n\n" + \
        " " + "\n" + \
        "".join(["   " + symbol + "\n" for symbol in removed_stocks]) + \
        "".join(["   " + symbol + "\n" for symbol in new_stocks]) + \
        "".join(["   " + symbol + "\n" for symbol in hold_stocks]) + "\n" + \
        " " + "\n" + \
        " " + icons["rise" if w_return > 0 else "fall"] + " : {:.2%} ".format(w_return) + "\n" + \
        " " + icons["rise" if m_return > 0 else "fall"] + " : {:.2%} ".format(m_return) + "\n(Power by FinLab)"

        return self