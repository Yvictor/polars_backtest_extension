# cython: language_level=3
# distutils: language = c++
# distutils: extra_compile_args = -O3

import numpy as org_np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool
from libc.math cimport isnan
from cython.operator cimport dereference, postincrement

# Global variables
cdef np.ndarray price = org_np.zeros((0,0), dtype=float)
cdef np.ndarray close = org_np.zeros((0,0), dtype=float)
cdef map[int, int] pos2price
cdef int window
cdef int window_step

cdef np.ndarray entry = org_np.zeros(0, dtype=int)
cdef np.ndarray position = org_np.zeros(0, dtype=float)
cdef np.ndarray entry_trasnaction = org_np.zeros(0, dtype=int)

cdef double fee_ratio
cdef double tax_ratio

cdef int date

cdef void record_date(int d):
  global date
  date = d

cdef void start_analysis(
  np.ndarray[np.float64_t, ndim=2] price_,
  np.ndarray[np.float64_t, ndim=2] close_,
  map[int, int] pos2price_,
  int nstocks, double fee_ratio_, double tax_ratio_,
  int window_, int window_step_):

  global price, close, pos2price, window, window_step
  price = price_
  close = close_
  pos2price = pos2price_
  window = window_
  window_step = window_step_

  global trades, trade_positions, mae_mfe
  trades.clear()
  trade_positions.clear()
  mae_mfe.clear()

  global entry, position, entry_trasnaction
  entry = org_np.zeros(nstocks, dtype=int)
  position = org_np.zeros(nstocks, dtype=float)
  entry_trasnaction = org_np.zeros(nstocks, dtype=int)

  global fee_ratio, tax_ratio
  fee_ratio = fee_ratio_
  tax_ratio = tax_ratio_

cdef end_analysis(map[int, double] &pos):

  global trades, trade_positions, mae_mfe, date

  cdef int sid = 0
  cdef map[int, double].iterator it = pos.begin()

  # Iterate over all positions and record exit for remaining assets
  # When the backtest ends, any remaining positions should be treated as exiting.
  # However, if a position's value is 0, it means the position has already been exited,
  # which will result in a duplicate exit record being created.
  # Therefore, we need to handle this scenario to prevent duplication of exit records.
  while it != pos.end():
    if dereference(it).second != 0:
        sid = dereference(it).first

        # record exit for existing assets
        trades.append([sid, entry[sid], -1])
        trade_positions.append(position[sid])
        mae_mfe.append(mae_mfe_analysis(sid, entry[sid], date, position[sid] > 0, entry_trasnaction[sid], False))

    postincrement(it)


cdef void record_entry(int sid, double position_sid, int has_entry_trasnaction):
  global entry, position, date, entry_trasnaction
  entry[sid] = date
  position[sid] = position_sid
  entry_trasnaction[sid] = has_entry_trasnaction

cdef void record_exit(int sid, int has_exit_trasnaction):

  global trades, trade_positions, mae_mfe, entry, window, date

  trades.append([sid, entry[sid], date])
  trade_positions.append(position[sid])
  mae_mfe.append(mae_mfe_analysis(sid, entry[sid], date, position[sid] > 0, entry_trasnaction[sid], has_exit_trasnaction))

cdef np.ndarray mae_mfe_analysis(int sid, int i_entry, int i_exit,  is_long, int i_entry_trasnaction, int i_exit_trasnaction):

#   print(sid, i_entry, i_exit, '-------------------')

  global price, close, pos2price, window, window_step, entry, date, fee_ratio, tax_ratio

  cdef int pid = pos2price[sid]

  cdef double price_ratio = 1

  cdef vector[double] cummax, cummin, mdd, profit_period, returns
  cdef vector[int] cummin_i

  cdef int has_entry_transaction = i_entry_trasnaction
  cdef int has_exit_transaction = i_exit_trasnaction

  i_exit_max = i_exit
  if window + i_entry > i_exit_max:
    i_exit_max = window + i_entry

#   print('i_exit_max 1', i_exit_max)

  cdef int plength = (<object> price).shape[0]

  if i_exit_max >= plength:
    i_exit_max = plength-1

  cummax.reserve(i_exit_max - i_entry + 1)
  cummin.reserve(i_exit_max - i_entry + 1)
  cummin_i.reserve(i_exit_max - i_entry + 1)
  mdd.reserve(i_exit_max - i_entry + 1)
  profit_period.reserve(i_exit_max - i_entry + 1)
  returns.reserve(i_exit_max - i_entry + 1)

  cdef double entry_price = price[i_entry][pid]
  cdef double entry_close = close[i_entry][pid]
  price_ratio = entry_close / entry_price if is_long else 2 - entry_close / entry_price

  if has_entry_transaction:
    price_ratio *= (1 - fee_ratio)

  returns.push_back(price_ratio)
  cummax.push_back(max(price_ratio, 1))
  cummin.push_back(min(price_ratio, 1))
  cummin_i.push_back(0)
  mdd.push_back(min(0, price_ratio - 1))
  profit_period.push_back(1 if price_ratio > 1 else 0)


  cdef double v = 1
  cdef double pv = close[i_entry][pid]
  cdef double p = close[i_entry][pid]
  cdef int i = 0

  for i, ith in enumerate(range(i_entry+1, i_exit_max+1)):

    p = close[ith][pid]

    if not isnan(p):
      v = p / pv # if is_long else pv / p
      pv = p

      if not isnan(v):
        if is_long:
          price_ratio *= v
        else:
          price_ratio = 2 - (2 - price_ratio) * v

    cmax = cummax[i]
    cmin = cummin[i]
    if price_ratio > cmax:
      cummax.push_back(price_ratio)
      cmax = price_ratio
    else:
      cummax.push_back(cmax)

    if price_ratio < cmin:
      cummin.push_back(price_ratio)
      cummin_i.push_back(i)
    else:
      cummin.push_back(cmin)
      cummin_i.push_back(cummin_i[i])

    newmdd = price_ratio / cmax - 1
    if newmdd < mdd[i]:
        mdd.push_back(newmdd)
    else:
        mdd.push_back(mdd[i])

    profit_period.push_back(profit_period[i] + (price_ratio > 1))
    returns.push_back(price_ratio)

  # print('original return', returns)

  if has_exit_transaction:

    if i_entry != i_exit_max:
      pv = price[ith][pid]
      p = close[ith][pid]

      # use trade price to calculate final return instead of close price
      if is_long:
        returns[len(returns)-1] *= pv/p
      else:
        returns[len(returns)-1] = 2 - (2 - returns[len(returns)-1]) * pv/p

    # apply fee and tax
    returns[len(returns)-1] *= (1 - fee_ratio - tax_ratio)
    # print('adjust for fee and tax', (1 - fee_ratio - tax_ratio))

  # print('i_exit_max', i_exit_max)
  # print('adjusted return', returns)

  # print('i_exit_max', i_exit_max)
  # print('i_entry', i_entry)
  # print('i_exit', i_exit)

  # print('cummax', cummax)
  # print('cummin', cummin)
  # print('mdd', mdd)
  # print('returns', returns)
  # plt.plot(cummax, label='cummax')
  # plt.plot(cummin, label='cummin')
  # plt.plot(mdd, label='mdd')
  # plt.legend()
  # plt.show()

  cdef int arsize = ((window-1) // window_step + 2) * 6
  cdef np.ndarray[np.float64_t, ndim=1] ret = org_np.empty(arsize)
  ret.fill(-1)

  # maes = []
  # gmfes = []
  # bmfes = []

  i = 0
  for w in range(0, min(cummax.size(), window), window_step):
    mae = cummin[w] - 1
    gmfe = cummax[w] - 1

    mae_i = cummin_i[w]
    bmfe = cummax[mae_i] - 1
    ret[i] = mae
    i+=1
    ret[i] = gmfe
    i+=1
    ret[i] = bmfe
    i+=1
    ret[i] = mdd[w]
    i+=1
    ret[i] = profit_period[w]
    i+=1
    ret[i] = returns[w] - 1
    i+=1
    # maes.append(mae)
    # gmfes.append(gmfe)
    # bmfes.append(bmfe)

  # plt.plot(maes, label='mae')
  # plt.plot(gmfes, label='gmfe')
  # plt.plot(bmfes, label='bmfe')
  # plt.legend()
  # plt.show()

  w = min(i_exit - i_entry, cummax.size()-1)
  mae = cummin[w] - 1
  gmfe = cummax[w] - 1
  mae_i = cummin_i[w]

  bmfe = cummax[mae_i] - 1

  ret_length = len(ret)

  ret[ret_length-6] = mae
  ret[ret_length-5] = gmfe
  ret[ret_length-4] = bmfe
  ret[ret_length-3] = mdd[w]
  ret[ret_length-2] = profit_period[w]
  ret[ret_length-1] = returns[w] - 1

  return ret

# Module-level lists (initialized at module import)
trades = []
trade_positions = []
mae_mfe = []
