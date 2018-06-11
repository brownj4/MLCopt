import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
import itertools
import sys
import matplotlib.pyplot as plt

# length of stored fixed point number in bits
num_bits = 4

# number of cells used to store the fixed point number
num_cells = 2

# standard deviation of storage error as ratio of cell width
sigma = .1

# cumulative of storage error distribution
def cum_error_dist(x):
  return .5 * (1 + erf(x / (sigma * np.sqrt(2))))

# returns how bad a given error is given both stored and retrieved values
def error_weight(stored, retrieved):
  return abs(retrieved - stored)

# returns how likely a given value is to be stored
def usage_prob(stored):
  return 1. / (2 ** num_bits)

# calculate expected error for a given Spock layout
def calc_error(x_full, level_vals, offsets):
  error_probs = []
  errors = []
  # precalculate errors and their probabilities for each cell
  for cell in range(num_cells):
    # extract this cell's variables from the optimization vector
    x = x_full[offsets[cell]:offsets[cell+1]]
    # extract the interleaved levels and cutoffs
    levels = x[::2]
    cutoffs = [-np.inf] + list(x[1::2]) + [np.inf]
    num_levels = len(levels)
    errors.append([])
    # iterate over all possible values stored in this cell
    for (stored_level_index, stored_level) in enumerate(levels):
      stored_val = level_vals[cell][stored_level_index]
      # iterate over all possible values retrieved from this cell
      for (retrieved_level_index, retrieved_level) in enumerate(levels):
        retrieved_val = level_vals[cell][retrieved_level_index]
        cutoff_start = cutoffs[retrieved_level_index]
        cutoff_end = cutoffs[retrieved_level_index+1]
        # calculate the probability of reading retrieved_val when stored_val was stored
        prob = cum_error_dist(cutoff_end - stored_level) - \
               cum_error_dist(cutoff_start - stored_level)
        errors[-1].append((np.log(max(prob,0)), stored_val, retrieved_val))
  errors = np.array(errors)
  # calculate total weighted error for all of the cells combined
  expected_error = 0
  for error in itertools.product(*errors):
    # multiply probabilities (adding in log) and add up stored/retrieved values
    (log_prob, stored, retrieved) = np.sum(error, axis=0)
    # add the weighted contribution of this combination of errors
    expected_error += np.exp(log_prob) * error_weight(stored, retrieved) * usage_prob(stored)
  # print(expected_error)
  return expected_error

# optimize a set of cells for a given allocation of bits
def opt_cell(bit_orders):
  x0 = []
  cons = []
  offsets = [0]
  level_vals = []
  for cell in range(num_cells):
    num_levels = 2**len(bit_orders[cell])
    # the number of optimization variables is the number of levels plus the number of cutoffs
    num_vars = 2*num_levels-1
    # provide evenly spaced initial values for levels and cutoffs
    x0 += [i / float(2 * (num_levels - 1)) for i in range(num_vars)]
    # add constraints limiting the first and last levels to 0 and 1 respectively
    cons += [{'type': 'ineq', 'fun': lambda x,o=offsets[cell]: x[o]},
             {'type': 'ineq', 'fun': lambda x,o=offsets[cell]: 1-x[o-1]}]
    # add constraints to keep levels in ascending order and cutoffs in ascending order
    cons += [{'type': 'ineq', 'fun': lambda x,o=offsets[cell],i=i: x[o+i+2] - x[o+i]} for i in range(num_vars-2)]
    # optional symmetry constraint for possible speedup (only valid for certain distributions)
    # cons += [{'type': 'eq', 'fun': lambda x,o=offsets[cell],i=i,n=num_vars: x[o+i] + x[o+n-i-1] - 1} for i in range(num_levels)]
    # calculate value each level represents (final output is sum of all cells' values)
    level_vals.append(np.zeros(num_levels))
    for (bit_index, bit_order) in enumerate(bit_orders[cell]):
      for level in range(num_levels):
        level_vals[-1][level] += (2 ** bit_order) * ((level // (2 ** bit_index)) % 2)
    offsets.append(offsets[-1]+num_vars)
  # run the minimization algorithm to find the optimal Spock layouts for this bit allocation
  sol = minimize(calc_error, x0, args=(level_vals, offsets), constraints=cons)
  return calc_error(sol.x, level_vals, offsets), sol.x, offsets

# recursively tries all cell bit allocations to find the one with lowest error
def opt(cells_to_pick=num_cells,bit_orders=[]):
  global best
  unpicked = set(range(num_bits)) - set(sum(bit_orders,[]))
  # if all except one cell already have bits assigned, the remaining bits go to that last cell
  if cells_to_pick == 1:
    bit_orders += [sorted(list(unpicked))]
    # print current progress on the same line to avoid console spam
    sys.stdout.write("\b"*30 + "best so far: " + str(best[:2]) + ", trying: " + str(bit_orders) + "\r")
    sys.stdout.flush()
    (err, x, offsets) = opt_cell(bit_orders)
    # if this bit allocation is better than the previous best, replace it
    if err < best[1]:
      best = (bit_orders, err, x, offsets)
  else:
    # always select the largest remaining bit to avoid redundancy and uniquely order the cells
    max_bit = max(unpicked)
    unpicked.remove(max_bit)
    # determine how many more bits to use in this cell
    for num_extra_bits in range(len(unpicked)-cells_to_pick+2):
      # select which specific bits to use in this cell
      for extra_bits in itertools.combinations(unpicked, num_extra_bits):
        # recursive call to select bits for the next cell
        opt(cells_to_pick - 1, bit_orders + [sorted(list(extra_bits)) + [max_bit]])

# silence "divide by 0" warnings caused by taking np.log(0) in probability calculations
np.seterr(divide='ignore')
best = ([], np.inf)
opt()
(bit_orders, err, x, offsets) = best
print("")
print("")
print("Bit Orders: " + str(bit_orders))
for cell in range(num_cells):
  levels = x[offsets[cell]:offsets[cell+1]:2]
  cutoffs = x[offsets[cell]+1:offsets[cell+1]:2]
  print("Cell " + str(cell) + " Levels:  " + str(np.round(levels,5)))
  print("Cell " + str(cell) + " Cutoffs: " + str(np.round(cutoffs,5)))
print("Error: " + str(np.round(err,5)))

def error_dist(x,acc=10**-5):
  return (cum_error_dist(x+acc)-cum_error_dist(x-acc))/(2*acc)

fig, ax = plt.subplots(nrows=num_cells, ncols=1)
samples = np.linspace(0, 1, 1000)
for (cell, row) in enumerate(ax):
  levels = x[offsets[cell]:offsets[cell+1]:2]
  cutoffs = x[offsets[cell]+1:offsets[cell+1]:2]
  for level in levels:
    row.plot(samples, error_dist(samples-level), color='blue', alpha=.5)
  for cutoff in cutoffs:
    row.axvline(x=cutoff, color='red', alpha=.3)
  row.set_title("Cell " + str(cell+1))
  row.set_xlim([0,1])
  row.set_ylim(bottom=0)
  row.get_xaxis().set_visible(False)
  row.get_yaxis().set_visible(False)
plt.show()













