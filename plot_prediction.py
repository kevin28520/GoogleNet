# horizontal bar to show predictions
def barplot_prob(prob, name):
  # prob: np.ndarray, 1xn
  # name: np.ndarray, 1xn
  
  assert prob.size == name.size
  n_name = prob.size
  
  arg_sort = prob.argsort()
  
  name = name[0, arg_sort]
  name = np.reshape(name, (n_name, ))

  p_sorted = prob[0, arg_sort]
  p_sorted = np.reshape(p_sorted, (n_name, ))

  plt.figure(figsize=(12, 9))
  plt.barh(np.arange(prob.size), p_sorted)

  plt.xlim([0., 1.1])
  plt.xlabel('probability')
  plt.ylabel('class')
  plt.yticks(np.arange(n_name), name)

  plt.show()
