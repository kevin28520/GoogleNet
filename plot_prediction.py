# horizontal bar to show predictions
def plot_prob(prob, name):
  arg_sort = prob.argsort()

  p_sorted = prob[0, arg_sort]
  p_sorted = np.reshape(p_sorted, (20, ))

  name = np.array(train_all.target_names)
  name = name[arg_sort]
  name = np.reshape(name, (20, ))

  plt.figure(figsize=(12, 9))
  plt.barh(np.arange(20), p_sorted)

  plt.xlim([0., 1.1])
  plt.xlabel('probability')
  plt.ylabel('class')

  plt.yticks(np.arange(20), name)

  plt.show()
