def aggregate_mean_std(mean1, std1, num1, mean2, std2, num2, ddof=1):
    """
    Calculate the mean and std of a group of data composed by two groups of data, given the mean, the std, and the
    number of data of the two subsets.

    Args:
          mean1 (float): Mean value of group1.
          std1 (float): Std value of group1.
          num1 (int): Number of data in group1.
          mean2 (float): Mean value of group2.
          std2 (float): Std value of group2.
          num2 (int): Number of data in group2.
          ddof (int, optional, default=1): Means of delta degrees of freedom. Default is 1 for unbiased estimation.
    """
    mean = (num1 * mean1 + num2 * mean2) / (num1 + num2)
    d = ((num1 - ddof) * std1 ** 2 + (num2 - ddof) * std2 ** 2) / (num1 + num2 - ddof) + \
        num1 * num2 * (mean1 - mean2) ** 2 / (num1 + num2) / (num1 + num2 - ddof)
    std = d ** 0.5
    return mean, std
