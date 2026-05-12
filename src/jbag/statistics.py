import math


def pooled_mean_std(num_elements_per_subgroups: list[int] | tuple[int, ...],
                    means: list[float] | tuple[float, ...],
                    standard_deviations: list[float] | tuple[float, ...],
                    ddof=0):
    """
    Compute overall mean and std from subgroup statistics.

    Args:
        num_elements_per_subgroups (list | tuple): sample sizes per subgroup.
        means (list | tuple): means per subgroup.
        standard_deviations (list | tuple): standard deviations per subgroup (sample std).
        ddof (int, optional, default=0): degrees of freedom.

    Returns:
        (mean, std): overall mean and std
    """

    total_num_elements = sum(num_elements_per_subgroups)

    # overall mean
    mean = sum(n * mu for n, mu in zip(num_elements_per_subgroups, means)) / total_num_elements

    # numerator for pooled variance
    var_num = 0.0
    for n, mu, sigma in zip(num_elements_per_subgroups, means, standard_deviations):
        var_num += (n - ddof) * (sigma ** 2) + n * (mu - mean) ** 2

    # pooled variance (sample variance)
    var = var_num / (total_num_elements - ddof)
    std = math.sqrt(var)

    return mean, std

