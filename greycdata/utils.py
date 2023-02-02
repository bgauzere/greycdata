"""
Tiré de graphkit-learn @Linlin Jia
"""
from typing import List, Union


def one_hot_encode(
        val: Union[int, str],
        allowable_set: Union[List[str], List[int]],
        include_unknown_set: bool = False) -> List[float]:
    """One hot encoder for elements of a provided set.

    Examples
    --------
    >>> one_hot_encode("a", ["a", "b", "c"])
    [1.0, 0.0, 0.0]
    >>> one_hot_encode(2, [0, 1, 2])
    [0.0, 0.0, 1.0]
    >>> one_hot_encode(3, [0, 1, 2])
    [0.0, 0.0, 0.0]
    >>> one_hot_encode(3, [0, 1, 2], True)
    [0.0, 0.0, 0.0, 1.0]

    Parameters
    ----------
    val: int or str
            The value must be present in `allowable_set`.
    allowable_set: List[int] or List[str]
            List of allowable quantities.
    include_unknown_set: bool, default False
            If true, the index of all values not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
            An one-hot vector of val.
            If `include_unknown_set` is False, the length is `len(allowable_set)`.
            If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

    Raises
    ------
    ValueError
            If include_unknown_set is False and `val` is not in `allowable_set`.
    """
    if include_unknown_set is False:
        if val not in allowable_set:
            logger.info("input {0} not in allowable set {1}:".format(
                val, allowable_set))

    # init an one-hot vector
    if include_unknown_set is False:
        one_hot_legnth = len(allowable_set)
    else:
        one_hot_legnth = len(allowable_set) + 1
    one_hot = [0.0 for _ in range(one_hot_legnth)]

    try:
        one_hot[allowable_set.index(val)] = 1.0  # type: ignore
    except:
        if include_unknown_set:
            # If include_unknown_set is True, set the last index is 1.
            one_hot[-1] = 1.0
        else:
            pass
    return one_hot
