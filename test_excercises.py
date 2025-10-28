import numpy as np

from exercises import *


def test_linear_system():
    A = np.array([[1.]])
    b = np.array([1]).reshape(1,1)
    expected = np.array([-1])
    answer = linear_system(A, b)
    assert answer.shape == (1, 1)
    assert np.allclose(expected, answer)

    A = np.array([
        [1., 1.],
        [2., 2.]
    ])
    b = np.array([
        [1., 2.]
    ])
    assert linear_system(A, b) is None

    A = np.array([
        [2, 3, -5],
        [1, -1, 2],
        [-1, 4, -1]
    ])
    b = np.array([10, -3, 0]).reshape(3, 1)
    expected = np.array([-19, 11, 63]).reshape(3, 1) / 32
    answer = linear_system(A, b)
    assert answer.shape == (3, 1)
    assert np.allclose(expected, answer)


def test_chessboard_pattern():
    assert np.allclose(chessboard_pattern(0), np.array(
        [], dtype=np.int64))

    assert np.allclose(chessboard_pattern(2), np.array([
        [0, 1],
        [1, 0]
    ], dtype=np.int64))

    assert np.allclose(chessboard_pattern(4), np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=np.int64))

    assert np.allclose(chessboard_pattern(6), np.array([
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0]
    ], dtype=np.int64))


def test_sort_by_key():
    arr = np.array([-8, 2, 3, -1, 0])
    
    expected = np.array([0, -1, 2, 3, -8])
    assert np.allclose(expected, sort_by_key(arr, lambda x: np.abs(x)))

    result = sort_by_key(arr, lambda x: x % 3)
    assert np.allclose(result[:2], [0, 3]) or np.allclose(result[:2], [3, 0])
    assert result[2] == -8
    assert np.allclose(result[3:], [-1, 2]) or np.allclose(result[3:], [2, -1])


def test_row_magnitude():
    arr = np.array([
        [1, 0, 0],
        [3, 4, 0]
    ])
    expected = np.array([1, 5]).reshape(2, 1)
    result = row_magnitude(arr)
    assert result.shape == expected.shape
    assert np.allclose(expected, result)

    arr = np.array([
        [1, 0, 0, 1],
        [3, 4, 0, 123],
        [-3, -4, 0, 97]
    ])
    expected = np.sqrt(np.array([2, 15154, 9434])).reshape(3, 1)
    result = row_magnitude(arr)
    assert result.shape == expected.shape
    assert np.allclose(expected, result)


def test_outlier_filtering():
    arr = np.array([123, 111, 156, 145, 120])
    expected = np.array([123, 111, 156, 145, 120])
    assert np.allclose(expected, outlier_filtering(arr, 5))

    arr = np.concatenate((
        np.random.randint(100, 110, 100),
        np.array([3000])
    ))
    expected = arr[:-1]
    assert np.allclose(expected, outlier_filtering(arr, 5))

    arr = np.concatenate((
        np.random.randint(100, 110, 100),
        np.array([3])
    ))
    expected = arr[:-1]
    assert np.allclose(expected, outlier_filtering(arr, 5))


    arr = np.concatenate((
        np.random.randint(100, 110, 100),
        np.array([90])
    ))
    expected = arr
    assert np.allclose(expected, outlier_filtering(arr, 5))


    arr = np.concatenate((
        np.random.randint(100, 110, 100),
        np.array([90])
    ))
    expected = arr[:-1]
    assert np.allclose(expected, outlier_filtering(arr, 3))
    
def test_add_row_col():    
    arr = np.array([
        [4,5,2],
        [3,0,1],
        [2,5,8]
    ])
    expected = np.array([
        [4,5,2,8],
        [3,0,1,6],
        [2,5,8,4],
        [8,10,4,16]
    ])
    result = add_row_col(arr)
    assert np.allclose(expected, result)
    assert result.shape == expected.shape
    
    arr = np.array([
        [0,2,1],
        [-1,3,4]
    ])
    expected = np.array([
        [0,2,1,0],
        [-1,3,4,-2],
        [0,4,2,0]
    ])
    result = add_row_col(arr)
    assert np.allclose(expected, result)
    assert result.shape == expected.shape
    
    
def test_array_centering():
    arr = np.random.random((10, 3))
    result = array_centering(arr)
    result_mean = result.mean(axis=0)
    assert np.allclose(result_mean, np.zeros((1,3)))
    assert result.shape == (10,3)
    
    arr = np.random.random((10, 10))
    result = array_centering(arr)
    result_mean = result.mean(axis=0)
    assert np.allclose(result_mean, np.zeros((1,10)))
    assert result.shape == (10,10)
    

def test_substitute():
    data = np.arange(0, 10, 2)
    result = substitute(data)
    assert np.allclose(result, np.zeros(5))

    data = np.arange(1, 11, 2)
    result = substitute(data)
    assert np.allclose(result, data)


def test_flatten():
    arr1 = np.arange(3)
    arr2 = np.arange(3, 7)
    arr3 = np.arange(7, 10)
    arr = np.array([arr1, arr2, arr3], dtype=object)
    result = flatten(arr)
    assert result.shape == (10,)
    assert np.allclose(result, np.arange(10))
    

def test_is_duplicate():
    np.random.seed(100)
    
    data = np.random.randint(0, 5, 10)
    expected = np.array([False, True, False, True, False, False, True, True, True, True])
    result = is_duplicate(data)
    assert np.all(result == expected)
    
    data = np.random.randint(0, 5, 10)
    expected = np.array([False, False, True, False, False, True, False, True, True, True])
    result = is_duplicate(data)
    assert np.all(result == expected)
    
    