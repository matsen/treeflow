import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose
import pytest

from treeflow.coalescent import ConstantCoalescent

# TODO: Tests for broadcasting

def test_coalescent_homochronous():
    pop_size = tf.convert_to_tensor(np.array(10000.0), dtype=tf.float32)
    sampling_times = tf.convert_to_tensor(np.array([0.0, 0.0, 0.0]), dtype=tf.float32)
    heights = tf.convert_to_tensor(np.array([1.0, 2.0]), dtype=tf.float32)
    parent_indices = tf.convert_to_tensor(np.array([3, 3, 4, 4]), dtype=tf.int32)
    dist = ConstantCoalescent(pop_size, sampling_times)
    res = dist.log_prob(dict(heights=heights, topology=dict(parent_indices=parent_indices)))
    expected =  -(4 / pop_size) - 2 * np.log(pop_size)
    assert_allclose(res.numpy(), expected)

test_data = [(123.0,-14.446309163678226),(999.0,-20.721465537146862)]
@pytest.mark.parametrize('pop_size,expected', test_data)
def test_coalescent_heterochronous(pop_size, expected):
    pop_size = tf.convert_to_tensor(np.array(pop_size), dtype=tf.float32)
    sampling_times = tf.convert_to_tensor(np.array([0.0, 0.1, 0.4, 0.0]), dtype=tf.float32)
    heights = tf.convert_to_tensor(np.array([0.2, 0.5, 0.8]), dtype=tf.float32)
    parent_indices = tf.convert_to_tensor(np.array([4, 4, 5, 6, 5, 6]), dtype=tf.int32)
    dist = ConstantCoalescent(pop_size, sampling_times)
    res = dist.log_prob(dict(heights=heights, topology=dict(parent_indices=parent_indices)))
    assert_allclose(res.numpy(), expected)