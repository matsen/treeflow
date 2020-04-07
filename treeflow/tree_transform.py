import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.tf_util

class ParentCorrelation(tfp.bijectors.ScaleMatvecLU):
    def __init__(self, parent_indices, beta, name='ParentAffine'):
        non_root_count = parent_indices.shape[-1]
        node_count = non_root_count + 1
        perm = tf.range(node_count)
        indices = tf.concat([tf.expand_dims(tf.range(non_root_count), 1), tf.expand_dims(tf.convert_to_tensor(parent_indices, tf.int32), 1)], axis=1)
        build_triu = lambda beta: tf.eye(node_count) + tf.scatter_nd(indices, beta, [node_count, node_count])
        super(ParentCorrelation, self).__init__(tfp.util.DeferredTensor(beta, build_triu, shape=[node_count, node_count]), perm, name=name)

class LogSigmoid(tfp.bijectors.Softplus):
    def __init__(self, name='logsigmoid', **kwargs):
        return super(LogSigmoid, self).__init__(name=name, **kwargs)
    
    def _forward(self, x):
        return -super(LogSigmoid, self)._forward(-x)
    
    def _inverse(self, y):
        return -super(LogSigmoid, self)._inverse(-y)
    
    def _inverse_log_det_jacobian(self, y):
        return super(LogSigmoid, self)._inverse_log_det_jacobian(-y)
    
    def _forward_log_det_jacobian(self, x):
        return super(LogSigmoid, self)._forward_log_det_jacobian(-x)

class BranchBreaking(tfp.bijectors.Bijector): # TODO: Broadcast over batch_dims
    def __init__(self, parent_indices, preorder_node_indices, anchor_heights=None, name='BranchBreaking'):
        super(BranchBreaking, self).__init__(forward_min_event_ndims=1,name=name)
        self.parent_indices = parent_indices
        self.preorder_node_indices = preorder_node_indices # Don't include root
        self.anchor_heights = tf.zeros(len(preorder_node_indices) + 1, dtype=tf.dtypes.float32) if anchor_heights is None else anchor_heights

    def _forward_1d(self, log_x):
        length = log_x.shape[-1]
        init = tf.scatter_nd([[length - 1]], tf.expand_dims(tf.math.exp(log_x[-1]) + self.anchor_heights[-1], 0), self.anchor_heights.shape)
        def f(out, elems):
            node_index, parent_index, log_proportion, anchor_height = elems
            node_height = tf.exp(tf.math.log(out[parent_index] - anchor_height) + log_proportion) + anchor_height
            return tf.tensor_scatter_nd_update(out, tf.reshape(node_index, [1, 1]), tf.expand_dims(node_height, 0))
        return tf.scan(
            f,
            (
                self.preorder_node_indices,
                tf.gather(self.parent_indices, self.preorder_node_indices),
                tf.gather(log_x, self.preorder_node_indices), tf.gather(self.anchor_heights, self.preorder_node_indices)
            ),
            init)[-1]

    def _forward(self, x):
        return treeflow.tf_util.vectorize_1d_if_needed(self._forward_1d, x, x.shape.rank - 1)

    def _inverse_1d(self, y):
        return tf.math.log(y - self.anchor_heights) - tf.concat([tf.math.log(tf.gather(y, self.parent_indices) - self.anchor_heights[:-1]), tf.zeros((1,), dtype=tf.dtypes.float32)], 0)

    def _inverse(self, y):
        return treeflow.tf_util.vectorize_1d_if_needed(self._inverse_1d, y, y.shape.rank - 1)

    def _inverse_log_det_jacobian(self, y):
        return -tf.reduce_sum(tf.math.log(y - self.anchor_heights))


class TreeChain(tfp.bijectors.Chain):
    def __init__(self, parent_indices, preorder_node_indices, anchor_heights=None, name='TreeChain'):
        branch_breaking = BranchBreaking(parent_indices, preorder_node_indices, anchor_heights=anchor_heights)
        blockwise = tfp.bijectors.Blockwise(
            [LogSigmoid(), tfp.bijectors.Identity()],
            block_sizes=tf.concat([parent_indices.shape, [1]], 0)
        )
        super(TreeChain, self).__init__([branch_breaking, blockwise], name=name)

class FixedTopologyDistribution(tfp.distributions.JointDistributionNamed):
    def __init__(self, height_distribution, topology, name='FixedTopologyDistribution'):
        super(FixedTopologyDistribution, self).__init__(dict(
            topology=tfp.distributions.JointDistributionNamed({
                key: tfp.distributions.Independent(
                    tfp.distributions.Deterministic(loc=value),
                    reinterpreted_batch_ndims=height_distribution.event_shape.ndims
                ) for key, value in topology.items()
            }),
            heights=height_distribution
        ))
        self.topology_keys = topology.keys()
        self.heights_reparam = height_distribution.reparameterization_type

    @property
    def reparameterization_type(self): # Hack to allow VI
        return dict(heights=self.heights_reparam, topology={ key: tfp.distributions.FULLY_REPARAMETERIZED for key in self.topology_keys })

    
