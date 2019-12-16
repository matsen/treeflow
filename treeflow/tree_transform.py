import tensorflow as tf
import tensorflow_probability as tfp

class BranchBreaking(tfp.bijectors.Bijector): # TODO: Broadcast over batch_dims
    def __init__(self, parent_indices, preorder_node_indices, anchor_heights=None, name='BranchBreaking'):
        super(BranchBreaking, self).__init__(forward_min_event_ndims=1,name=name)
        self.parent_indices = parent_indices
        self.preorder_node_indices = preorder_node_indices # Don't include root
        self.taxon_count = preorder_node_indices.shape[-1] + 2
        self.anchor_heights = tf.zeros(self.taxon_count - 1, dtype=tf.dtypes.float32) if anchor_heights is None else anchor_heights
        
    def _forward(self, x):
        batch_shape = x.shape[:-1]
        init = tf.scatter_nd(
            tf.reshape(self.taxon_count - 2, [1, 1]),
            tf.expand_dims(x[..., -1] + self.anchor_heights[..., -1], 0),
            (self.taxon_count - 1) + batch_shape
        ) # Begin with node on first axis
        def f(out, elems):
            node_index, parent_index, proportion, anchor_height = elems
            out_t = tf.transpose(out,) # TODO: Can we do this by supplying axis to gather? Or by not permuting output?
            parent_height = tf.gather(out_t, parent_index, batch_dims=-1)
            node_height = (parent_height - anchor_height) * proportion + anchor_height
            return tf.tensor_scatter_nd_update(out, tf.reshape(node_index, [-1, 1]), node_height)

        batch_dim_count = tf.rank(x) - 1
        preorder_node_indices_b = tf.broadcast_to(self.preorder_node_indices, batch_shape + (self.taxon_count - 2))
        parent_indices_b = tf.broadcast_to(self.parent_indices, batch_shape + (self.taxon_count - 2))
        anchor_heights_b = tf.broadcast_to(self.anchor_heights, x.shape)

        preorder_parent_indices = tf.gather(parent_indices_b, preorder_node_indices_b, batch_dims=-1)
        preorder_anchor_heights = tf.gather(anchor_heights_b, preorder_node_indices_b, batch_dims=-1)
        preorder_x = tf.gather(x, preorder_node_indices_b, batch_dims=-1)

        perm = tf.concat([tf.expand_dims(batch_dim_count, 0), tf.range(batch_dim_count)], axis=0)
        preorder_node_indices_trans = tf.transpose(preorder_node_indices_b, perm)
        preorder_parent_indices_trans = tf.transpose(preorder_parent_indices, perm)
        preorder_anchor_heights_trans = tf.transpose(preorder_anchor_heights, perm)
        preorder_x_trans = tf.transpose(preorder_x, perm)
        
        scan_result = tf.scan(
            f,
            (
                preorder_node_indices_trans,
                tf.expand_dims(preorder_parent_indices_trans, axis=-1),
                preorder_x_trans,
                preorder_anchor_heights_trans
            ),
            init)
        scan_result = scan_result[-1]

        unperm = tf.concat([tf.range(1, batch_dim_count + 1), tf.zeros(1, dtype=tf.int32)], axis=0)
        return tf.transpose(scan_result, unperm)

    def _inverse(self, y):
        batch_shape = y.shape[:-1]
        batch_dim_count = tf.rank(y) - 1
        anchor_heights_b = tf.broadcast_to(self.anchor_heights, batch_shape + (self.taxon_count - 1))
        parent_indices_b = tf.broadcast_to(self.parent_indices, batch_shape + (self.taxon_count - 2))
        parent_heights = tf.gather(y, parent_indices_b, batch_dims=-1)
        return (y - anchor_heights_b) / tf.concat([(parent_heights - anchor_heights_b[..., :-1]), tf.ones(batch_shape + 1)], batch_dim_count)

    def _inverse_log_det_jacobian(self, y):
        batch_shape = y.shape[:-1]
        anchor_heights_b = tf.broadcast_to(self.anchor_heights, batch_shape + (self.taxon_count - 1))
        parent_indices_b = tf.broadcast_to(self.parent_indices, batch_shape + (self.taxon_count - 2))
        parent_heights = tf.gather(y, parent_indices_b, batch_dims=-1)
        return -tf.reduce_sum(tf.math.log(parent_heights - anchor_heights_b[..., :-1]), axis=-1)


class TreeChain(tfp.bijectors.Chain):
    def __init__(self, parent_indices, preorder_node_indices, anchor_heights=None, name='TreeChain'):
        branch_breaking = BranchBreaking(parent_indices, preorder_node_indices, anchor_heights=anchor_heights)
        blockwise = tfp.bijectors.Blockwise(
            [tfp.bijectors.Sigmoid(), tfp.bijectors.Exp()],
            block_sizes=tf.concat([parent_indices.shape[-1:], [1]], 0)
        )
        super(TreeChain, self).__init__([branch_breaking, blockwise], name=name)

class FixedTopologyDistribution(tfp.distributions.JointDistributionNamed):
    def __init__(self, height_distribution, topology, name='FixedTopologyDistribution'):
        super(FixedTopologyDistribution, self).__init__(dict(
        topology=tfp.distributions.JointDistributionNamed({ key: tfp.distributions.Deterministic(loc=value) for key, value in topology.items() }),
            heights=height_distribution
        ))
        self.topology_keys = topology.keys()
        self.heights_reparam = height_distribution.reparameterization_type

    @property
    def reparameterization_type(self): # Hack to allow VI
        return dict(heights=self.heights_reparam, topology={ key: tfp.distributions.FULLY_REPARAMETERIZED for key in self.topology_keys })

    