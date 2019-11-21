import tensorflow as tf
import numpy as np
import re
from collections import Counter
from treeflow.base_likelihood import BaseLikelihood
import treeflow.substitution_model

init_partials_dict = {
    'A':[1.,0.,0.,0.],
    'C':[0.,1.,0.,0.],
    'G':[0.,0.,1.,0.],
    'T':[0.,0.,0.,1.],
    '-':[1.,1.,1.,1.],
    '?':[1.,1.,1.,1.],
    'N':[1.,1.,1.,1.],
    #'R':[1.,1.,0.,0.],# TODO: Fix indexing of these
    #'Y':[0.,0.,1.,1.],
    #'S':[0.,1.,1.,0.], 
    #'W':[1.,0.,0.,1.],
    #'K':[0.,1.,0.,1.],
    #'M':[1.,0.,1.,0.],
    #'B':[0.,1.,1.,1.],
    #'D':[1.,1.,0.,1.],
    #'H':[1.,0.,1.,1.],
    #'V':[1.,1.,1.,0.],
    '.':[1.,1.,1.,1.],
    'U':[0.,0.,0.,1.]
}

def parse_fasta(filename):
    f = open(filename)
    x = f.read()
    f.close()
    def process_block(block):
        lines = block.split('\n')
        return lines[0], ''.join(lines[1:])
    return dict([process_block(block) for block in x.split('>')[1:]])

def compress_sites(sequence_dict):
    taxa = sorted(list(sequence_dict.keys()))
    sequences = [sequence_dict[taxon] for taxon in taxa]
    patterns = list(zip(*sequences)) 
    count_dict = Counter(patterns)
    pattern_ordering = sorted(list(count_dict.keys()))
    compressed_sequences = list(zip(*pattern_ordering))
    counts = [count_dict[pattern] for pattern in pattern_ordering]
    pattern_dict = dict(zip(taxa, compressed_sequences))
    return pattern_dict, counts

class TensorflowLikelihood(BaseLikelihood):
    def __init__(self, fasta_file, category_count=1, *args, **kwargs):
        super(TensorflowLikelihood, self).__init__(fasta_file=fasta_file, *args, **kwargs)
        self.child_indices = self.get_child_indices() # TODO: Tidy initialisation
        self.postorder_node_indices = self.get_postorder_node_traversal_indices()
        self.category_count = category_count
        self.node_indices_tensor = tf.convert_to_tensor(self.postorder_node_indices)
        self.child_indices_tensor = tf.convert_to_tensor(self.child_indices[self.postorder_node_indices])
        preorder_node_indices = self.get_preorder_traversal_indices()[1:]
        self.preorder_node_indices_tensor = tf.convert_to_tensor(preorder_node_indices)
        self.preorder_sibling_indices_tensor = tf.convert_to_tensor(self.get_sibling_indices()[preorder_node_indices])
        self.preorder_parent_indices_tensor = tf.convert_to_tensor(self.get_parent_indices()[preorder_node_indices])

        self.init_postorder_partials(fasta_file)

    def init_postorder_partials(self, fasta_file):
        newick = self.inst.tree_collection.newick()
        leaf_names = re.findall(r'(\w+)(?=:)', newick)
        sequence_dict = parse_fasta(fasta_file)
        pattern_dict, self.pattern_counts = compress_sites(sequence_dict) 
        postorder_partials = np.zeros((self.get_vertex_count(), len(self.pattern_counts), self.category_count, 4))
        for leaf_index in range(len(leaf_names)):
            for pattern_index in range(len(self.pattern_counts)):
                postorder_partials[leaf_index, pattern_index] = np.array(init_partials_dict[pattern_dict[leaf_names[leaf_index]][pattern_index]])[np.newaxis, :]
        self.postorder_partials = tf.convert_to_tensor(postorder_partials)

    def compute_postorder_partials(self, transition_probs):
        node_indices = tf.reshape(self.node_indices_tensor, [-1, 1, 1])
        child_transition_probs =  tf.gather(transition_probs, self.child_indices_tensor)
        def do_integration(partials, elems):
            node_index, node_child_transition_probs, node_child_indices = elems
            child_partials = tf.gather(partials, node_child_indices)
            node_partials = tf.reduce_prod(tf.reduce_sum(tf.expand_dims(node_child_transition_probs, 1) * tf.expand_dims(child_partials, 3), axis=4), axis=0)
            return tf.tensor_scatter_nd_update(partials, node_index, tf.expand_dims(node_partials, axis=0))
        self.postorder_partials = tf.scan(do_integration, (node_indices, child_transition_probs, self.child_indices_tensor), self.postorder_partials)[-1]
    
    def compute_likelihood_from_partials(self, freqs, category_weights):
        cat_likelihoods = tf.reduce_sum(freqs * self.postorder_partials[-1], axis=-1)
        site_likelihoods = tf.reduce_sum(category_weights * cat_likelihoods, axis=-1)
        return tf.reduce_sum(tf.math.log(site_likelihoods) * self.pattern_counts, axis=-1)

    def compute_likelihood(self, branch_lengths, category_rates, category_weights, freqs, eigendecomp):
        transition_probs = treeflow.substitution_model.transition_probs(eigendecomp, category_rates, branch_lengths)
        self.compute_postorder_partials(transition_probs)
        return self.compute_likelihood_from_partials(freqs, category_weights)

    def init_preorder_partials(self, frequencies):
        zeros = tf.zeros([self.get_vertex_count(), len(self.pattern_counts), self.category_count, 4], dtype=tf.float64)
        self.preorder_partials = tf.tensor_scatter_nd_update(
            zeros,
            np.array([[self.get_vertex_count() - 1]]),
            tf.expand_dims(tf.broadcast_to(tf.reshape(frequencies, [1, 1, 4]), (len(self.pattern_counts), self.category_count, 4)), 0))

    def compute_preorder_partials(self, transition_probs):
        node_indices = tf.reshape(self.preorder_node_indices_tensor, [-1, 1, 1])
        preorder_transition_probs = tf.gather(transition_probs, self.preorder_node_indices_tensor)
        sibling_transition_probs = tf.gather(transition_probs, self.preorder_sibling_indices_tensor)
        sibling_postorder_partials = tf.gather(self.postorder_partials, self.preorder_sibling_indices_tensor)
        sibling_sums = tf.reduce_sum(tf.expand_dims(sibling_transition_probs, 1) * tf.expand_dims(sibling_postorder_partials, 3), axis=4)
        def do_integration(partials, elems):
            node_index, node_sibling_sums, node_transition_probs, node_parent_index = elems
            parent_partials = partials[node_parent_index]
            parent_prods = parent_partials * node_sibling_sums
            node_partials = tf.reduce_sum(tf.expand_dims(node_transition_probs, 0) * tf.expand_dims(parent_prods, 3), axis=2)
            return tf.tensor_scatter_nd_update(partials, node_index, tf.expand_dims(node_partials, axis=0))
        self.preorder_partials = tf.scan(do_integration, (node_indices, sibling_sums, preorder_transition_probs, self.preorder_parent_indices_tensor), self.preorder_partials)[-1]

    def compute_cat_derivatives(self, differential_matrices, sum_branches=False):
        differential_transpose = tf.transpose(differential_matrices, perm=[0, 1, 3, 2])
        return tf.reduce_sum(
            tf.expand_dims(self.postorder_partials[:-1], 4) *
                tf.expand_dims(differential_transpose, 1) *
                tf.expand_dims(self.preorder_partials[:-1], 3),
            axis=([0, 3, 4] if sum_branches else [3, 4])
        )

    def compute_site_derivatives(self, differential_matrices, category_weights, sum_branches=False):
        cat_derivatives = self.compute_cat_derivatives(differential_matrices, sum_branches=sum_branches)
        return tf.reduce_sum(cat_derivatives * category_weights, axis=-1)

    def compute_site_likelihoods(self, category_weights):
        return tf.reduce_sum(tf.reduce_sum(self.postorder_partials[-1] * self.preorder_partials[-1], axis=-1) * category_weights, axis=-1)

    def compute_derivative(self, differential_matrices, category_weights):
        site_derivatives = self.compute_site_derivatives(differential_matrices, category_weights, sum_branches=True)
        site_coefficients = self.pattern_counts / self.compute_site_likelihoods(category_weights)
        return tf.reduce_sum(site_derivatives * site_coefficients)

    def compute_edge_derivatives(self, differential_matrices, category_weights):
        site_likelihoods = self.compute_site_likelihoods(category_weights)
        site_derivatives = self.compute_site_derivatives(differential_matrices, category_weights)
        return tf.reduce_sum(self.pattern_counts / site_likelihoods * site_derivatives, axis=-1)

    def compute_branch_length_derivatives(self, q, category_weights):
        return self.compute_edge_derivatives(tf.reshape(q, [1, 1, 4, 4]), category_weights)

    def compute_frequency_derivative(self, differential_matrices, frequency_index, category_weights):
        site_likelihoods = self.compute_site_likelihoods(category_weights)
        cat_derivatives = self.compute_cat_derivatives(differential_matrices, sum_branches=True)
        site_coefficients = self.pattern_counts / site_likelihoods
        return tf.reduce_sum(site_coefficients * tf.reduce_sum((cat_derivatives + self.postorder_partials[-1, :, frequency_index]) * category_weights, axis=-1))
