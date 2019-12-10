import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import Counter
import treeflow.tensorflow_likelihood
import treeflow.tree_processing
import treeflow.substitution_model

init_partials_dict = {
    'A':[1.,0.,0.,0.],
    'C':[0.,1.,0.,0.],
    'G':[0.,0.,1.,0.],
    'T':[0.,0.,0.,1.],
    '-':[1.,1.,1.,1.],
    '?':[1.,1.,1.,1.],
    'N':[1.,1.,1.,1.],
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

def encode_sequence_dict(sequence_dict, taxon_names):
    return tf.convert_to_tensor(np.array([[init_partials_dict[char] for char in sequence_dict[taxon_name]] for taxon_name in taxon_names]))

def get_encoded_sequences(fasta_file, taxon_names):
    sequence_dict = parse_fasta(fasta_file)
    pattern_dict, counts = compress_sites(sequence_dict)
    return encode_sequence_dict(pattern_dict, taxon_names), counts

def get_branch_lengths(tree):
    heights = tree['heights']
    return tf.gather(heights, tree['topology']['parent_indices']) - heights[:-1]

def log_prob_conditioned(value, topology, category_count):

    likelihood = treeflow.tensorflow_likelihood.TensorflowLikelihood(category_count=category_count)
    likelihood.set_topology(treeflow.tree_processing.update_topology_dict(topology))
    likelihood.init_postorder_partials(value['sequences'], pattern_counts=(value['weights'] if 'weights' in value else None))

    def log_prob(branch_lengths, subst_model, frequencies, category_weights, category_rates, **subst_model_params):
        subst_model_param_keys = list(subst_model_params.keys())
        def redict_params(subst_model_params):
            return dict(zip(subst_model_param_keys, subst_model_params))
        @tf.custom_gradient
        def log_prob_flat(branch_lengths, frequencies, category_weights, category_rates, *subst_model_params_list):
            subst_model_params = redict_params(subst_model_params_list)
            eigendecomp = subst_model.eigen(frequencies, **subst_model_params)
            transition_probs = treeflow.substitution_model.transition_probs(eigendecomp, category_rates, branch_lengths)
            likelihood.compute_postorder_partials(transition_probs)
            def grad(dbranch_lengths, dfrequencies, dcategory_weights, dcategory_rates, *dsubst_model_params):
                likelihood.compute_preorder_partials(transition_probs)
                q = subst_model.q_norm(frequencies, **subst_model_params)
                q_freq_differentials = subst_model.q_norm_frequency_differentials(frequencies, **subst_model_params)
                freq_differentials = [treeflow.substitution_model.transition_probs_differential(q_freq_differentials[i], eigendecomp, branch_lengths, category_rates) for i in range(4)]
                q_param_differentials = subst_model.q_norm_param_differentials(frequencies, **subst_model_params)
                param_grads = [likelihood.compute_derivative(treeflow.substitution_model.transition_probs_differential(q_param_differentials[param_key], eigendecomp, branch_lengths, category_rates), category_weights) for param_key in subst_model_param_keys]
                return [
                    dbranch_lengths * likelihood.compute_branch_length_derivatives(q, category_rates, category_weights),
                    dfrequencies * tf.stack([likelihood.compute_frequency_derivative(freq_differentials[i], i, category_weights) for i in range(4)]),
                    dcategory_weights * likelihood.compute_weight_derivatives(category_weights),
                    dcategory_rates * likelihood.compute_rate_derivatives(q, branch_lengths, category_weights)
                ] + [dparam * param_grad for dparam, param_grad in zip(dsubst_model_params, param_grads)]
            return likelihood.compute_likelihood_from_partials(frequencies, category_weights), grad # TODO: Cache site likelihoods
        return log_prob_flat(branch_lengths, frequencies, category_weights, category_rates,
            *[subst_model_params[key] for key in subst_model_param_keys])
    return log_prob
    

class LeafSequences(tfp.distributions.Distribution):
    def __init__(self, tree, subst_model, frequencies, category_weights, category_rates,
        validate_args=False, allow_nan_stats=True, **subst_model_params):
        super(LeafSequences, self).__init__(
            dtype={ 'sequences': tf.int64, 'weights': tf.int64 },
            reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats, 
            parameters=dict(locals()))
        self.tree = tree
        self.subst_model = subst_model
        self.frequencies = frequencies
        self.category_weights = category_weights
        self.category_rates = category_rates
        self.subst_model_params = subst_model_params

    def _log_prob(self, value):
        return log_prob_conditioned(value, self.tree['topology'], len(self.category_weights))(self.tree['heights'], self.subst_model, self.frequencies, self.category_weights, self.category_rates, **self.subst_model_params)

    def _sample_n(self, n, seed=None):
        raise NotImplementedError('Sequence simulator not yet implemented')