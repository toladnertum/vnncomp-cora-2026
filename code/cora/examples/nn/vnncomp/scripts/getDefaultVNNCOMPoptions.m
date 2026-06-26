function options = getDefaultVNNCOMPoptions(benchName)
% getDefaultVNNCOMPoptions - return the default verification options for a benchmark.
%
% Syntax:
%    options = getDefaultVNNCOMPoptions(benchName)
%
% Inputs:
%    benchName - name of the benchmark (e.g. 'acasxu_2023', 'safenlp_2024')
%
% Outputs:
%    options - struct with options.nn set to benchmark defaults
%
% See also: prepare_instance, hyperparam_tuning_vnncomp

% Authors:       Benedikt Kellner, Lukas Koller
% Written:       04-May-2026
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% Create evaluation options.
options.nn = struct(...
    'use_approx_error',true,...
    'poly_method','bounds',... {'bounds','singh','center'}
    'train',struct(...
    'backprop',false,...
    'mini_batch_size',2^10 ...
    ) ...
);
% Set default training parameters
options = nnHelper.validateNNoptions(options,true);
% Disable the interval-center by default.
options.nn.interval_center = false;
% Use the moving statistics for the batch normalization.
options.nn.batch_norm_moving_stats = true;

% Specify falsification method: {'center','fgsm','zonotack'}.
options.nn.falsification_method = 'zonotack';
% Specify input set refinement method: {'naive','zonotack','zonotack-layerwise'}.
options.nn.refinement_method = 'zonotack';
% Set number of input generators.
options.nn.train.num_init_gens = inf;
% Set number of approximation error generators per layer.
options.nn.approx_error_order = 'sensitivity*length';
% Compute the exact bounds of the constraint zonotope.
options.nn.conzonotope_bounding_method = 'fourier-motzkin'; % {'fourier-motzkin','dual-iter','exact'}.
options.nn.polytope_bound_approx_max_iter = 4; % only for 'fourier-motzkin'
options.nn.conzonotope_bound_max_iter = 200; % only for 'dual-iter'
options.nn.conzonotope_bound_step_size = 1e-2; % only for 'dual-iter'
% Specify number of splits, dimensions, and neuron-splits.
options.nn.num_pieces_per_split = 2;
options.nn.num_input_dimension_splits = 1;
options.nn.num_neuron_splits = 0;
% Add relu tightening constraints.
options.nn.num_relu_constraints = 0;
% Specify the number of iterations.
options.nn.refinement_min_iter = 4;
options.nn.refinement_max_iter = 8;
% Specify the queue-style.
options.nn.verify_dequeue_type = 'half-half';
options.nn.verify_enqueue_type = 'prepend';

% Specify the heuristics.
options.nn.input_generator_heuristic = 'zono-norm-gradient';
options.nn.input_split_heuristic = 'zono-norm-gradient';
options.nn.neuron_split_heuristic = 'zono-norm-gradient';
options.nn.relu_constraint_heuristic = 'zono-norm-gradient';

% benchmark-specific option overrides (VNN-COMP'25)
if strcmp(benchName,'acasxu')
    % options.nn.num_pieces_per_split = 2;
    % options.nn.num_input_dimension_splits = 0;
    % options.nn.num_neuron_splits = 1;

elseif strcmp(benchName,'cgan')
    options.nn.num_pieces_per_split = 2;
    options.nn.num_input_dimension_splits = 1;
    options.nn.num_neuron_splits = 1;
    options.nn.train.mini_batch_size = 2^2;
    options.nn.neuron_split_heuristic = 'least-unstable';
    options.nn.verify_cascade_unsafe_set_constraints = false;
    options.nn.num_relu_constraints = 0;

elseif strcmp(benchName,'challenging_certified_training')
    % CNN7 image-classification robustness nets; image-net baseline, to tune.
    options.nn.interval_center = true;
    options.nn.train.num_init_gens = 500;
    options.nn.train.num_approx_err = 0;
    options.nn.num_relu_constraints = 0;
    options.nn.num_pieces_per_split = 2;
    options.nn.num_input_dimension_splits = 1;
    options.nn.num_neuron_splits = 1;
    options.nn.batch_union_conzonotope_bounds = false;

elseif strcmp(benchName,'cifar100') % large image classification
    options.nn.train.mini_batch_size = 8;
    options.nn.input_split_heuristic = 'zono-norm-gradient';
    options.nn.neuron_split_heuristic = 'least-unstable';
    options.nn.refinement_method = 'zonotack';
    options.nn.falsification_method = 'fgsm';
    options.nn.num_input_dimension_splits = 0;
    options.nn.num_pieces_per_split = 2;
    options.nn.num_neuron_splits = 1;
    options.nn.num_relu_constraints = 100;
    options.nn.train.num_init_gens = 250;
    options.nn.train.num_approx_err = 25;
    options.nn.verify_dequeue_type = 'half-half';
    options.nn.verify_enqueue_type = 'append';

elseif strcmp(benchName,'collins_rul_cnn') % VNN-COMP'24
    options.nn.interval_center = true;
    options.nn.train.num_init_gens = inf;
    options.nn.train.num_approx_err = 100;

elseif strcmp(benchName,'cora')
    options.nn.num_relu_constraints = inf;
    options.nn.train.mini_batch_size = 2^5;
    options.nn.num_pieces_per_split = 2;
    options.nn.num_input_dimension_splits = 1;
    options.nn.num_neuron_splits = 1;
    options.nn.batch_union_conzonotope_bounds = false;

elseif strcmp(benchName,'metaroom')
    options.nn.train.num_init_gens = 500;
    options.nn.train.num_approx_err = 100;
    options.nn.train.mini_batch_size = 2^2;
    options.nn.num_pieces_per_split = 2;
    options.nn.num_input_dimension_splits = 1;
    options.nn.num_neuron_splits = 0;
    options.nn.num_relu_constraints = 0;
    options.nn.batch_union_conzonotope_bounds = false;
    options.nn.max_verif_iter = 10;

elseif strcmp(benchName,'mnist_fc') % VNN-COMP'22
    options.nn.interval_center = true;
    options.nn.train.num_init_gens = 500;
    options.nn.train.num_approx_err = 100;
    options.nn.batch_union_conzonotope_bounds = false;

elseif strcmp(benchName,'oval21')
    options.nn.interval_center = true;
    options.nn.train.num_init_gens = 100;
    options.nn.train.num_approx_err = 10;
    options.nn.batch_union_conzonotope_bounds = false;

elseif ismember(benchName,{'monotonic_acasxu','isomorphic_acasxu'})
    % VNN-COMP'26 multi-network ACAS Xu benchmarks.
    % Small composite networks (joint f+g, 5-in each); use input-radius
    % heuristics to avoid backprop through the composite layer.
    options.nn.num_pieces_per_split = 2;
    options.nn.num_input_dimension_splits = 1;
    options.nn.num_neuron_splits = 0;
    options.nn.train.mini_batch_size = 2^10;

elseif strcmp(benchName,'safenlp')
    options.nn.num_pieces_per_split = 2;
    options.nn.num_input_dimension_splits = 1;
    options.nn.num_neuron_splits = 1;
    options.nn.num_relu_constraints = 100;

elseif strcmp(benchName,'tinyimagenet') % VNN-COMP'24
    options.nn.interval_center = true;
    options.nn.train.num_init_gens = 500;
    options.nn.train.num_approx_err = 0;
    options.nn.num_relu_constraints = 0;
    options.nn.num_pieces_per_split = 2;
    options.nn.num_input_dimension_splits = 1;
    options.nn.num_neuron_splits = 1;
    options.nn.batch_union_conzonotope_bounds = false;
    options.nn.train.mini_batch_size = 2^5;

end

end

% ------------------------------ END OF CODE ------------------------------
