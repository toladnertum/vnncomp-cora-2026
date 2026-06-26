function res = testnn_neuralNetwork_verify_metrics()
% testnn_neuralNetwork_verify_metrics - test configurable progress metrics
%
% Syntax:
%    res = testnn_neuralNetwork_verify_metrics()
%
% Inputs:
%    -
%
% Outputs:
%    res - true/false
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: testnn_neuralNetwork_verify

% Authors:       Benedikt Kellner
% Written:       07-April-2026
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

res = true;

verbose = true;
timeout = 10;

% Load ACAS Xu instance (n0=5, so unknown_volume is enabled by default).
model1Path = [CORAROOT '/models/Cora/nn/ACASXU_run2a_1_2_batch_2000.onnx'];
prop1Filename = [CORAROOT '/models/Cora/nn/prop_1.vnnlib'];
[nn,options,x,r,A,b,safeSet] = aux_readNetworkAndOptions(model1Path,prop1Filename);
options.nn.falsification_method = 'fgsm';
options.nn.refinement_method = 'naive';

% 1. Default metrics (n0=5 → unknown_volume + expansion_rate + global_lower_bound).
[verifRes,~,~] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);
assert(isfield(verifRes,'numVerified'));

% 2. global_lb should be finite after at least one iteration.
strcmp(verifRes.str,'VERIFIED');

% 3. Only global_lower_bound column.
options.nn.progress_metrics = {'global_lower_bound'};
[verifRes2,~,~] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);

% 4. Only expansion_rate column.
options.nn.progress_metrics = {'expansion_rate'};
[verifRes3,~,~] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);

% 5. All three columns explicitly.
options.nn.progress_metrics = {'unknown_volume','expansion_rate','global_lower_bound'};
[verifRes4,~,~] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);

% 6. With neuron splitting (zonotack).
options.nn.progress_metrics = {};
options.nn.falsification_method = 'zonotack';
options.nn.refinement_method = 'zonotack';
options.nn.num_splits = 2;
options.nn.num_dimensions = 2;
options.nn.num_neuron_splits = 1;
options.nn.train.num_init_gens = 5;
options.nn.train.num_approx_err = 50;
options.nn.approx_error_order = 'sensitivity*length';
options.nn.num_relu_tighten_constraints = inf;
[verifRes5,~,~] = nn.verify(x,r,A,b,safeSet,options,timeout,verbose);

end


% Auxiliary functions -----------------------------------------------------

function [nn,options,x,r,A,b,safeSet] = ...
        aux_readNetworkAndOptions(modelPath,vnnlibPath)
  % Set default options.
  options.nn = struct(...
      'use_approx_error',true,...
      'poly_method','bounds',...
      'train',struct(...
          'backprop',false,...
          'mini_batch_size',2^8 ...
      ) ...
  );
  options = nnHelper.validateNNoptions(options,true);
  options.nn.interval_center = false;

  % Read network and specification.
  nn = neuralNetwork.readONNXNetwork(modelPath,false,'BSSC');
  [X0,specs] = vnnlib2cora(vnnlibPath);
  x = 1/2*(X0{1}.sup + X0{1}.inf);
  r = 1/2*(X0{1}.sup - X0{1}.inf);
  % Extract specification matrices.
  if isa(specs.set,'halfspace')
      A = specs.set.c';
      b = specs.set.d;
  else
      A = specs.set.A;
      b = specs.set.b;
  end
  safeSet = strcmp(specs.type,'safeSet');
end

% ------------------------------ END OF CODE ------------------------------
