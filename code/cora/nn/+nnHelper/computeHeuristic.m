function h = computeHeuristic(heuristic,l,u,er,sens,grad,varargin)
% computeHeuristic - Computes the splitting heuristic score.
%
% Syntax:
%    h = computeHeuristic(heuristic,layerIdx,l,u,er,sens,grad,options)
%
% Inputs:
%    heuristic - name of the heuristic
%    l, u - lower and upper bounds
%    er - approximation error radius
%    sens - sensitivity
%    grad - gradient
%    varargin - optional arguments (prevNrXs, neuronIds, onlyUnstable,
%       normPerLayer, betaHyperplanes, simDiversity)
%
% Outputs:
%    h - computed heuristic score
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Authors:       Benedikt Kellner, Lukas Koller
% Written:       16-January-2026
% Last update:   07-April-2026 (LK, clean up; merge as auxiliary functions)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% Set default parameters.
[prevNrXs,neuronIds,onlyUnstable,normPerLayer,betaHyperplanes,simDiversity] = ...
    setDefaultValues({[],[],true,true,[],0}, varargin);

% Obtain the number of dimensions and batchsize.
[n,bSz] = size(l);

% Match the batch size of the gradient and the sensitity. (Due to splitting
% the sets can be replicated; therefore, we just replicate the gradient or
% sensitivity to avoid recomputation.)
if ~isempty(sens) && size(sens,2) < bSz
    % Replicate the sensitvity to match the batch size.
    sens = repelem(sens,1,floor(bSz/size(sens,2)));
end
if ~isempty(grad) && size(grad,2) < bSz
    % Replicate the sensitvity to match the batch size.
    grad = repelem(grad,1,floor(bSz/size(grad,2)));
end

% Compute the raw heuristic using the specific function.
switch heuristic
    case 'input-radius'
        h = aux_inputRadius(l,u,er,sens,grad);
    case 'least-unstable'
        h = aux_leastUnstable(l,u,er,sens,grad);
    case 'least-unstable-gradient'
        h = aux_leastUnstableGradient(l,u,er,sens,grad);
    case 'most-unstable'
        h = aux_mostUnstable(l,u,er,sens,grad);
    case 'gap-sensitivity'
        h = aux_gapSensitivity(l,u,er,sens,grad);
    case 'product-sensitivity'
        h = aux_productSensitivity(l,u,er,sens,grad);
    case 'centered-sensitivity'
        h = aux_centeredSensitivity(l,u,er,sens,grad);
    case 'most-sensitive-approx-error'
        h = aux_mostSensitiveApproxError(l,u,er,sens,grad);
    case 'most-sensitive-input-radius'
        h = aux_mostSensitiveInputRadius(l,u,er,sens,grad);
    case 'zono-norm-gradient'
        h = aux_zonoNormGradient(l,u,er,sens,grad);
    otherwise
        % Invalid option.
        throw(CORAerror('CORA:wrongFieldValue','heuristic', ...
            {'input-radius','least-unstable','least-unstable-gradient', ...
            'most-unstable','gap-sensitivity', ...
            'product-sensitivity','centered-sensitivity', ...
            'most-sensitive-approx-error', ...
            'most-sensitive-input-radius','zono-norm-gradient'}));
end

% Ensure the output has the correct shape.
h = reshape(h,[n bSz]);

% --- Common Post-Processing ---

if ~isempty(prevNrXs)
    % Obtain the batch size.
    [~,bSz] = size(prevNrXs);

    % We floor all entries. We mark unnecessary splits with decimal
    % numbers, thereby the split is not applied.
    prevNrXs = floor(prevNrXs);
    % Reduce redundancy by not add constraints for split neurons.
    if size(h,2) > size(prevNrXs,2)
        newSplits = size(h,2)/size(prevNrXs,2);
        prevNrXs_ = repelem(prevNrXs,1,newSplits);
    else
        prevNrXs_ = prevNrXs;
    end

    % Identify already split neurons.
    wasSplit = any(permute(abs(prevNrXs_),[3 2 1]) ...
        == repmat(neuronIds,bSz,1)',3);

    if simDiversity > 0 && ~isempty(betaHyperplanes)
        % Diversify the splits: reduce the heuristic of neurons whose
        % split hyperplane (in beta-space) aligns with that of a
        % previously split neuron of the same layer. The split
        % hyperplane normal of a neuron is its generator row. Splits
        % with orthogonal hyperplanes are not penalized.
        G = betaHyperplanes;
        % Normalize each neuron's split-hyperplane normal.
        nG = G./max(sqrt(sum(G.^2,2)),1e-12); % [n q bSz]
        % Pairwise cosine similarity of the split hyperplanes.
        simMat = abs(pagemtimes(nG,'none',nG,'transpose')); % [n n bSz]
        % Only compare against previously split neurons (along dim 2).
        splitMask = permute(wasSplit,[3 1 2]); % [1 n bSz]
        % For each neuron, obtain the most similar previous split.
        maxSim = reshape(max(simMat.*splitMask,[],2),size(h));
        % Reduce the heuristic for neurons aligned with a previous
        % split; leave the -inf scores of stable neurons untouched to
        % avoid -inf*0 = NaN.
        penalty = max(1 - simDiversity.*maxSim,0);
        h = h.*penalty;
    end

    % There is no similarity; just prevent splitting the same
    % neuron twice by setting the heuristic to -inf.
    h(wasSplit) = -inf;
end

if normPerLayer
  % Normalize the heuristic per layer (per batch element) so scores are
  % comparable across layers with different magnitudes.
  hMax = max(abs(h).*isfinite(h),[],1);
  hMax(hMax == 0) = 1;
  h = h ./ hMax;
end

if onlyUnstable
    % Flag unstable neurons.
    unstable = (l < 0 & 0 < u);
    % Only consider unstable neurons.
    h(~unstable) = -inf;
end

end


% Auxiliary functions -----------------------------------------------------

function h = aux_inputRadius(l,u,er,sens,grad)
% The radius of the input interval.

% Compute the radius.
r = 1/2*(u - l);
% Compute the heuristic.
h = r;
end

function h = aux_leastUnstable(l,u,er,sens,grad)
% Targets near-stable neurons (small min(|l|,|u|)) weighted by
% sensitivity; prioritizes cleaning up boundary cases to reduce the
% number of active ReLU switches.

% Inverse instability: neurons closer to stability score higher.
minBnd = 1./min(-l,u);
% Compute the heuristic.
h = minBnd.*sens;
end

function h = aux_leastUnstableGradient(l,u,er,sens,grad)
% Same as least-unstable but uses gradient magnitude instead of
% sensitivity as the weighting factor.

% Take the absolute value and add small epsilon to avoid
% numerical problems.
grad = abs(grad) + 1e-3;
% Inverse instability: neurons closer to stability score higher.
minBnd = 1./min(-l,u);
% Compute the heuristic.
h = minBnd.*grad;
end

function h = aux_mostUnstable(l,u,er,sens,grad)
% Prioritizes neurons deep in the unstable region (large min(|l|,|u|))
% weighted by sensitivity; the opposite of least-unstable.

% Instability magnitude: distance from 0 to the nearest bound.
instability = min(-l, u);
% Compute the heuristic.
h = instability.*sens;
end

function h = aux_gapSensitivity(l,u,er,sens,grad)
% Weighs the triangular ReLU relaxation gap (-l*u)/(u-l) by
% sensitivity; mimics the BaBSR heuristic from Alpha-Beta-CROWN.

% Compute the max relaxation gap for [l, u] (gap at x=0).
gap = (-l .* u) ./ (u - l);
% Handle potential division by zero or stable neurons.
gap(isnan(gap)) = 0;
% Compute the heuristic.
h = gap .* sens;
end

function h = aux_productSensitivity(l,u,er,sens,grad)
% Scores by |l*u| times sensitivity; favors neurons with both a large
% relaxation gap and a wide domain (equivalent to gap*width*sens).

% Compute the product of bound magnitudes (valid for l < 0 < u).
prodVal = -l .* u;
% Compute the heuristic.
h = prodVal .* sens;
end

function h = aux_centeredSensitivity(l,u,er,sens,grad)
% Favors neurons centered around zero (l ~ -u) where a split yields
% two roughly equal intervals, promoting a balanced search tree.

width = u - l;
bias = abs(l + u);
% Closeness to center (1 = perfect center, 0 = boundary).
symmetry = 1 - (bias ./ width);
% Compute the heuristic.
h = symmetry .* sens;
end

function h = aux_mostSensitiveApproxError(l,u,er,sens,grad)
% Prioritizes neurons with the largest zonotope approximation error
% weighted by output sensitivity.

% Compute the heuristic.
h = er.*sens;

% Ensure the output has the correct shape.
h = reshape(h,[],size(l,2));
end

function h = aux_mostSensitiveInputRadius(l,u,er,sens,grad)
% Scores by input interval radius (u-l)/2 times sensitivity; also
% known as "smearing" in abstract interpretation.

% Compute the radius.
r = 1/2*(u - l);
% Compute the heuristic.
h = r.*sens;
end

function h = aux_zonoNormGradient(l,u,er,sens,grad)
% Uses gradient magnitude (instead of sensitivity) weighted by the
% approximation error as a proxy for zonotope norm improvement.

% Take the absolute value and add small epsilon to avoid
% numerical problems.
grad = abs(grad) + 1e-3;
% Compute the heuristic.
h = grad.*er;
end

% ------------------------------ END OF CODE ------------------------------
