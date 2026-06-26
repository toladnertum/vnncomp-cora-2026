function [l,u,bl,bu] = approximateBoundsWithGPU(cZs,numUnionConst,options)
% approximateBoundsWithGPU - Approximate the bounds of a batched
%   constrained zonotope. Mainly, used for the input refinement procedure 
%   in neuuralNetwork/verify. This function is tuned towards speed rather
%   than accuracy; it significantly benefits from GPU computations. 
%   This is an efficent implementation of [1, Prop 3]. 
%   WARNING: The input zonotopes are constraint zonotopes with inequality
%       constraints; not equality- Please transform accordingly!
%
% Syntax:
%    cZs.c = <center [n x batch size]>;
%    cZs.G = <generator matrix [n x q x batch size]>; % q: number of generators.
%    cZs.A = <constraint matrix (on factor space) [p x q x batch size]>; % p: number of constraints.
%    cZs.b = <constraint offset [p x batch size]>;
%    cZs.dr = <interval error [n x batch size]>; % optional interval error
%       on the zonotope.
%    numUnionConst = 1; % Compute the intersection of all constraints.
%    options.nn.polytope_bound_approx_max_iter = 4; % Maximum number of iterations
%    options.nn.exact_conzonotope_bounds = false; % Approximate the bounds.
%    options.nn.batch_union_conzonotope_bounds = true; % Move union constraint to the batch.
%    [l,u,bl,bu] = approximateBoundsWithGPU(cZs,numUnionConst,options)
%
% Inputs:
%    cZs - (struct) batch constraint zonotope, please see the syntax for 
%       an example. For GPU ultilization, move everything to GPU before,
%       i.e., x = cast(x,'single gpuArray');
%    numUnionConst - number of unions constraints; the first 
%       #numUnionConst constraints of cZs are unified (needed for 
%       safeSet specifications).
%    options.nn.conzonotope_bounding_method - bounding method, i.e., 
%       {'fourier-motzkin','dual-iter','exact'}
%    options.nn.batch_union_conzonotope_bounds - (bool) batch union 
%       constraints
%
% Outputs:
%    obj - generated object
%
% References:
%    [1] Koller, L. "Out of the Shadows: Exploring a Latent Space for 
%       Neural Network Verification". ICLR (2026)
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: neuralNetwork/verify

% Authors:       Lukas Koller
% Written:       19-November-2025
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% Input arguments represent a constraint zonotope with inequality
% constraints.
% numUnionConst: number of unions constraints; the first #numUnionConst
% constraints of cZs are unified (needed for safeSet specifications).
% options.nn.conzonotope_bounding_method: use linear programs to compute the bounds.
% options.nn.batch_union_conzonotope_bounds: batch union constraints

% Extract parameters of the constraint zonotope.
c = cZs.c;
G = cZs.G;
dr = cZs.dr;
A = cZs.A;
b = cZs.b;

% Obtain number of dimensions, generators, and batch size.
[n,q,bSz] = size(G);

if isempty(A)
    % There are no constraints. Just compute the bounds of the
    % zonotope.
    r = dr + reshape(sum(abs(G),2),[n bSz]);
    l = c - r;
    u = c + r;
    % The bounds of the hypercube are just -1 and 1;
    bl = -ones([q bSz],'like',G);
    bu = ones([q bSz],'like',G);
    return;
end

% Pad the generator matrix with 0s if there are more dimensions in the 
% hypercube.
G = cat(2,G,zeros([n size(A,2)-q bSz],'like',G));
% Update the number of generators.
[~,q,~] = size(G);

% Specify indices of intersection constraints.
intConIdx = (numUnionConst+1):size(A,1);

% The safe set is the union of all constraints. Thus, we have to create a 
% new set for each constraint. Either compute the bounds in batch-wise
% fashion (for speed) or for-loop (to save memory space).
if options.nn.batch_union_conzonotope_bounds
    % Move union constraints into the batch.
    Au = reshape(permute(A(1:numUnionConst,:,:),[4 2 3 1]),...
        [1 q bSz*numUnionConst]);
    bu = reshape(permute(b(1:numUnionConst,:),[3 2 1]),...
        [1 bSz*numUnionConst]);
    % Replicate intersection constraints.
    Ai = repmat(A(intConIdx,:,:),1,1,numUnionConst);
    bi = repmat(b(intConIdx,:),1,numUnionConst);
    % Append intersection constraints.
    A = cat(1,Au,Ai);
    b = cat(1,bu,bi);

    % Replicate the center and generator matrix.
    c = repmat(c,1,numUnionConst);
    if numel(dr) > 1
        dr = repmat(dr,1,numUnionConst);
    end
    G = repmat(G,1,1,numUnionConst);

    % Approximate the bounds of the hypercube (bounded polytope).
    [l,u,bl,bu] = aux_boundConZonotope(c,G,dr,A,b,options);

    if numUnionConst > 1
        % Unify sets if a safe set is specified.
        l = min(reshape(l,[n bSz numUnionConst]),[],3);
        u = max(reshape(u,[n bSz numUnionConst]),[],3);
        bl = min(reshape(bl,[q bSz numUnionConst]),[],3);
        bu = max(reshape(bu,[q bSz numUnionConst]),[],3);
    end
else
    % Initialize the bounds.
    l = inf([n bSz],'like',G);
    u = -inf([n bSz],'like',G);
    bl = -ones([q bSz],'like',G);
    bu = ones([q bSz],'like',G);

    % Loop over the union constraints.
    for k=1:numUnionConst
        % Use the k-th union constraint and all intersection
        % constraints.
        Ak = A([k intConIdx],:,:);
        bk = b([k intConIdx],:);
        % Approximate the bounds of the hypercube.
        [lk,uk,blk,buk] = aux_boundConZonotope(c,G,dr,Ak,bk,options);
        % Unify constraints.
        l = min(l,lk);
        u = max(u,uk);
        bl = min(bl,blk);
        bu = max(bu,buk);
    end
end

% Identify empty sets.
isEmpty = any(bl > bu,1) | any(isnan(l) | isnan(u),1);
l(:,isEmpty) = NaN;
u(:,isEmpty) = NaN;
bl(:,isEmpty) = 0;
bu(:,isEmpty) = 0;

end


% Auxiliary functions -----------------------------------------------------

function [l,u,bl,bu] = aux_boundConZonotope(c,G,dr,A,b,options)
    % Compute the bounds [l,u] of a constrained zonotope.

    % Obtain number of dimensions, generators, and batch size.
    [n,q,bSz] = size(G);

    switch options.nn.conzonotope_bounding_method
        case 'fourier-motzkin'
            % Approximate the bounds of the costrained hypercube.
            [bl,bu] = aux_fourierMotzkinApproximation(A,b,options);

            % Map bounds of the factors to bounds of the constraint 
            % zonotope. We use interval arithmetic for that.
            bc = 1/2*permute(bu + bl,[1 3 2]);
            br = 1/2*permute(bu - bl,[1 3 2]);
            % Map bounds of the factors to bounds of the constraint 
            % zonotope.
            c = c + reshape(pagemtimes(G,bc),[n bSz]);
            r = dr + reshape(pagemtimes(abs(G),br),[n bSz]);
            l = c - r;
            u = c + r;
        case 'dual-iter'
            % Approximate the bounds of the costrained hypercube for a
            % simple feasibility check.
            [bl,bu] = aux_fourierMotzkinApproximation(A,b,options);

            % Map bounds of the factors to bounds of the constraint 
            % zonotope. We use interval arithmetic for that.
            bc = 1/2*permute(bu + bl,[1 3 2]);
            br = 1/2*permute(bu - bl,[1 3 2]);
            % Map bounds of the factors to bounds of the constraint 
            % zonotope.
            c_ = c + reshape(pagemtimes(G,bc),[n bSz]);
            r = dr + reshape(pagemtimes(abs(G),br),[n bSz]);
            l = c_ - r;
            u = c_ + r;

            % Bound the solution of the linear program.
            rl = -aux_boundConZonotopeSupportFunction(A,b,-G,bl,bu,options);
            % Bound the solution of the linear program.
            ru = aux_boundConZonotopeSupportFunction(A,b,G,bl,bu,options);

            % Compute the bounds of the constrained zonotope from the
            % bounded solutions.
            l = max(l,c + rl - dr);
            u = min(u,c + ru + dr);

            % Tighten the hypercube (factor) bounds by bounding the
            % support function along each factor direction (G_factor=I_q).
            % The FM bounds [bl,bu] are typically loose on dense
            % constraints; without this pass the refinement loop in
            % neuralNetwork/verify sees only the FM factor bounds and
            % cannot exploit the dual tightening of l,u.
            Iq = repmat(eye(q,'like',G),1,1,bSz);
            bld = -aux_boundConZonotopeSupportFunction( ...
                A,b,-Iq,bl,bu,options);
            bud = aux_boundConZonotopeSupportFunction( ...
                A,b,Iq,bl,bu,options);
            bl = max(bl,bld);
            bu = min(bu,bud);
        case 'exact'
            % Slow implementation that computes the exact bounds by solving 
            % 2*n linear programs.
    
            % Initialize the bounds.
            l = NaN([n bSz]);
            u = NaN([n bSz]);
            bl = NaN([n bSz]);
            bu = NaN([n bSz]);
    
            % Iterate over the batch entries.
            for i=1:bSz
                % Obtain parameters of the i-th batch entry.
                Ai = double(gather(A(:,:,i)));
                bi = double(gather(b(:,i)));
                ci = double(gather(c(:,i)));
                Gi = double(gather(G(:,:,i)));
                if all(~isnan(Ai),'all') && all(~isnan(bi),'all')
                    % Construct linear program.
                    prob = struct('Aineq',Ai,'bineq',bi, ...
                        'lb',-ones([q 1]),'ub',ones([q 1]));
                    % Loop over the dimensions.
                    for j=1:n
                        % Find the lower bound for the j-th dimension.
                        prob.f = Gi(j,:);
                        % Solve the linear program.
                        [~,lij,efl] = CORAlinprog(prob);
                        % Find the upper bound for the j-th dimension.
                        prob.f = -Gi(j,:);
                        % Solve the linear program
                        [~,uij,efu] = CORAlinprog(prob);

                        if efl > 0 && efu > 0
                            % Solutions found; assign values.
                            l(j,i) = ci(j) + lij;
                            u(j,i) = ci(j) - uij;
                        else
                            % No solution; the constrained zonotope is empty.
                            break;
                        end
                    end

                    % Loop over the generators.
                    for j=1:q
                        % Find the lower bound for the j-th dimension of 
                        % the constrained hypercube.
                        prob.f = double((1:q) == j);
                        % Solve the linear program.
                        [~,blij,efbl] = CORAlinprog(prob);
                        % Find the upper bound for the j-th dimension of 
                        % the constrained hypercube.
                        prob.f = -double((1:q) == j);
                        % Solve the linear program
                        [~,buji,efbu] = CORAlinprog(prob);

                        if efbl > 0 && efbu > 0
                            % Solutions found; assign values.
                            bl(j,i) = blij;
                            bu(j,i) = -buji;
                        else
                            % No solution; the constrained zonotope is empty.
                            break;
                        end
                    end
                end
            end
            % Add the approximation error.
            l = l - dr;
            u = u + dr;
        otherwise 
            % Invalid option.
            throw(CORAerror('CORA:wrongFieldValue', ...
                'options.nn.conzonotope_bounding_method', ...
                    {'fourier-motzkin','dual-iter','exact'}));
    end
end

function [bl,bu] = aux_fourierMotzkinApproximation(A,b,options)
    % Compute the bounds [bl,bu] of a bounded polytope P:
    % Given P=(A,b) \cap [-1,1], compute its bounds, i.e., 
    % [bl,bu]\supseteq {x\in\R^q\mid A\,x\leq b} \cap [-1,1].

    % Specify a numerical tolerance to avoid numerical instability.
    tol = 1e-8;

    % Efficient approximation by isolating the i-th variable. -------------
    % We compute a box-approximation of the valid factor for the 
    % constraint zonotope, 
    % i.e., [\underline{\beta},\overline{\beta}] 
    %   \supseteq \{\beta \in [-1,1]^q \mid A\,\beta\leq b\}.
    % We view each constraint separately and use the tightest 
    % bounds. For each constraint A_{(i,\cdot)}\,\beta\leq b_{(i)}, 
    % we isolate each factor \beta_{(j)} and extract bounds:
    % A_{(i,\cdot)}\,\beta\leq b_{(i)} 
    %   \implies A_{(i,j)}\,\beta_{(j)} \leq 
    %       b_{(i)} - \sum_{k=1,...,q, k\neq j} A_{(i,k)}\,\beta_{(k)}
    % Based on the sign of A_{(i,j)} we can either tighten the 
    % lower or upper bound of \beta_{(j)}.
    
    % Specify maximum number of iterations.
    maxIter = options.nn.polytope_bound_approx_max_iter;
    
    % Initialize bounds of the factors.
    bl = -ones(size(A,[2 3]),'like',A);
    bu = ones(size(A,[2 3]),'like',A);
    
    % Permute the dimension of the constraints for easier handling.
    A_ = permute(A,[2 1 3]);
    b_ = permute(b,[3 1 2]);
    % Reshape factor bounds for easier multiplication.
    bl_ = permute(bl,[1 3 2]);
    bu_ = permute(bu,[1 3 2]);
        
    % Initialize iteration counter.
    iter = 1;
    tighterBnds = 1;
    while tighterBnds && iter <= maxIter
        % Scale the matrix entries with the current bounds.
        ABnd = A_.*((A_ > 0).*bl_ + (A_ < 0).*bu_);
        % Isolate the i-th variable of the j-th constraint. Sum all
        % variables execpt the i-th. Compute right-hand side of the 
        % inequalities.
        rh = min(max((b_ - (sum(ABnd,1) - ABnd))./A_,bl_),bu_);
        % Update the lower bounds.
        bnd = repmat(bl_,1,size(A,1),1);
        bnd(A_ < 0) = rh(A_ < 0);
        bl_ = max(bnd,[],2);
        % Update the upper bounds.
        bnd = repmat(bu_,1,size(A,1),1);
        bnd(A_ > 0) = rh(A_ > 0);
        bu_ = min(bnd,[],2);
        % Check if the bounds could be tightened.
        tighterBnds = any( ...
            (bl + tol < bl_(:,:) | bu_(:,:) < bu - tol) ... tighter bounds
                & bl_(:,:) <= bu_(:,:), ... not empty
            'all');
        bl = reshape(bl_, size(bl));
        bu = reshape(bu_, size(bu));
        % Increment iteration counter.
        iter = iter + 1;
    end
end

function h = aux_boundConZonotopeSupportFunction(A,b,G,bl,bu,options)
    % We compute bounds of a constrained zonotope by computing a bound on 
    % the support function in both directions for each dimension. 
    % For a constrained zonotope (c,G,A,b) we the bounds 
    % [l_{(i)},u_{(i)}] for the i-th dimension are
    %
    % [Lower Bound]     l_{(i)} = c_{(i)} + min G_{(i,.)}\,\beta 
    %                                       s.t. A\,\beta <= b
    %                                              -\beta <= 1
    %                                               \beta <= 1
    %
    % [Upper Bound]     u_{(i)} = c_{(i)} - min -G_{(i,.)}\,\beta 
    %                                       s.t. A\,\beta <= b
    %                                              -\beta <= 1
    %                                               \beta <= 1
    %
    % We bound support function of a constrained zonotope by optimizing the
    % dual variables. By the weak duality every dual solution is 
    % automatically a bound on the primal solution. The dual problems with 
    % variables \lambda are:
    % (Additional variables \mu and \nu are required for the bounds of
    % \beta, i.e., -1 <= \beta <= 1.)
    %
    % [Lower Bound]     max -b^T\,\lambda - 1^T\,(\mu + \nu)
    %                   s.t. A^T\,\lambda - \mu + \nu = -G_{(i,.)}^T
    %                               \lambda >= 0, \mu >= 0, \nu >= 0
    % 
    % [Upper Bound]     max -b^T\,\lambda - 1^T\,(\mu + \nu)
    %                   s.t. A^T\,\lambda - \mu + \nu = G_{(i,.)}^T
    %                               \lambda >= 0, \mu >= 0, \nu >= 0
    %
    % We can reduce the variables \mu and \nu: For a fixed \lambda >= 0; 
    % from the constraints, we have
    %
    % [Lower Bound] -\mu + \nu = -A^T\,\lambda - G_{(i,.)}^T, and 
    %
    % [Upper Bound] -\mu + \nu = -A^T\,\lambda + G_{(i,.)}^T. 
    %
    % Therefore, \mu and \nu are optimized by (for both bounds)
    %
    % [Lower Bound]    max -1^T\,(\mu + \nu) = -1^T\,\abs(-A^T\,\lambda - G_{(i,.)}^T)\,1
    %                  s.t. \mu,\nu >= 0                
    % 
    % [Upper Bound]    max -1^T\,(\mu + \nu) = -1^T\,\abs(-A^T\,\lambda + G_{(i,.)}^T)\,1
    %                  s.t. \mu,\nu >= 0
    %
    % Thus, we can use the weak duality to obtain bounds on l_{(i)} and 
    % u_{(i)} with any \lambda >= 0 by
    %
    % [Lower Bound]    l_{(i)} >= c_{(i}) - min_{\lambda >= 0} \lambda^T\,b + \abs{-G_{(i,.)}^T - A^T\,\lambda}\,1 
    %
    % [Upper Bound]    u_{(i)} <= c_{(i}) + min_{\lambda >= 0} \lambda^T\,b + \abs{G_{(i,.)}^T - A^T\,\lambda}\,1  
    %
    % We use gradient-based optimization to optimize \lambda. The problem
    % is convex, but non-smooth, i.e., the optimization objective is
    %   \phi(\lambda) = \lambda^T\,b + \abs{+/-G_{(i,.)}^T - A^T\,\lambda}\,1,
    % for \lambda >= 0.
    % We use projected ADAM with subgradients of the true (non-smooth)
    % objective. A subgradient of \abs{.} is \sign{.}, so the subgradient
    % of \phi is
    %   \partial_{\lambda} \phi(\lambda) = b - A\,\sign(+/-G_{(i,.)}^T - A^T\,\lambda).
    % This approach follows Wong & Kolter (2018) and De Palma et al. (2021)
    % who use ADAM for dual LP relaxation optimization in NN verification.

    % Obtain the number of dimensions, generators, and the batch size.
    [n,~,bSz] = size(G);
    % Obtain the number of constraints.
    [p,~,~] = size(A);
    % Reshape for easier batch-wise computations.
    b = reshape(b,[p 1 bSz]);

    % Compute scaling factors for the constraints (rows of A).
    % This prevents certain constraints from dominating the gradient
    % purely due to magnitude.
    scalingA = 1./(max(1,sqrt(sum(A.^2,2))));
    A = scalingA.*A;
    b = scalingA.*b;

    % Obtain the number of iterations.
    maxIter = options.nn.conzonotope_bound_max_iter;

    % Define strong convexity parameter (1e-6 to 1e-4 is usually safe).
    % We anneal mu over iterations: large early (smoothing for faster
    % convergence) and near-zero later (tighter bounds).
    mu = 1e-5;
    % Smoothing parameter for the subgradient approximation.
    % Using x./(abs(x)+epsSmooth) instead of sign(x) provides gradient
    % information near zero, improving ADAM's moment estimates.
    epsSmooth = cast(1e-6,'like',G);
    % Construct a function handle for the optimization objective.
    f = @(x,muK) reshape(pagemtimes(x,'transpose',b,'none') ...
        + sum(abs(G - pagemtimes(x,'transpose',A,'none')),2),n,bSz,[]) ...
        + (muK/2)*reshape(sum(x.^2,1),n,bSz,[]);
    % Construct a function handle for the smoothed subgradient.
    df = @(x,muK) b - pagemtimes(A,'none', ...
        (G - pagemtimes(x,'transpose',A,'none')) ./ ...
        (abs(G - pagemtimes(x,'transpose',A,'none')) + epsSmooth), ...
        'transpose') ...
        + muK*x;

    % Warm-start the dual variables using the Fourier-Motzkin bounds.
    % For each dimension i, the unconstrained maximizer of G_{(i,:)} beta
    % within the FM box [bl,bu] is:
    %   beta*_j = bu_j if G_{(i,j)} > 0,  bl_j if G_{(i,j)} < 0.
    % Constraints violated by beta* are the ones that need positive dual
    % variables (by complementary slackness). We initialize lambda
    % proportional to the constraint violation.
    bl_ = permute(bl,[1 3 2]); % [q x 1 x bSz]
    bu_ = permute(bu,[1 3 2]); % [q x 1 x bSz]
    Gp = permute(G,[2 1 3]); % [q x n x bSz]
    betaStar = (Gp > 0).*bu_ + (Gp <= 0).*bl_; % [q x n x bSz]
    slack = b - pagemtimes(A,betaStar); % [p x n x bSz]
    % Set lambda proportional to violation (slack < 0 means violated).
    l = max(-slack, 0);

    % ADAM parameters.
    lr = options.nn.conzonotope_bound_step_size;
    beta1 = 0.9;
    beta2 = 0.999;
    epsAdam = 1e-8;
    % Initialize first and second moment estimates.
    mAdam = zeros([p n bSz],'like',G);
    vAdam = zeros([p n bSz],'like',G);
    % Counter for stagnation-based momentum restart.
    stagnationCount = 0;

    % Track the best objective value found during optimization.
    % We evaluate with mu=0 (true dual objective) for fair comparison
    % across iterations with different muK values.
    bestF = f(l,0);

    % Do an iterative optimization of the dual variables.
    for iter = 1:maxIter
        % Anneal the regularization parameter with cosine schedule.
        muK = mu * 0.5 * (1 + cos(pi * (iter - 1) / maxIter));

        % Compute the smoothed subgradient at the current position.
        grad = df(l,muK);

        % ADAM moment updates.
        mAdam = beta1*mAdam + (1 - beta1)*grad;
        vAdam = beta2*vAdam + (1 - beta2)*grad.^2;
        % Bias-corrected moment estimates.
        mHat = mAdam / (1 - beta1^iter);
        vHat = vAdam / (1 - beta2^iter);

        % Cosine annealing learning rate schedule.
        lrK = lr * 0.5 * (1 + cos(pi * (iter - 1) / maxIter));

        % ADAM step (minimization).
        l = l - lrK * mHat ./ (sqrt(vHat) + epsAdam);
        % Project onto the feasible set, i.e., lambda >= 0.
        l = max(l, 0);

        % Track the best solution (since subgradient methods are not
        % monotone, we keep the best iterate). Evaluate with mu=0 (true
        % dual objective) for consistent comparison across iterations.
        fl = f(l,0);
        improved = fl < bestF;
        bestF(improved) = fl(improved);

        % Restart ADAM momentum on stagnation to escape cycles.
        if any(improved,'all')
            stagnationCount = 0;
        else
            stagnationCount = stagnationCount + 1;
            if stagnationCount >= 4
                mAdam(:) = 0;
                vAdam(:) = 0;
                stagnationCount = 0;
            end
        end
    end
    % Compute the final bound on the support function using the best
    % dual variables found.
    h = bestF;
end

% ------------------------------ END OF CODE ------------------------------
