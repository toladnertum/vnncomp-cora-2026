function cora_server(srvDir)
% cora_server - persistent CORA verification daemon for the VNN-COMP
%    background server.
%
%    MATLAB + CORA have a slow per-script startup. Instead of paying it for
%    every instance, this daemon is started ONCE (by cora_server.sh, which
%    prepare_instance.sh launches lazily) and then services one job at a time
%    as files written by prepare_instance.sh / run_instance.sh. It dispatches
%    to CORA's real entry points:
%       type=prepare -> prepare_instance(bench,onnx,vnnlib[,overrides])
%       type=run     -> run_instance(bench,onnx,vnnlib,<result>,timeout,true)
%    run_instance.m writes the verdict file itself (incl. v1/v2 counterexample
%    formatting) and enforces the per-instance timeout internally;
%    prepare_instance.m returns a skip/ok code.
%
%    It is deliberately UNAWARE of process leases / teardown: if the owning
%    *_instance.sh is killed, the cora_server.sh supervisor kills THIS process
%    for a clean restart. See README.md.
%
%    File channel (all paths under the ABSOLUTE srvDir, so the per-job cd
%    below never hides them):
%       ping     - daemon deletes it and touches `pong` (liveness, when idle)
%       request  - job spec, key=value lines: id, type, cwd, bench, onnx,
%                  vnnlib, timeout, overrides
%       running  - daemon writes <id> while a job is in flight (lease watch)
%       job.log  - per-job output, relayed live to the website by the owner
%       result   - verdict (written by run_instance.m for a run job)
%       prep_rc  - the job's return code (prepare's skip/ok code; 0 for a run)
%       done     - <id>, written LAST and atomically; signals job finished
%
% Syntax:
%    cora_server(srvDir)
%
% Inputs:
%    srvDir - absolute path to the server's working directory (defaults to
%             $HOME/.cora_server); holds the file-based channel files
%
% Outputs:
%    -
%
% Other m-files required: prepare_instance, run_instance
% Subfunctions: aux_resetState, aux_cleanupStale, aux_readJob, aux_writeAtomic
% MAT-files required: none
%
% See also: prepare_instance, run_instance

% Authors:       Tobias Ladner
% Written:       27-June-2026
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

    if nargin < 1 || isempty(srvDir)
        srvDir = fullfile(getenv('HOME'), '.cora_server');
    end
    if ~isfolder(srvDir); mkdir(srvDir); end
    aux_cleanupStale(srvDir);             % don't treat a dead daemon's leftovers as live work
    homeDir = pwd;                        % the daemon's launch dir, to restore after each job
    fprintf('[cora_server] started, watching %s\n', srvDir);

    pingPath = fullfile(srvDir, 'ping');
    reqPath  = fullfile(srvDir, 'request');

    while true
        % --- liveness ping: answered only here, i.e. while the daemon is idle ---
        if isfile(pingPath)
            try delete(pingPath); catch; end
            fclose(fopen(fullfile(srvDir, 'pong'), 'w'));
        end

        if isfile(reqPath)
            job = aux_readJob(reqPath);
            try delete(reqPath); catch; end
            aux_writeAtomic(srvDir, 'running', job.id);

            jobLog = fullfile(srvDir, 'job.log');
            fclose(fopen(jobLog, 'w'));      % truncate; CORA appends below
            % Capture ALL output for the live website relay. diary flushes per line since
            % ~R2013a (undocumented but stable; verified live on R2025b), so the owner's
            % `tail -f` sees output as it is produced.
            diary(jobLog); diary on;
            rc = 0;
            resultPath = fullfile(srvDir, 'result');
            try
                % Run each job from the OWNER's cwd so relative onnx/vnnlib paths resolve and the
                % prepare->run .mat handoff (a bare filename) is written/read consistently.
                if ~isempty(job.cwd) && isfolder(job.cwd); cd(job.cwd); end
                aux_resetState();           % make a warm job behave like a fresh MATLAB
                switch job.type
                    case 'prepare'
                        if isempty(job.overrides)
                            rc = prepare_instance(job.bench, job.onnx, job.vnnlib);
                        else
                            rc = prepare_instance(job.bench, job.onnx, job.vnnlib, job.overrides);
                        end
                    case 'run'
                        run_instance(job.bench, job.onnx, job.vnnlib, ...
                            resultPath, job.timeout, true);   % writes resultPath itself
                    otherwise
                        throw(CORAerror('CORA:specialError', ...
                            sprintf('unknown job type "%s"', job.type)));
                end
            catch err
                fprintf('[cora_server] ERROR: %s\n', getReport(err, 'extended', 'hyperlinks', 'off'));
                rc = 1;
                % Make sure a run job still leaves a result for the owner to copy.
                if strcmp(job.type, 'run') && ~isfile(resultPath)
                    fid = fopen(resultPath, 'w'); fprintf(fid, 'unknown\n'); fclose(fid);
                end
            end
            cd(homeDir);
            diary off;

            aux_writeAtomic(srvDir, 'prep_rc', num2str(rc));
            try delete(fullfile(srvDir, 'running')); catch; end
            aux_writeAtomic(srvDir, 'done', job.id);   % LAST: the owner waits on this
        end
        pause(0.05);
    end
end


% Auxiliary functions -----------------------------------------------------

function aux_resetState()
    % Make each job behave like a freshly started MATLAB so a warm server stays deterministic
    % and does not leak memory across instances.
    rng('default');                        % reset RNG (CORA falsification/sampling) to startup state
    try
        if gpuDeviceCount > 0
            reset(gpuDevice);              % free GPU memory accumulated by the previous instance
        end
    catch
        % no GPU / no Parallel Computing Toolbox -> nothing to reset
    end
end

function aux_cleanupStale(srvDir)
    % remove any leftover per-job channel files from a previous (killed) daemon
    for f = {'running', 'done', 'result', 'request', 'prep_rc', 'pong'}
        p = fullfile(srvDir, f{1});
        if isfile(p); try delete(p); catch; end; end
    end
end

function job = aux_readJob(reqPath)
    % parse the key=value job request file into a job struct
    job = struct('id','', 'type','', 'cwd','', 'bench','', 'onnx','', 'vnnlib','', ...
        'timeout',0, 'overrides','');
    lines = splitlines(string(fileread(reqPath)));
    for i = 1:numel(lines)
        % split each "key=value" line (values may themselves contain '=')
        kv = split(lines(i), '=');
        if numel(kv) < 2; continue; end
        key = strtrim(kv(1)); val = strtrim(strjoin(kv(2:end), '='));
        % store recognised keys; ignore anything unknown
        switch key
            case "id";        job.id = char(val);
            case "type";      job.type = char(val);
            case "cwd";       job.cwd = char(val);
            case "bench";     job.bench = char(val);
            case "onnx";      job.onnx = char(val);
            case "vnnlib";    job.vnnlib = char(val);
            case "timeout";   job.timeout = str2double(val);
            case "overrides"; job.overrides = char(val);
        end
    end
end

function aux_writeAtomic(srvDir, name, contents)
    % write to a staging file then rename, so a reader never sees a partial file
    stagingPath = fullfile(srvDir, [name '.tmp']);
    fid = fopen(stagingPath, 'w'); fprintf(fid, '%s', contents); fclose(fid);
    movefile(stagingPath, fullfile(srvDir, name));
end

% ------------------------------ END OF CODE ------------------------------
