function cora_server(srvDir)
%CORA_SERVER  Persistent CORA verification daemon for the VNN-COMP background server.
%   MATLAB + CORA have a slow per-script startup. Instead of paying it for every instance, this
%   daemon is started ONCE (by cora_server.sh, which prepare_instance.sh launches lazily) and
%   then services one job at a time as files written by prepare_instance.sh / run_instance.sh.
%   It dispatches to CORA's real entry points:
%       type=prepare -> prepare_instance(bench,onnx,vnnlib[,overrides])  (parses + writes .mat)
%       type=run     -> run_instance(bench,onnx,vnnlib,<result>,timeout,true)  (verifies)
%   run_instance.m writes the verdict file itself (incl. v1/v2 counterexample formatting) and
%   enforces the per-instance timeout internally; prepare_instance.m returns a skip/ok code.
%
%   It is deliberately UNAWARE of process leases / teardown: if the owning *_instance.sh is
%   killed, the cora_server.sh supervisor kills THIS process for a clean restart. See README.md.
%
%   File protocol (all paths under the ABSOLUTE srvDir, so the per-job cd below never hides them):
%     ping     -> daemon deletes it and touches `pong`        (liveness, answered when idle)
%     request  -> job spec, key=value lines: id, type, cwd, bench, onnx, vnnlib, timeout, overrides
%     running  -> daemon writes <id> while a job is in flight  (the lease watch trigger)
%     job.log  -> per-job output, relayed live to the website by the owning *_instance.sh
%     result   -> verdict (written by run_instance.m for a run job)
%     prep_rc  -> the job's return code (prepare's skip/ok code; 0 for a normal run)
%     done     -> <id>, written LAST and atomically; signals the owner the job is finished

    if nargin < 1 || isempty(srvDir)
        srvDir = fullfile(getenv('HOME'), '.cora_server');
    end
    if ~isfolder(srvDir); mkdir(srvDir); end
    cleanupStale(srvDir);                 % don't treat a dead daemon's leftovers as live work
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
            job = readJob(reqPath);
            try delete(reqPath); catch; end
            writeAtomic(srvDir, 'running', job.id);

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
                resetState();               % make a warm job behave like a fresh MATLAB
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
                        error('cora_server:badType', 'unknown job type "%s"', job.type);
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

            writeAtomic(srvDir, 'prep_rc', num2str(rc));
            try delete(fullfile(srvDir, 'running')); catch; end
            writeAtomic(srvDir, 'done', job.id);   % LAST: the owner waits on this
        end
        pause(0.05);
    end
end

function resetState()
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

function cleanupStale(srvDir)
    for f = {'running', 'done', 'result', 'request', 'prep_rc', 'pong'}
        p = fullfile(srvDir, f{1});
        if isfile(p); try delete(p); catch; end; end
    end
end

function job = readJob(reqPath)
    job = struct('id','', 'type','', 'cwd','', 'bench','', 'onnx','', 'vnnlib','', ...
        'timeout',0, 'overrides','');
    lines = splitlines(string(fileread(reqPath)));
    for i = 1:numel(lines)
        kv = split(lines(i), '=');
        if numel(kv) < 2; continue; end
        key = strtrim(kv(1)); val = strtrim(strjoin(kv(2:end), '='));
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

function writeAtomic(srvDir, name, contents)
    tmp = fullfile(srvDir, [name '.tmp']);
    fid = fopen(tmp, 'w'); fprintf(fid, '%s', contents); fclose(fid);
    movefile(tmp, fullfile(srvDir, name));
end
