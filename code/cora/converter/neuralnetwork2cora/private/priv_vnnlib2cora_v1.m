function [X0, spec] = priv_vnnlib2cora_v1(file)
% priv_vnnlib2cora_v1 - import specifications from VNN-LIB 1.x .vnnlib files
%
% Syntax:
%    [X0,spec] = priv_vnnlib2cora_v1(file)
%
% Inputs:
%    file - path to a file .vnnlib file storing the specification
%
% Outputs:
%    X0 - initial set represented as an object of class interval
%    spec - specifications represented as an object of class specification
%
% Reference:
%    - https://www.vnnlib.org/
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: vnnlib2cora, specification

% Authors:       Niklas Kochdumper, Tobias Ladner
% Written:       23-November-2021
% Last update:   26-July-2023 (TL, speed up)
%                30-August-2023 (TL, bug fix multiple terms in and)
%                14-June-2024 (TL, major speed up)
%                19-April-2026 (BK, moved into private/ as v1 backend)
%                27-June-2026 (TL, O(n) cursor-based parser)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% read in text from file
text = fileread(file);
% Numeric code view of the text for fast per-character scanning. The parser
% walks an integer cursor over 't'; it never re-slices the (multi-MB) string,
% which is what made the previous implementation O(n^2) on specs with many
% disjuncts. 'text' is only indexed to extract the few characters of a token
% when a numeric value/index actually has to be parsed.
t = double(text);
n = numel(t);

% determine number of inputs and number of outputs
nrInputs = 0;
nrOutputs = 0;
lineBreaks = strfind(text, newline);
for ln = 1:(numel(lineBreaks)-1)
    % iterate through file
    i = lineBreaks(ln)+1;
    i1 = lineBreaks(ln+1);
    if startsWith(text(i:i1), '(declare-const ')
        declText = text(i+15:i1);
        ind = find(declText == ' ');
        declText = declText(1:ind(1)-1);
        if strcmp(declText(1), 'X')
            % found new input; read out index
            nrInputs = max(nrInputs, str2double(declText(3:end)));
        elseif strcmp(declText(1), 'Y')
            % found new output; read out index
            nrOutputs = max(nrOutputs, str2double(declText(3:end)));
        end
    end
end

% +1 due to 0-indexing in vnnlib files
data.nrInputs = nrInputs + 1;
data.nrOutputs = nrOutputs + 1;
data.currIn = 0;

% parse file
data.polyInput = [];
data.polyOutput = [];

ASSERT = double('(assert');
pos = 1;
while pos <= n
    pos = aux_skipws(t, pos, n);
    if pos > n
        break;
    end
    if pos+6 <= n && isequal(t(pos:pos+6), ASSERT)
        pos = pos + 7;
        pos = aux_skipws(t, pos, n);
        [pos, data] = aux_parseExpr(t, text, pos, n, data);
        % consume the closing ')' of the assert statement
        pos = aux_skipws(t, pos, n);
        if pos <= n && t(pos) == 41 % ')'
            pos = pos + 1;
        end
    else
        % not an assert (e.g. declare-const / comment / blank): skip the line
        while pos <= n && t(pos) ~= 10 % newline
            pos = pos + 1;
        end
        pos = pos + 1;
    end
end

% convert data to polytopes ---

% a) convert input
% potentially convert input polytopes to intervals
X0 = cell(1, length(data.polyInput));
for i = 1:length(X0)
    polyStruct = data.polyInput(i);
    P = polytope(polyStruct.C, polyStruct.d);
    [res, I] = representsa_(P, 'interval', eps);

    if res
        X0{i} = I;
    else
        throw(CORAerror('CORA:specialError', 'Input set is not an interval.'))
    end
end

% b) convert output
Y = cell(1, length(data.polyOutput));
for i = 1:length(data.polyOutput)
    Y{i} = polytope(data.polyOutput(i).C, data.polyOutput(i).d);
end

% construct specification from list of output polytopes
if isempty(Y)
    throw(CORAerror("CORA:converterIssue",sprintf('Unable to convert file: %s', file)));
elseif isscalar(Y)
    % We have a single unsafe set.
    spec = specification(Y{1}, 'unsafeSet');
else
    % We have a union of unsafe sets, i.e., any point contained in any set
    % is a counter example.
    % We want to reduce the number of specification sets. Therefore, we
    % try to convert to the union of unsafe sets to a union of safe sets.
    Y_safe = safeSet2unsafeSet(Y); % We can use the function; safe vs. unsafe is juts
    % Check if we could reduce the number of specifications.
    if length(Y_safe) < length(Y)
        % We could reduce the number of specifications; we add the sets
        % as unsafe sets.
        Y = Y_safe;
        type = 'safeSet'; % Type of union, e.g. Y is a union of safe sets
    else
        % We could not reduce the number of specifications; we add the
        % sets as unsafe sets.
        type = 'unsafeSet'; % Type of union, e.g. Y is a union of unsafe sets
    end
    % Initialize the result
    spec = [];
    % Add all the specifications together. We can only add inverse
    % sets, because then all specification have to hold simultaneously.
    for i = 1:length(Y)
        switch type
            case 'safeSet'
                % We have to convert the safe set to an unsafe set.
                Yi_safe = specification(Y{i},'safeSet');
                % All inverted sets unsafe sets have to be avoided.
                spec = add(spec,Yi_safe);
            case 'unsafeSet'
                % We have to convert the unsafe set to an safe set.
                Yi_unsafe = specification(Y{i},'unsafeSet');
                % We add the unsafe set.
                spec = add(spec,Yi_unsafe);
        end
    end
end

end


% Auxiliary functions -----------------------------------------------------

function pos = aux_skipws(t, pos, n)
% advance the cursor past spaces, tabs, newlines and carriage returns
while pos <= n
    c = t(pos);
    if c == 32 || c == 10 || c == 13 || c == 9
        pos = pos + 1;
    else
        break;
    end
end
end

function e = aux_tokenEnd(t, pos, n)
% first index >= pos that ends a bare token (whitespace or closing paren)
e = pos;
while e <= n
    c = t(e);
    if c == 32 || c == 41 || c == 10 || c == 13 || c == 9
        break;
    end
    e = e + 1;
end
end

function [pos, data] = aux_parseExpr(t, text, pos, n, data)
% parse one S-expression starting at '(' and leave the cursor just past its
% matching closing ')'

if pos > n || t(pos) ~= 40 % '('
    throw(CORAerror('CORA:converterIssue', ...
        sprintf('Failed to parse vnnlib file. Parsed up to position %i.', pos)))
end

c = t(pos+1);
if c == 60 || c == 62 % '<' or '>' -> linear constraint
    [pos, data] = aux_parseLinearConstraint(t, text, pos, n, data);

elseif c == 111 % 'o' -> (or ...)
    pos = pos + 3; % past '(or'
    while true
        pos = aux_skipws(t, pos, n);
        if pos > n
            break;
        end
        if t(pos) == 41 % ')'
            pos = pos + 1;
            break;
        end

        % parse one or-condition into a fresh sub-context
        data_ = data;
        data_.polyInput = [];
        data_.polyOutput = [];
        data_.currIn = 0;

        [pos, data_] = aux_parseExpr(t, text, pos, n, data_);

        % update input conditions
        if ~isempty(data_.polyInput)
            if ~isempty(data.polyInput)
                data.polyInput(end+1) = data_.polyInput(1);
            else
                data.polyInput = data_.polyInput;
            end
        end

        % update output conditions
        if ~isempty(data_.polyOutput)
            if ~isempty(data.polyOutput)
                data.polyOutput(end+1) = data_.polyOutput(1);
            else
                data.polyOutput = data_.polyOutput;
            end
        end
    end

elseif c == 97 % 'a' -> (and ...)
    pos = pos + 4; % past '(and'
    while true
        pos = aux_skipws(t, pos, n);
        if pos > n
            break;
        end
        if t(pos) == 41 % ')'
            pos = pos + 1;
            break;
        end
        [pos, data] = aux_parseExpr(t, text, pos, n, data);
    end

else
    throw(CORAerror('CORA:converterIssue', ...
        sprintf('Failed to parse vnnlib file. Parsed up to position %i.', pos)))
end
end

function S = aux_createPolytopeStruct(n)
S = struct;
S.C = zeros(2*n,n);
S.d = zeros(2*n,1);
end

function [pos, data] = aux_parseLinearConstraint(t, text, pos, n, data)
% parse a linear constraint '(<= arg1 arg2)' / '(>= arg1 arg2)'

% extract operator ('<' for '<=', '>' for '>=') and step past '(<='/'(>='
opLeq = (t(pos+1) == 60); % '<'
pos = pos + 3;
pos = aux_skipws(t, pos, n);

% get type of constraint (on inputs X or on outputs Y)
type = aux_getTypeOfConstraint(t, pos, n);

% initialization
if strcmp(type, 'input')
    C = zeros(1, data.nrInputs);
else
    C = zeros(1, data.nrOutputs);
end
d = 0;

% parse first argument
[C1, d1, pos] = aux_parseArgument(t, text, pos, n, C, d);
pos = aux_skipws(t, pos, n);

% parse second argument
[C2, d2, pos] = aux_parseArgument(t, text, pos, n, C, d);
pos = aux_skipws(t, pos, n);

% consume the closing ')' of the constraint
if pos <= n && t(pos) == 41 % ')'
    pos = pos + 1;
end

% combine the two arguments
if opLeq
    C = C1 - C2;
    d = d2 - d1;
else
    C = C2 - C1;
    d = d1 - d2;
end

% combine the current constraint with previous constraints
if strcmp(type, 'input')
    if isempty(data.polyInput)
        data.polyInput = aux_createPolytopeStruct(data.nrInputs);
    end

    data.currIn = data.currIn+1;

    for i = 1:length(data.polyInput)
        data.polyInput(i).C(data.currIn,:) = C;
        data.polyInput(i).d(data.currIn) = d;
    end

else % output
    if isempty(data.polyOutput)
        data.polyOutput = aux_createPolytopeStruct(0);
    end

    for i = 1:length(data.polyOutput)
        data.polyOutput(i).C = [data.polyOutput(i).C; C];
        data.polyOutput(i).d = [data.polyOutput(i).d; d];
    end
end
end

function [C, d, pos] = aux_parseArgument(t, text, pos, n, C, d)
% parse next argument and advance the cursor past it

c = t(pos);

if c == 88 || c == 89 % 'X' or 'Y'

    e = aux_tokenEnd(t, pos, n);
    % token is '<X|Y>_<index>'; the numeric index starts after the '_'
    index = str2double(text(pos+2:e-1)) + 1;
    C(index) = C(index) + 1;
    pos = e;

elseif c == 40 % '(' -> '(+ ...)' or '(- ...)'

    op = t(pos+1); % '+' (43) or '-' (45)
    pos = pos + 2;
    pos = aux_skipws(t, pos, n);

    % parse first argument
    [C1, d1, pos] = aux_parseArgument(t, text, pos, n, C, d);
    pos = aux_skipws(t, pos, n);

    % parse second argument
    [C2, d2, pos] = aux_parseArgument(t, text, pos, n, C, d);
    pos = aux_skipws(t, pos, n);

    % consume the closing ')'
    if pos <= n && t(pos) == 41
        pos = pos + 1;
    end

    % combine both arguments
    if op == 43 % '+'
        C = C1 + C2;
        d = d1 + d2;
    else % '-'
        C = C1 - C2;
        d = d1 - d2;
    end

else % numeric constant

    e = aux_tokenEnd(t, pos, n);
    d = d + str2double(text(pos:e-1));
    pos = e;
end
end

function type = aux_getTypeOfConstraint(t, pos, n)
% check if the current constraint is on the inputs or on the outputs by
% scanning forward to the first variable ('X' -> input, 'Y' -> output)

i = pos;
while i <= n
    c = t(i);
    if c == 88 % 'X'
        type = 'input';
        return;
    elseif c == 89 % 'Y'
        type = 'output';
        return;
    end
    i = i + 1;
end

% neither X nor Y found
throw(CORAerror('CORA:notSupported', 'File format not supported'));
end

% ------------------------------ END OF CODE ------------------------------
