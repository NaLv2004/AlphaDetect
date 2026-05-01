function [M,IND] = combn(V,N)
narginchk(2,2);
if isempty(V) || N == 0
    M = reshape(V([]), 1, 0);
    IND = [] ;
elseif fix(N) ~= N || N < 1 || numel(N) ~= 1
    error('combn:negativeN','Second argument should be a positive integer') ;
elseif N==1
    M = V' ;
    IND = 1:numel(V) ;
else
    % speed depends on the number of output arguments
    if nargout<2
        M = local_allcomb(V,N) ;
    else
        % indices requested
        IND = local_allcomb(1:numel(V),N) ;
        M = V(IND) ;
    end
end

% LOCAL FUNCTIONS

function Y = local_allcomb(X,N)
% See ALLCOMB, available on the File Exchange
if N>1
    % create a list of all possible combinations of N elements
    [Y{N:-1:1}] = ndgrid(X) ;
    % concatenate into one matrix, reshape into 2D and flip columns
    Y = reshape(cat(N+1,Y{:}),[],N) ;
else
    % no combinations have to be made
    Y = X';
end