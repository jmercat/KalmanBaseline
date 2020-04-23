function [b,c]=find_sorted(x,range)
%findInSorted fast binary search replacement for ismember(A,B) for the
%special case where the first input argument is sorted.
%   
%   [a,b] = findInSorted(x,s) returns the range which is equal to s. 
%   r=a:b and r=find(x == s) produce the same result   
%  
%   [a,b] = findInSorted(x,[from,to]) returns the range which is between from and to
%   r=a:b and r=find(x >= from & x <= to) return the same result
%
%   For any sorted list x you can replace
%   [lia] = ismember(x,from:to)
%   with
%   [a,b] = findInSorted(x,[from,to])
%   lia=a:b
%
%   Examples:
%
%       x  = 1:99
%       s  = 42
%       r1 = find(x == s)
%       [a,b] = myFind(x,s)
%       r2 = a:b
%       %r1 and r2 are equal
%
%   See also FIND, ISMEMBER.
%
% Author Daniel Roeske <danielroeske.de>

A=range(1);
B=range(end);
a=1;
b=numel(x);
c=1;
d=numel(x);
if A<=x(1)
   b=a;
end
if B>=x(end)
    c=d;
end
while (a+1<b)
    lw=(floor((a+b)/2));
    if (x(lw)<A)
        a=lw;
    else
        b=lw;
    end
end
while (c+1<d)
    lw=(floor((c+d)/2));
    if (x(lw)<=B)
        c=lw;
    else
        d=lw;
    end
end
end