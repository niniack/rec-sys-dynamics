function [var] = get_fun(Ni_s, Ai, si, mi, p1)

var=[];
if p1 == false
for n = mi-1:Ai-1
    vars = get_fun_var(Ni_s, Ai, si, mi, n, Ai-1);
    var = [var;[vars]];
end
elseif p1 == true
 for n = mi:Ai-1
    vars = get_fun_var(Ni_s, Ai, si, mi, n, Ai-1);
    var = [var;[vars]];
 end 
end