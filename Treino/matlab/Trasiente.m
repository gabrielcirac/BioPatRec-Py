

function [data] = Trasiente(onda)
    idx = onda>-0.03 & onda < 0.05 ; 
    onda(idx) = [];
    data = onda;
    plot(data)
end


