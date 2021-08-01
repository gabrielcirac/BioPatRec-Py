function [celula] = criaAmostras(onda)
    celula = {100};
    desloca = 1;
    for k = 1 : 100
        celula{1, k} = onda(desloca:800*k);
        desloca = desloca +800;
        celula{1, k} = transpose(celula{1, k});
    end
   
    celula = transpose(celula)
   
end