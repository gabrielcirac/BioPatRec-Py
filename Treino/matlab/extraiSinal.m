function [csv] = extraiSinal(recSession)
   
  mov1 = recSession.tdata(:, :, 1);
  mov1 = reshape(mov1, 288000, 1);
  plot(mov1)
  t1 = Trasiente(mov1);
  celula = criaAmostras(t1);
  csv = celula;
end