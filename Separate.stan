data {
  int<lower=1> N1; 
  int<lower=1> N2; 
  int<lower=1> N3; 
  int<lower=1> N_features; 
  matrix[N1, N_features] x1;
  matrix[N2, N_features] x2;
  matrix[N3, N_features] x3;
  vector[N1] y1;
  vector[N2] y2;
  vector[N3] y3;
}

parameters {
  real alpha1;
  vector[N_features] beta1;
  real alpha2;
  vector[N_features] beta2;
  real alpha3;
  vector[N_features] beta3;
  
  real<lower=0> sigma;
}

model {
  alpha1 ~ normal(1000, 100);
  beta1 ~ normal(0, 10);
  alpha2 ~ normal(1000, 100);
  beta2 ~ normal(0, 10);
  alpha3 ~ normal(1000, 100);
  beta3 ~ normal(0, 10);
  
  sigma ~ normal(0, 10);
  
  y1 ~ normal(alpha1 + x1*beta1, sigma);
  y2 ~ normal(alpha2 + x2*beta2, sigma);
  y3 ~ normal(alpha3 + x3*beta3, sigma);
}

generated quantities {
  array[N1] real y1_rep = normal_rng(alpha1 + x1*beta1, sigma);
  array[N2] real y2_rep = normal_rng(alpha2 + x2*beta2, sigma);
  array[N3] real y3_rep = normal_rng(alpha3 + x3*beta3, sigma);
  
  vector[N1+N2+N3] log_lik;
  
  for (i in 1:N1) {
    log_lik[i] = normal_lpdf(y1[i] | x1[i] * beta1 + alpha1, sigma);
  }
  for (i in 1:N2) {
    log_lik[i+N1] = normal_lpdf(y2[i] | x2[i] * beta2 + alpha2, sigma);
  }
  for (i in 1:N3) {
    log_lik[i+N1+N2] = normal_lpdf(y3[i] | x3[i] * beta3 + alpha3, sigma);
  }
}





