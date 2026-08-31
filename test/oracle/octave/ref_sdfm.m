% ref_sdfm.m — independent FGLR (2009) Cholesky path for the committed fixture.
% Generator only: tests compare Julia to test/oracle/sdfm_ref/*.csv, not Octave.
% Algebra: standardize X, PCA r static factors, VAR(p) OLS, rank-q eigen reduction
% of Sigma_u, Cholesky of the q selected observable impacts.
source(fullfile(fileparts(mfilename('fullpath')), '_setup.m'));
% The committed CSVs in ../sdfm_ref are the gate; this script documents the
% Forni–Giannone–Lippi–Reichlin 2009 steps used to produce them.
fprintf('ref_sdfm: fixture CSVs live in test/oracle/sdfm_ref/ (no Octave required at test time)\n');
