% test/oracle/octave/_setup.m — add the BVAR_ reference toolbox to the Octave path.
% Sourced (relative to repo-root pwd) by every oracle script:  source('test/oracle/octave/_setup.m')
% Run octave from the repository root so the relative paths below resolve.
REF = '/Users/chung/Downloads/BVAR_-master-2';   % reference toolbox (Ferroni & Canova)
addpath(fullfile(REF, 'bvartools'));
addpath(fullfile(REF, 'cmintools'));
pkg load statistics;   % Octave: Wishart / quantile helpers used by the toolbox
OUTDIR  = fullfile(pwd, 'test', 'oracle', '_out');
DATADIR = fullfile(pwd, 'test', 'oracle', '_data');
if ~exist(OUTDIR,  'dir'); mkdir(OUTDIR);  end
if ~exist(DATADIR, 'dir'); mkdir(DATADIR); end
