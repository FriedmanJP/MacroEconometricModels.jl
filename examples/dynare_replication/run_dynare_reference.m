% run_dynare_reference.m — Extract Dynare reference values for replication comparison
% Usage: cd examples/dynare_replication && octave --no-gui run_dynare_reference.m
%
% Requires: Dynare 6.5+ (Homebrew), Octave 11+
% Output:   dynare_results/<model_name>.mat for each model

addpath('/opt/homebrew/opt/dynare/lib/dynare/matlab');

dsge_mod_path = '/Users/chung/Desktop/CODES/DSGE_mod';
output_path = fullfile(pwd, 'dynare_results');
[~,~] = mkdir(output_path);

function save_dynare_results(model_name, output_path)
    global oo_ M_;

    result = struct();
    result.steady_state = oo_.steady_state';
    result.endo_names = cellstr(M_.endo_names);
    result.exo_names = cellstr(M_.exo_names);

    % IRFs
    if isfield(oo_, 'irfs')
        fn = fieldnames(oo_.irfs);
        for i = 1:length(fn)
            result.irfs.(fn{i}) = oo_.irfs.(fn{i});
        end
    end

    % Variance-covariance matrix
    if ~isempty(oo_.var)
        result.var_matrix = oo_.var;
    end

    % Variance decomposition
    if isfield(oo_, 'variance_decomposition') && ~isempty(oo_.variance_decomposition)
        result.variance_decomposition = oo_.variance_decomposition;
    end

    % Autocorrelation
    if isfield(oo_, 'autocorr') && ~isempty(oo_.autocorr)
        result.autocorr = oo_.autocorr{1};
    end

    % Estimation outputs (when estimation() was run)
    if isfield(oo_, 'posterior_mode')
        result.posterior_mode = oo_.posterior_mode;
    end
    if isfield(oo_, 'MarginalDensity')
        result.marginal_density = oo_.MarginalDensity;
    end
    if isfield(oo_, 'likelihood_at_initial_parameters')
        result.loglik_at_mode = oo_.likelihood_at_initial_parameters;
    end

    % Parameter names and values (always available)
    result.param_names = cellstr(M_.param_names);
    result.param_values = M_.params;

    save(fullfile(output_path, [model_name '.mat']), '-struct', 'result');
    fprintf('  Saved %s.mat\n', model_name);
end

% Model list: {output_name, directory, mod_file}
models = {
    'rbc_baseline',         'RBC_baseline',         'RBC_baseline.mod';
    'collard_2001',         'Collard_2001',         'Collard_2001_example1.mod';
    'hansen_1985',          'Hansen_1985',           'Hansen_1985.mod';
    'sgu_2003',             'SGU_2003',              'SGU_2003.mod';
    'aguiar_gopinath_2007', 'Aguiar_Gopinath_2007',  'Aguiar_Gopinath_2007.mod';
    'mccandless_ch9',       'McCandless_2008',       'McCandless_2008_Chapter_9.mod';
    'mccandless_ch13',      'McCandless_2008',       'McCandless_2008_Chapter_13.mod';
    'rbc_capitalstock',     'RBC_capitalstock_shock', 'RBC_capitalstock_shock.mod';
    'kiyotaki_moore_1997',  'Kiyotaki_Moore_1997',   'Kiyotaki_Moore_1997.mod';
    'ascari_sbordone_2014', 'Ascari_Sbordone_2014',  'Ascari_Sbordone_2014.mod';
    'gali_2008_ch2',        'Gali_2008',             'Gali_2008_chapter_2.mod';
    'gali_2015_ch2',        'Gali_2015',             'Gali_2015_chapter_2.mod';
    'gali_2015_ch3_nl',     'Gali_2015',             'Gali_2015_chapter_3_nonlinear.mod';
    'fv_2007',              'FV_et_al_2007',         'FV_et_al_2007_ABCD.mod';
    'sgu_2004',             'SGU_2004',              'SGU_2004.mod';
    'jermann_1998',         'Jermann_1998',          'Jermann_1998.mod';
    'rbc_news',             'RBC_news_shock_model',  'RBC_news_shock_model.mod';
    'gi2015_rbc',           'Guerrieri_Iacoviello_2015', 'Guerrieri_Iacoviello_2015_rbc.mod';
    'gi2015_nk',            'Guerrieri_Iacoviello_2015', 'Guerrieri_Iacoviello_2015_nk.mod';
    'gali_2008_ch3',        'Gali_2008',             'Gali_2008_chapter_3.mod';
    'gali_2015_ch3',        'Gali_2015',             'Gali_2015_chapter_3.mod';
    'smets_wouters_2007',   'Smets_Wouters_2007',    'Smets_Wouters_2007.mod';
};

results_summary = {};
for i = 1:size(models, 1)
    model_name = models{i,1};
    model_dir = fullfile(dsge_mod_path, models{i,2});
    mod_file = models{i,3};
    fprintf('[%d/%d] %s/%s ... ', i, size(models,1), models{i,2}, mod_file);
    try
        cd(model_dir);
        evalc(['dynare ' mod_file ' noclearall']);
        save_dynare_results(model_name, output_path);
        results_summary{end+1} = sprintf('OK   %s', model_name);
    catch e
        fprintf('FAILED: %s\n', e.message);
        results_summary{end+1} = sprintf('FAIL %s: %s', model_name, e.message);
    end
end

fprintf('\n=== Summary ===\n');
for i = 1:length(results_summary)
    fprintf('%s\n', results_summary{i});
end
