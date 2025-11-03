% compare_impedance_multi_no_base.m
% LabOne MFLI (current channel) — NO base file required
% Produces 8 figures (linear/log X axis configurable) + console summaries.
%
% Figures:
%  1) Current magnitude |I|
%  2) Current phase φI
%  3) Voltage RMS (flat line per file)
%  4) Impedance magnitude |Z|
%  5) Impedance phase ∠Z
%  6) Impedance components: Re(Z)=R and Im(Z)=X (two subplots)
%  7) Current components: Re(I)=X and Im(I)=Y (two subplots)
%  8) RMSD heatmap of |Z| (all-vs-all) + bar chart RMSD(|Z|) to group mean
%
% Notes:
% - X,Y are LabOne demodulated current components (current channel).
% - |I| = hypot(X,Y). Impedance Z = V/I using the same RMS convention as I.
% - Mass [g] and Temp [°C] are auto-parsed from filenames like ..._t11480_23.6.mat
%   (114.80 g) or ..._t115_23.6.mat (115 g). No need to edit arrays.

clear; clc; close all;

%% ================= USER OPTIONS =================
DATA_DIR = 'C:\Users\au585732\OneDrive - Aarhus universitet\Thesis\Datasets\ML_Dmg_Mehdi_Dataset\Datafolder_Drill';

% Get list of all .mat files in the directory
matFiles = dir(fullfile(DATA_DIR, '*.mat'));

% Preallocate cell array for file paths
FILES = cell(1, numel(matFiles));

% Fill FILES with full paths
for k = 1:numel(matFiles)
    FILES{k} = fullfile(DATA_DIR, matFiles(k).name);
end

% Extract the weight number from each filename
weights = zeros(1, numel(FILES));

for k = 1:numel(FILES)
    [~, name, ~] = fileparts(FILES{k}); % get the filename without path/extension
    % Find the part that starts with 't' and extract the number after it
    t_idx = strfind(name, '_t');  % find '_t'
    if ~isempty(t_idx)
        % extract the numeric part after 't' until the next underscore or end of string
        num_str = regexp(name(t_idx+2:end), '\d+', 'match'); 
        if ~isempty(num_str)
            weights(k) = str2double(num_str{1}) / 100; % divide by 100 to get grams
        end
    end
end

% Sort FILES based on weights from highest to lowest
[~, sortIdx] = sort(weights, 'descend');
FILES_sorted = FILES(sortIdx);
weights = weights(sortIdx);

%%

% ---- Per-file parameters (you can give scalars; they auto-expand) ----
V_APPLIED = 0.707;      % [V RMS] (scalar or 1xN array)
XY_in_uA  = false;      % true if s.x/s.y are in microampere in your files; else false for ampere

% ---- Analysis range ----
F_MIN = 50e3;           % [Hz]
F_MAX = 500e3;          % [Hz]

% ---- Plot style & cosmetic smoothing (display only for |Z| & ∠Z) ----
USE_LOG_X  = false;     % false=linear freq (like LabOne), true=log freq
USE_SMOOTH = false;
SMOOTH_WIN = 1;

% ---- Unit sanity messages ----
SHOW_WARN_UNIT_MISMATCH = true;

%% ================= PREP =================
nF = numel(FILES);
if isscalar(V_APPLIED), V_APPLIED = repmat(V_APPLIED,1,nF); end
if numel(V_APPLIED)~=nF, error('V_APPLIED must have %d elements.', nF); end

I_SCALE = XY_in_uA * 1e-6 + (~XY_in_uA) * 1;  % µA→A or A→A

% Auto-parse mass [g] and temperature [°C] from filenames
MASS_G = nan(1,nF);
TEMP_C = nan(1,nF);
for k=1:nF
    [~,nm,~] = fileparts(FILES{k});
    % Look for "..._t<digits>_<temp>.mat"
    tokens = regexp(nm,'_t([0-9]+)_([0-9]+(?:\.[0-9]+)?)$','tokens','once');
    if ~isempty(tokens)
        mass_digits = tokens{1}; temp_str = tokens{2};
        % Convert digit string to mass with implied 2 decimal places when length>=3
        if numel(mass_digits) >= 3
            MASS_G(k) = str2double(mass_digits(1:end-2) + "." + mass_digits(end-2:end));
        else
            MASS_G(k) = str2double(mass_digits);
        end
        TEMP_C(k) = str2double(temp_str);
    else
        % Fallback: attempt another common pattern or leave NaN
        TEMP_C(k) = NaN; MASS_G(k) = NaN;
    end
end

% Legend labels
LEG = cell(1,nF);
for k=1:nF
    [~,nm,~]= fileparts(FILES{k});
    mtxt = ternary(isfinite(MASS_G(k)), sprintf('%.3f',MASS_G(k)), 'NA');
    ttxt = ternary(isfinite(TEMP_C(k)), sprintf('%.1f',TEMP_C(k)), 'NA');
    LEG{k} = sprintf('%s | V=%.3g V, m=%s g, T=%s°C', nm, V_APPLIED(k), mtxt, ttxt);
end

%% ================= LOAD ALL FILES + PRINT SUMMARY =================
DATA = cell(1,nF);
for k=1:nF
    D = load_one(FILES{k}, V_APPLIED(k), I_SCALE, F_MIN, F_MAX, USE_SMOOTH, SMOOTH_WIN);
    DATA{k} = D;

    if SHOW_WARN_UNIT_MISMATCH && median(D.I_mag) < 1e-7 && ~XY_in_uA
        warning('File %d: Current looks very small; if X,Y are µA, set XY_in_uA=true.', k);
    end
    if SHOW_WARN_UNIT_MISMATCH && median(D.I_mag) > 1 && XY_in_uA
        warning('File %d: Current looks very large; if X,Y are A, set XY_in_uA=false.', k);
    end

    % Console summary
    [Zmin, imin] = min(D.Zmag);
    [Zmax, imax] = max(D.Zmag);
    fprintf('--- %s ---\n', D.file);
    fprintf('V=%.3g V | N=%d | f=[%.0f..%.0f] Hz | |I|=[%.3e..%.3e] A | |Z|=[%.3g..%.3g] Ω\n', ...
        D.V, numel(D.f), min(D.f), max(D.f), min(D.I_mag), max(D.I_mag), min(D.Zmag), max(D.Zmag));
    fprintf('   Min|Z| @ %.1f Hz: |Z|=%.3g Ω, R=%.3g Ω, X=%.3g Ω, ∠Z=%.1f°\n', ...
        D.f(imin), Zmin, D.R(imin), D.Xc(imin), D.Zdeg(imin));
    fprintf('   Max|Z| @ %.1f Hz: |Z|=%.3g Ω, R=%.3g Ω, X=%.3g Ω, ∠Z=%.1f°\n\n', ...
        D.f(imax), Zmax, D.R(imax), D.Xc(imax), D.Zdeg(imax));
end

%% ================= BUILD SUMMARY NOTE FOR FIGURES =================
note = "Summary per file:";
for k=1:1:nF
    D = DATA{k}; [Zmin, imin] = min(D.Zmag); [Zmax, imax] = max(D.Zmag);
    [~,nm,~] = fileparts(D.file);
    mtxt = ternary(isfinite(MASS_G(k)), sprintf('%.3f',MASS_G(k)), 'NA');
    ttxt = ternary(isfinite(TEMP_C(k)), sprintf('%.1f',TEMP_C(k)), 'NA');
    note = note + sprintf('\n%s | V=%.3gV, m=%sg, T=%s°C | Min|Z|=%.3gΩ@%.0fHz | Max|Z|=%.3gΩ@%.0fHz', ...
        nm, D.V, mtxt, ttxt, Zmin, D.f(imin), Zmax, D.f(imax));
end


%% ================= PLOTTING HELPERS =================
cols = lines(nF);
plotfun = @(f,y) ( USE_LOG_X * semilogx(f,y,'-','LineWidth',1.2) + ...
                   (~USE_LOG_X) * plot    (f,y,'-','LineWidth',1.2) ); %#ok<NASGU>

% %% ================= FIGURE 4: |Z| =================
% figure('Name','4) |Z|','Position',[100 100 1180 520]); hold on; grid on;
% for k=1:nF, plot_x(DATA{k}.f, DATA{k}.Zmag_plot, USE_LOG_X, '-', 'LineWidth',1.5, 'Color', cols(k,:)); end
% xlim([F_MIN F_MAX]); ylabel('|Z| [\Omega]'); title('Impedance magnitude');
% legend(LEG,'Location','bestoutside'); add_note(gcf, note);
% 
% %% ================= FIGURE 5: ∠Z =================
% figure('Name','5) ∠Z','Position',[120 120 1180 520]); hold on; grid on;
% for k=1:nF, plot_x(DATA{k}.f, DATA{k}.Zdeg_plot, USE_LOG_X, '-', 'LineWidth',1.5, 'Color', cols(k,:)); end
% xlim([F_MIN F_MAX]); ylabel('\angle Z [deg]'); title('Impedance phase');
% legend(LEG,'Location','bestoutside'); add_note(gcf, note);

%% ================= FIGURE 6: Re/Im(Z) =================
figure('Name','6) Re/Im(Z)','Position',[140 140 1180 720]);
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

ax1 = nexttile; hold on; grid on;
for k=1:1:nF, plot_x(DATA{k}.f, DATA{k}.R , USE_LOG_X, '-', 'LineWidth',1.4, 'Color', cols(k,:)); end
xlim([F_MIN F_MAX]); ylabel('R [\Omega]'); title('Resistance Re(Z)');
legend(ax1, LEG,'Location','bestoutside');

ax2 = nexttile; hold on; grid on;
for k=1:1:nF, plot_x(DATA{k}.f, DATA{k}.Xc, USE_LOG_X, '-', 'LineWidth',1.4, 'Color', cols(k,:)); end
xlim([F_MIN F_MAX]); ylabel('X [\Omega]'); xlabel('Frequency [Hz]'); title('Reactance Im(Z)');
legend(ax2, LEG,'Location','bestoutside');;

%%

% ---- Extract weights (in grams) and temperatures (in °C) ----
nFiles = numel(FILES);
%weights = zeros(1, nFiles);
temps = zeros(1, nFiles);

for i = 1:nFiles
    [~, name, ~] = fileparts(FILES{i});
    % Match the pattern "t#####" for weight and the trailing "_##.#" for temperature
    tokens = regexp(name, 't(\d+)_([\d\.]+)', 'tokens');
    if ~isempty(tokens)
        %weights(i) = str2double(tokens{1}{1}) / 100;  % e.g. t10920 → 109.20 g
        temps(i)   = str2double(tokens{1}{2});        % e.g. 25.8 → 25.8 °C
    end
end

% Example: extract data from all files
for k = 1:nF
    RealData{k}   = movmean(DATA{k}.R,1);
    Imaginary{k}  = DATA{k}.Xc;
    Freq{k}       = DATA{k}.f;
end

% Define frequency limits
f_min = 165e3;  % 165 kHz
f_max = 200e3;  % 200 kHz

% Apply mask to all datasets
for k = 1:nFiles
    freq_mask    = (Freq{k} >= f_min) & (Freq{k} <= f_max);
    RealData{k}  = RealData{k}(freq_mask);
    Imaginary{k} = Imaginary{k}(freq_mask);
    Freq{k}      = Freq{k}(freq_mask);
end



%%
% Compute RMSD of each dataset vs its reference
nFiles = numel(RealData);
repeats = nFiles;

n = numel(RealData);
base = 1:repeats:n;
ref_idx = repelem(base, repeats);
ref_idx = ref_idx(1:n);
R_ref = RealData(ref_idx);

nGroups = numel(base);
rmsd_vals = zeros(repeats, nGroups);
delta_weight_vals = zeros(repeats, nGroups);
slopes = zeros(1, nGroups);

for g = 1:nGroups
    ref_i = base(g);
    for r = 1:repeats
        k = (g-1)*repeats + r;
        if k <= nFiles
            rmsd_vals(r,g) = computeRMSD(RealData{k}, RealData{ref_i});
            delta_weight_vals(r,g) = weights(ref_i) - weights(k);
        end
    end
end

% Plot all groups with their linear trends
figure; hold on;
for g = 1:nGroups
    x = delta_weight_vals(:,g);
    y = rmsd_vals(:,g);
    p = polyfit(x,y,1);
    slopes(g) = p(1);
    trend = polyval(p,x);
    plot(x,y,'-o','LineWidth',1.2);
    plot(x,trend,'--','LineWidth',1.2);
end

xlabel('Delta Weight (change from 115g)');
ylabel('RMSD');
title('RMSD vs Weight per Reference Group');

% Legend with slope per gram
legend(arrayfun(@(g)sprintf('Group %d: %.4f RMSD/g', g, slopes(g)), 1:nGroups, 'UniformOutput', false), ...
    'Location','best');

grid on; hold off;



%%
% Preallocate
nFiles = numel(RealData);
peak_freq  = zeros(1, nFiles);
peak_amp   = zeros(1, nFiles);
data_skew  = zeros(1, nFiles);
data_mean  = zeros(1, nFiles);
data_std   = zeros(1, nFiles);
data_rms   = zeros(1, nFiles);
data_median = zeros(1, nFiles);
data_ptp   = zeros(1, nFiles);
data_auc   = zeros(1, nFiles);

for k = 1:nFiles
    R = RealData{k};
    f = Freq{k};

    % Existing features
    [peak_amp(k), idx] = max(R);
    peak_freq(k) = f(idx);
    data_skew(k) = skewness(R);
    data_mean(k) = mean(R);

    % New features
    data_std(k)    = std(R);
    data_rms(k)    = rms(R);
    data_median(k) = median(R);
    data_ptp(k)    = max(R) - min(R);
    data_auc(k)    = trapz(f, R);  % integrate over frequency
end




% Group setup
n = numel(RealData);
base = 1:repeats:n;
ref_idx = repelem(base, repeats);
ref_idx = ref_idx(1:n);
nGroups = numel(base);

% Delta values per group
delta_peak_amp  = zeros(repeats, nGroups);
delta_peak_freq = zeros(repeats, nGroups);
delta_skew      = zeros(repeats, nGroups);
delta_mean      = zeros(repeats, nGroups);
delta_std   = zeros(repeats, nGroups);
delta_rms   = zeros(repeats, nGroups);
delta_median = zeros(repeats, nGroups);
delta_ptp   = zeros(repeats, nGroups);
delta_auc   = zeros(repeats, nGroups);
delta_weight_vals = zeros(repeats, nGroups);

for g = 1:nGroups
    ref_i = base(g);
    for r = 1:repeats
        k = (g-1)*repeats + r;
        if k <= nFiles
            delta_peak_amp(r,g)  = peak_amp(k)  - peak_amp(ref_i);
            delta_peak_freq(r,g) = peak_freq(k) - peak_freq(ref_i);
            delta_skew(r,g)      = data_skew(k) - data_skew(ref_i);
            delta_mean(r,g)      = data_mean(k) - data_mean(ref_i);
            delta_std(r,g)    = data_std(k) - data_std(ref_i);
            delta_rms(r,g)    = data_rms(k) - data_rms(ref_i);
            delta_median(r,g) = data_median(k) - data_median(ref_i);
            delta_ptp(r,g)    = data_ptp(k) - data_ptp(ref_i);
            delta_auc(r,g)    = data_auc(k) - data_auc(ref_i);
            delta_weight_vals(r,g) = weights(k) - weights(end);
        end
    end
end


amp_coeff = (corrcoef(delta_peak_amp,delta_weight_vals));
freq_coeff = (corrcoef(delta_peak_freq,delta_weight_vals));
skew_coeff = (corrcoef(delta_skew,delta_weight_vals));
mean_coeff = (corrcoef(delta_mean,delta_weight_vals));
std_coeff    = corrcoef(delta_std, delta_weight_vals);
rms_coeff    = corrcoef(delta_rms, delta_weight_vals);
median_coeff = corrcoef(delta_median, delta_weight_vals);
ptp_coeff    = corrcoef(delta_ptp, delta_weight_vals);
auc_coeff    = corrcoef(delta_auc, delta_weight_vals);

% Collect coefficients
feature_names = {'Amp', 'Freq', 'Skew', 'Mean', 'Std', 'RMS', 'Median', 'PtP', 'AUC'};
coeffs = [amp_coeff(2), freq_coeff(2), skew_coeff(2), mean_coeff(2), ...
          std_coeff(2), rms_coeff(2), median_coeff(2), ptp_coeff(2), auc_coeff(2)];



% Sort by absolute value descending
[sorted_abs, idx] = sort(abs(coeffs), 'descend');

% Display sorted results
disp('Feature correlations sorted by strength:')
for k = 1:length(coeffs)
    f = feature_names{idx(k)};
    c = coeffs(idx(k));
    disp(f + " coeff: " + c)
end

% Collect all features in a matrix
all_features = [peak_freq', peak_amp', data_skew', data_mean', data_std', data_rms', data_median', data_ptp', data_auc'];

% Compute correlation matrix
corr_matrix = corrcoef(all_features);

% Optionally, display as a heatmap
figure;
heatmap(feature_names, feature_names, corr_matrix, 'ColorLimits', [-1 1]);
title('Feature Correlation Matrix');

%%
% 
% load('Coeffs_uniform.mat')
% load('Coeffs_drill.mat')
% 
% % Assume each contains a variable named 'y' (adjust if different)
% y = [coeffs_uniform; coeffs];  % or [y_uniform; y_drill] if named differently
% 
% feature_names = {'Amp','Freq','Skew','Mean','Std','RMS','Median','PtP','AUC'};
% 
% bar(y')
% set(gca, 'XTickLabel', feature_names)
% xtickangle(45)
% legend({'Uniform','Drill'})
% ylabel('Coefficient value')
% title('Feature Coefficients Comparison')
% 



%%

% Plot grouped subplots with slopes in legend
figure;

% --- Peak Amplitude ---
subplot(2,2,1); hold on;
slopes = zeros(1,nGroups);
for g = 1:nGroups
    x = delta_weight_vals(:,g);
    y = delta_peak_amp(:,g);
    p = polyfit(x,y,1);
    slopes(g) = p(1);
    plot(x,y,'-o','LineWidth',1.2);
    plot(x,polyval(p,x),'--','LineWidth',1.2);
end
legend(arrayfun(@(g)sprintf('Group %d: %.3f per g',g,slopes(g)),1:nGroups,'UniformOutput',false));
xlabel('Delta Weight (change from 115g)'); ylabel('\Delta Peak Amplitude');
title('Peak Amplitude vs Weight'); grid on;

% --- Peak Frequency ---
subplot(2,2,2); hold on;
slopes = zeros(1,nGroups);
for g = 1:nGroups
    x = delta_weight_vals(:,g);
    y = delta_peak_freq(:,g)/1e3;
    p = polyfit(x,y,1);
    slopes(g) = p(1);
    plot(x,y,'-o','LineWidth',1.2);
    plot(x,polyval(p,x),'--','LineWidth',1.2);
end
legend(arrayfun(@(g)sprintf('Group %d: %.3f kHz/g',g,slopes(g)),1:nGroups,'UniformOutput',false));
xlabel('Delta Weight (change from 115g)'); ylabel('\Delta Peak Frequency [kHz]');
title('Peak Frequency vs Weight'); grid on;

% --- Skewness ---
subplot(2,2,3); hold on;
slopes = zeros(1,nGroups);
for g = 1:nGroups
    x = delta_weight_vals(:,g);
    y = delta_skew(:,g);
    p = polyfit(x,y,1);
    slopes(g) = p(1);
    plot(x,y,'-o','LineWidth',1.2);
    plot(x,polyval(p,x),'--','LineWidth',1.2);
end
legend(arrayfun(@(g)sprintf('Group %d: %.3f per g',g,slopes(g)),1:nGroups,'UniformOutput',false));
xlabel('Delta Weight (change from 115g)'); ylabel('\Delta Skewness');
title('Skewness vs Weight'); grid on;

% --- Mean Value ---
subplot(2,2,4); hold on;
slopes = zeros(1,nGroups);
for g = 1:nGroups
    x = delta_weight_vals(:,g);
    y = delta_mean(:,g);
    p = polyfit(x,y,1);
    slopes(g) = p(1);
    plot(x,y,'-o','LineWidth',1.2);
    plot(x,polyval(p,x),'--','LineWidth',1.2);
end
legend(arrayfun(@(g)sprintf('Group %d: %.3f per g',g,slopes(g)),1:nGroups,'UniformOutput',false));
xlabel('Delta Weight (change from 115g)'); ylabel('\Delta Mean Value');
title('Mean Value vs Weight'); grid on;





%% ================= FIGURE 6: Re/Im(Z) =================
figure;
hold on; grid on;
% Use semilogx if USE_LOG_X is true, otherwise regular plot
for k = 1:nF
        plot(f, RealData{k}, '-', 'LineWidth',1.4, 'Color', cols(k,:));
end

hold off
%xlim([f_min f_max]); 
ylabel('R [\Omega]'); 
title('Resistance Re(Z)');
legend(LEG, 'Location', 'bestoutside');




%% ================= LOCAL FUNCTIONS =================
function D = load_one(file_path, V_rms, I_SCALE, F_MIN, F_MAX, USE_SMOOTH, SMOOTH_WIN)
    if exist(file_path,'file')~=2
        error('File not found:\n%s', file_path);
    end
    S  = load(file_path);
    fn = fieldnames(S);
    dev = S.(fn{1});
    if ~isfield(dev,'demods') || ~isfield(dev.demods,'sample')
        error('No dev.demods.sample in %s', file_path);
    end
    s = dev.demods.sample; if iscell(s), s = s{1}; end

    f     = double(s.frequency(:));   % Hz
    X_u   = double(s.x(:));           % Re(I) raw
    Y_u   = double(s.y(:));           % Im(I) raw
    phi_I = double(s.phase(:));       % rad (info)

    % scale to Ampere
    X = I_SCALE * X_u;
    Y = I_SCALE * Y_u;

    % finite + frequency ROI
    m = isfinite(f) & isfinite(X) & isfinite(Y) & isfinite(phi_I);
    f = f(m); X = X(m); Y = Y(m); phi_I = phi_I(m);
    m = (f>=F_MIN & f<=F_MAX);
    f = f(m); X = X(m); Y = Y(m); phi_I = phi_I(m);

    % complex current & magnitude
    I       = X + 1i*Y;
    I_mag   = hypot(X,Y);
    phiIdeg = rad2deg(unwrap(phi_I));

    % avoid division by near-zero
    epsI = max(max(I_mag)*1e-12, 1e-18);
    I(I_mag < epsI) = epsI;

    % impedance
    Z     = V_rms ./ I;
    Zmag  = abs(Z);
    Zang  = angle(Z);              % rad
    Zdeg  = rad2deg(unwrap(Zang)); % deg
    R     = real(Z);
    Xc    = imag(Z);

    % optional smoothing (display only)
    if USE_SMOOTH && numel(f)>=SMOOTH_WIN
        try
            Zmag_plot = movmedian(Zmag,SMOOTH_WIN);
            Zdeg_plot = movmean  (Zdeg ,SMOOTH_WIN);
        catch
            Zmag_plot = Zmag; Zdeg_plot = Zdeg;
        end
    else
        Zmag_plot = Zmag; Zdeg_plot = Zdeg;
    end

    % pack
    D.file  = file_path;
    D.V     = V_rms;
    D.f     = f;
    D.X     = X;           % Re(I)
    D.Y     = Y;           % Im(I)
    D.I     = I;
    D.I_mag = I_mag;
    D.phiIdeg = phiIdeg;

    D.Z     = Z;
    D.Zmag  = Zmag;
    D.Zang  = Zang;        % rad
    D.Zdeg  = Zdeg;        % deg
    D.R     = R;
    D.Xc    = Xc;

    D.Zmag_plot = Zmag_plot;
    D.Zdeg_plot = Zdeg_plot;
end

function h = plot_x(f,y,use_log,varargin)
    if use_log, h = semilogx(f,y,varargin{:});
    else,       h = plot    (f,y,varargin{:});
    end
end

function add_note(fig_handle, note_str)
    annotation(fig_handle,'textbox',[0.50 0.02 0.49 0.20], ...
        'String',note_str,'FitBoxToText','on', ...
        'BackgroundColor',[1 1 1 0.85], 'Interpreter','none');
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end



function rmsd = computeRMSD(array1, array2)
    % Check if input arrays are of the same size
    if length(array1) ~= length(array2)
        error('Input arrays must have the same size.');
    end
    
    % Compute the squared differences
    diffSquared = (array1 - array2).^2;
    
    % Compute the mean of the squared differences
    meanDiffSquared = mean(diffSquared);
    
    % Compute the RMSD
    rmsd = sqrt(meanDiffSquared);
end

