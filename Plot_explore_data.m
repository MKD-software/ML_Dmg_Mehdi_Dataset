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
DATA_DIR = 'C:\Users\au585732\ML_Dmg_Mehdi_Dataset\Datafolder';

% Get list of all .mat files in the directory
matFiles = dir(fullfile(DATA_DIR, '*.mat'));

% Preallocate cell array for file paths
FILES = cell(1, numel(matFiles));

% Fill FILES with full paths
for k = 1:numel(matFiles)
    FILES{k} = fullfile(DATA_DIR, matFiles(k).name);
end
FILES = flip(FILES);

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
for k=1:nF
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

%% ================= FIGURE 4: |Z| =================
figure('Name','4) |Z|','Position',[100 100 1180 520]); hold on; grid on;
for k=1:nF, plot_x(DATA{k}.f, DATA{k}.Zmag_plot, USE_LOG_X, '-', 'LineWidth',1.5, 'Color', cols(k,:)); end
xlim([F_MIN F_MAX]); ylabel('|Z| [\Omega]'); title('Impedance magnitude');
legend(LEG,'Location','bestoutside'); add_note(gcf, note);

%% ================= FIGURE 5: ∠Z =================
figure('Name','5) ∠Z','Position',[120 120 1180 520]); hold on; grid on;
for k=1:nF, plot_x(DATA{k}.f, DATA{k}.Zdeg_plot, USE_LOG_X, '-', 'LineWidth',1.5, 'Color', cols(k,:)); end
xlim([F_MIN F_MAX]); ylabel('\angle Z [deg]'); title('Impedance phase');
legend(LEG,'Location','bestoutside'); add_note(gcf, note);

%% ================= FIGURE 6: Re/Im(Z) =================
figure('Name','6) Re/Im(Z)','Position',[140 140 1180 720]);
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

ax1 = nexttile; hold on; grid on;
for k=1:nF, plot_x(DATA{k}.f, DATA{k}.R , USE_LOG_X, '-', 'LineWidth',1.4, 'Color', cols(k,:)); end
xlim([F_MIN F_MAX]); ylabel('R [\Omega]'); title('Resistance Re(Z)');

ax2 = nexttile; hold on; grid on;
for k=1:nF, plot_x(DATA{k}.f, DATA{k}.Xc, USE_LOG_X, '-', 'LineWidth',1.4, 'Color', cols(k,:)); end
xlim([F_MIN F_MAX]); ylabel('X [\Omega]'); xlabel('Frequency [Hz]'); title('Reactance Im(Z)');
legend(ax2, LEG,'Location','bestoutside'); add_note(gcf, note);

%%

% ---- Extract weights (in grams) and temperatures (in °C) ----
nFiles = numel(FILES);
weights = zeros(1, nFiles);
temps = zeros(1, nFiles);

for i = 1:nFiles
    [~, name, ~] = fileparts(FILES{i});
    % Match the pattern "t#####" for weight and the trailing "_##.#" for temperature
    tokens = regexp(name, 't(\d+)_([\d\.]+)', 'tokens');
    if ~isempty(tokens)
        weights(i) = str2double(tokens{1}{1}) / 100;  % e.g. t10920 → 109.20 g
        temps(i)   = str2double(tokens{1}{2});        % e.g. 25.8 → 25.8 °C
    end
end

% Example: extract data from all files
for k = 1:nF
    RealData{k}   = movmean(DATA{k}.R,50);
    Imaginary{k}  = DATA{k}.Xc;
    Freq{k}       = DATA{k}.f;
end

% Define frequency limits
f_min = 165e3;  % 165 kHz
f_max = 200e3;  % 200 kHz

% Apply mask to all datasets
for k = 1:nFiles
    freq_mask = (Freq{k} >= f_min) & (Freq{k} <= f_max);
    RealData{k}  = RealData{k}(freq_mask);
    Imaginary{k} = Imaginary{k}(freq_mask);
    Freq{k}      = Freq{k}(freq_mask);
end



%%
% Compute RMSD of each dataset vs the first one
nFiles = numel(RealData);
rmsd_vals = zeros(1, nFiles);

R_ref = RealData{1};
for k = 1:nFiles
    rmsd_vals(k) = computeRMSD(RealData{k}, R_ref);
end

% Compute delta weight relative to 115 g
delta_weight = 115-weights;

% Fit linear trend of RMSD vs delta_weight
p = polyfit(delta_weight, rmsd_vals, 1);  % slope and intercept
trend = polyval(p, delta_weight);

% Compute slope per gram
slope_per_gram = p(1);

% Plot RMSD as line and trend
figure;
plot(115-weights, rmsd_vals, '-o', 'LineWidth', 1.5); % line plot
hold on;
plot(115-weights, trend, '-r', 'LineWidth', 1.5);

xlabel('Delta Weight (change from 115g)');
ylabel('RMSD');
title('RMSD vs Weight with Trend Line');
legend('RMSD', sprintf('Trend: %.4f RMSD per g', slope_per_gram));
grid on;
hold off;

%%
% Preallocate
nFiles = numel(RealData);
peak_freq  = zeros(1, nFiles);
peak_amp   = zeros(1, nFiles);
data_skew  = zeros(1, nFiles);
data_mean  = zeros(1, nFiles);

for k = 1:nFiles
    R = RealData{k};
    f = Freq{k};

    % Peak amplitude and corresponding frequency
    [peak_amp(k), idx] = max(R);
    peak_freq(k) = f(idx);

    % Skew and mean
    data_skew(k) = skewness(R);
    data_mean(k) = mean(R);
end

% Compute delta relative to first measurement
delta_peak_amp  = peak_amp  - peak_amp(1);
delta_peak_freq = peak_freq - peak_freq(1);
delta_skew      = data_skew - data_skew(1);
delta_mean      = data_mean - data_mean(1);

% Plot in subplots
figure;

% --- Peak Amplitude ---
subplot(2,2,1)
plot(115-weights, delta_peak_amp, '-o','LineWidth',1.5)
hold on
p = polyfit(115-weights, delta_peak_amp, 1); % linear fit
plot(115-weights, polyval(p, 115-weights), '--r', 'LineWidth',1.5)
legend(sprintf('Data\nSlope = %.3f per g', p(1)))
xlabel('Delta Weight (change from 115g)'); ylabel('\Delta Peak Amplitude')
title('Peak Amplitude vs Weight')
grid on

% --- Peak Frequency ---
subplot(2,2,2)
plot(115-weights, delta_peak_freq/1e3, '-o','LineWidth',1.5)
hold on
p = polyfit(115-weights, delta_peak_freq, 1);
plot(115-weights, polyval(p, 115-weights)/1e3, '--r', 'LineWidth',1.5)
legend(sprintf('Data\nSlope = %.3f kHz per g', p(1)/1e3))
xlabel('Delta Weight (change from 115g)'); ylabel('\Delta Peak Frequency [kHz]')
title('Peak Frequency vs Weight')
grid on

% --- Skewness ---
subplot(2,2,3)
plot(115-weights, delta_skew, '-o','LineWidth',1.5)
hold on
p = polyfit(115-weights, delta_skew, 1);
plot(115-weights, polyval(p, 115-weights), '--r', 'LineWidth',1.5)
legend(sprintf('Data\nSlope = %.3f per g', p(1)))
xlabel('Delta Weight (change from 115g)'); ylabel('\Delta Skewness')
title('Skewness vs Weight')
grid on

% --- Mean Value ---
subplot(2,2,4)
plot(115-weights, delta_mean, '-o','LineWidth',1.5)
hold on
p = polyfit(115-weights, delta_mean, 1);
plot(115-weights, polyval(p, 115-weights), '--r', 'LineWidth',1.5)
legend(sprintf('Data\nSlope = %.3f per g', p(1)))
xlabel('Delta Weight (change from 115g)'); ylabel('\Delta Mean Value')
title('Mean Value vs Weight')
grid on

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

