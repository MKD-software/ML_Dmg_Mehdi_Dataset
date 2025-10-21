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
%% ================= USER OPTIONS =================
DATA_DIR = 'C:\Users\LEGION\Documents\MATLAB\poject\session_20250903_112550_05\sweep123_003';

% ---- Add/Remove files here (any count) ----
FILES = { ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t10920_25.8.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t10940_26.1.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t10965_26.6.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t10990_26.0.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11010_26.0.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11030_25.5.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11050_23.9.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11070_24.2.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11090_24.9.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11110_25.3.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11130_25.1.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11150_25.6.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11170_26.1.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11190_24.8.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11205_25.4.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11225_25.3.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11250_25.4.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11270_25.5.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11290_25.4.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11310_24.9.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11330_24.9.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11350_25.4.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11370_25.5.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11390_25.3.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11410_26.0.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11430_25.3.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11480_23.6.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11480_27.5.mat'), ...
    fullfile(DATA_DIR,'sweep_50_500K_9000_1v_t11500_23.6.mat')  ...
};

% ---- Per-file parameters (you can give scalars; they auto-expand) ----
V_APPLIED = 0.707;      % [V RMS] (scalar or 1xN array)
XY_in_uA  = false;      % true if s.x/s.y are in microampere in your files; else false for ampere

% ---- Analysis range ----
F_MIN = 50e3;           % [Hz]
F_MAX = 500e3;          % [Hz]

% ---- Plot style & cosmetic smoothing (display only for |Z| & ∠Z) ----
USE_LOG_X  = false;     % false=linear freq (like LabOne), true=log freq
USE_SMOOTH = false;
SMOOTH_WIN = 5;

% ---- Common frequency grid for RMSD(|Z|) computations ----
N_GRID = 2000;

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

%% ================= COMMON GRID + RMSD(|Z|) FOR FIG.8 =================
% Frequency overlap across all files
f_lo = F_MIN; f_hi = F_MAX;
for k=1:nF
    f_lo = max(f_lo, min(DATA{k}.f));
    f_hi = min(f_hi, max(DATA{k}.f));
end
if f_lo >= f_hi
    error('No frequency overlap across files in [F_MIN..F_MAX].');
end
fgrid = linspace(f_lo, f_hi, N_GRID);

% |Z| interpolated onto common grid
Zmat = zeros(nF, N_GRID);
for k=1:nF
    Zmat(k,:) = interp1(DATA{k}.f, DATA{k}.Zmag, fgrid, 'pchip');
end

% Pairwise RMSD(|Z|)
RMSD_pair = zeros(nF);
for i=1:nF
    for j=1:nF
        d = Zmat(i,:) - Zmat(j,:);
        RMSD_pair(i,j) = sqrt(mean(d.^2,'omitnan'));
    end
end

% RMSD of each file to the group mean
Z_mean = mean(Zmat,1,'omitnan');
RMSD_to_mean = zeros(1,nF);
for k=1:nF
    d = Zmat(k,:) - Z_mean;
    RMSD_to_mean(k) = sqrt(mean(d.^2,'omitnan'));
end

%% ================= PLOTTING HELPERS =================
cols = lines(nF);
plotfun = @(f,y) ( USE_LOG_X * semilogx(f,y,'-','LineWidth',1.2) + ...
                   (~USE_LOG_X) * plot    (f,y,'-','LineWidth',1.2) ); %#ok<NASGU>

%% ================= FIGURE 1: |I| =================
figure('Name','1) |I|','Position',[40 40 1180 520]); hold on; grid on;
for k=1:nF, plot_x(DATA{k}.f, DATA{k}.I_mag, USE_LOG_X, '-', 'LineWidth',1.2, 'Color', cols(k,:)); end
xlim([F_MIN F_MAX]); ylabel('|I| [A]'); title('Current magnitude |I| = sqrt(X^2 + Y^2)');
legend(LEG,'Location','bestoutside'); add_note(gcf, note);

%% ================= FIGURE 2: φI =================
figure('Name','2) φI','Position',[60 60 1180 520]); hold on; grid on;
for k=1:nF, plot_x(DATA{k}.f, DATA{k}.phiIdeg, USE_LOG_X, '-', 'LineWidth',1.2, 'Color', cols(k,:)); end
xlim([F_MIN F_MAX]); ylabel('\phi_I [deg]'); title('Current phase (unwrapped)');
legend(LEG,'Location','bestoutside'); add_note(gcf, note);

%% ================= FIGURE 3: V_RMS (flat lines) ===============
figure('Name','3) V_RMS','Position',[80 80 980 420]); hold on; grid on;
for k=1:nF
    f = DATA{k}.f;
    plot_x(f, ones(size(f))*V_APPLIED(k), USE_LOG_X, '-', 'LineWidth',1.5, 'Color', cols(k,:));
end
xlim([F_MIN F_MAX]); ylabel('V_{RMS} [V]'); title('Applied RMS voltage');
legend(LEG,'Location','bestoutside'); add_note(gcf, note);

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

%% ================= FIGURE 7: Re/Im(I) =================
figure('Name','7) Re/Im(I)','Position',[160 160 1180 720]);
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

ax3 = nexttile; hold on; grid on;
for k=1:nF, plot_x(DATA{k}.f, DATA{k}.X, USE_LOG_X, '-', 'LineWidth',1.3, 'Color', cols(k,:)); end
xlim([F_MIN F_MAX]); ylabel('Re(I) [A]'); title('Current real part X');

ax4 = nexttile; hold on; grid on;
for k=1:nF, plot_x(DATA{k}.f, DATA{k}.Y, USE_LOG_X, '-', 'LineWidth',1.3, 'Color', cols(k,:)); end
xlim([F_MIN F_MAX]); ylabel('Im(I) [A]'); xlabel('Frequency [Hz]'); title('Current imaginary part Y');
legend(ax4, LEG,'Location','bestoutside'); add_note(gcf, note);

%% ================= FIGURE 8: RMSD Heatmap + RMSD to Mean ==========
figure('Name','8) RMSD(|Z|) — all-vs-all','Position',[180 180 1200 560]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% all-vs-all heatmap
axh = nexttile; imagesc(RMSD_pair); colorbar; axis image;
title(axh, sprintf('RMSD(|Z|) over [%.0f..%.0f] Hz', f_lo, f_hi));
set(axh,'XTick',1:nF,'XTickLabel',1:nF,'YTick',1:nF,'YTickLabel',1:nF);
xlabel('File #'); ylabel('File #');

% RMSD to group mean
axb = nexttile; bar(RMSD_to_mean,'FaceAlpha',0.85); grid on;
title(axb,'RMSD(|Z|) to Group Mean'); xlabel('File #'); ylabel('RMSD [\Omega]');
xticklabels( arrayfun(@(k) sprintf('%d',k), 1:nF, 'uni',0) );

% file names under the figure
note_rmsd = "Files:";
for k=1:nF
    [~,nm,~] = fileparts(FILES{k});
    note_rmsd = note_rmsd + sprintf('\n#%d: %s',k,nm);
end
add_note(gcf, note_rmsd);

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
