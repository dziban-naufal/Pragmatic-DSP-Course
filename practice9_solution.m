%% PRACTICE ENVELOPE 9 - WORKED SOLUTION (annotated)
% Pressure-sensor front-end.  Release to students AFTER they have tried it themselves.
% Each %% block is one step of the viva.  Run section by section (Ctrl+Enter) and read the printed values.
clear; clc; close all;

%% Step 1 - Generate the discrete signal (sampling + ADC)
fs    = 400;          % sampling frequency (Hz)
nbits = 10;           % ADC resolution (bits)
Vmax  = 2;            % ADC range is -Vmax ... +Vmax (volts)
T     = 2;            % record length (s)
N     = round(fs*T);  % number of samples = 800
t     = (0:N-1)/fs;   % sampling instants t[n] = n/fs   (row vector)

% Continuous formulas EVALUATED at the sampling instants = sampling
x1 = 1.0*sin(2*pi*8*t);                              % wanted pressure waveform
x2 = 0.5*sin(2*pi*50*t) + 0.6*sin(2*pi*270*t);       % mains hum + out-of-band spur

% Uniform quantiser: step size, round to nearest level, saturate at the ends
delta = 2*Vmax/2^nbits;                              % = 4/1024 = 3.90625 mV
quant = @(x) min(max(delta*round(x/delta), -Vmax), Vmax - delta);
x1q = quant(x1);
x2q = quant(x2);

sqnr = 10*log10(mean(x1.^2)/mean((x1q-x1).^2));      % measured signal-to-quantisation-noise ratio
fprintf('delta = %.6f V   measured SQNR of x1 = %.1f dB\n', delta, sqnr);

%% Step 2 - Identify the alias
% The 270 Hz tone is above fs/2 = 200 Hz, so after sampling it folds back.
f_hi    = 270;
f_alias = abs(f_hi - fs*round(f_hi/fs));             % |270 - 400*1| = 130 Hz
fprintf('The %g Hz tone appears at %g Hz after sampling.\n', f_hi, f_alias);

% Proof: the spectrum of x2q has a peak at 130 Hz and NONE at 270 Hz (which is not even on the axis)
X2 = fft(x2q);  f = (0:N-1)*fs/N;
figure(1); plot(f(1:N/2), 2*abs(X2(1:N/2))/N); grid on
xlabel('Frequency (Hz)'); ylabel('Amplitude'); title('Spectrum of x2q: the 270 Hz tone shows up at 130 Hz');

%% Step 3 - Combine the signals (addition, sample by sample)
u = x1q + x2q;
figure;plot(u)
%% Step 4 - System defined by its impulse response
h = [0.5 0.5];                  % 2-point average: y[n] = 0.5 u[n] + 0.5 u[n-1]
y = conv(u, h);                 % output = input convolved with h   (length N + L - 1 = 801)
figure;plot(y)
H = fft(h, N);                  % frequency response sampled at the N DFT frequencies (zero-padded h)
% The system is a low-pass: |H(f)| = |cos(pi*f/fs)|.  Gain at 8, 50, 130 Hz:
for fk = [8 50 130]
    fprintf('|H| at %3d Hz = %.3f\n', fk, abs(H(round(fk*N/fs)+1)));
end

%% Step 5 - Spectral analysis: fft and our OWN DFT
yN = y(1:N);                    % first N samples: integer number of cycles of every tone -> clean peaks
Y  = fft(yN);                   % f was defined in Step 2: f = (0:N-1)*fs/N
A  = 2*abs(Y(1:N/2))/N;         % one-sided amplitude (factor 2 because the other half is the mirror image)
figure(2); plot(f(1:N/2), A); grid on
xlabel('Frequency (Hz)'); ylabel('Amplitude'); title('Spectrum of y (after the system)');

% Own DFT from the definition X[k] = sum_n x[n] exp(-j 2 pi k n / M), first M = 64 samples
M  = 64;  n = 0:M-1;  Xd = zeros(1, M);
for kk = 0:M-1
    Xd(kk+1) = sum(yN(1:M) .* exp(-1j*2*pi*kk*n/M));
end
fprintf('Largest difference between own DFT and fft: %.2e\n', max(abs(Xd - fft(yN(1:M)))));
% (Equivalent matrix form:  W = exp(-1j*2*pi*(0:M-1)'*(0:M-1)/M);  Xd = (W*yN(1:M).').' )

% Peaks: local maxima that are at least 10 % of the tallest peak
isp      = A(2:end-1) > A(1:end-2) & A(2:end-1) >= A(3:end) & A(2:end-1) > 0.10*max(A);
f_peaks  = f(find(isp)+1);                 % -> 8  50  130 Hz
f_wanted = f_peaks(1);                     % lowest = wanted (8 Hz)
f_unwanted = f_peaks(2:end);               % 50 Hz (mains) and 130 Hz (the ALIAS, indistinguishable from a real tone)
disp(f_peaks)

%% Step 6 - Filter out the unwanted frequencies
% All unwanted tones are ABOVE the wanted one, so a low-pass filter does the job.
fp  = 1.2*f_wanted;             % passband edge (Hz): a little above 8 Hz   (9.6 Hz)
fst = min(f_unwanted);          % stopband edge (Hz): the nearest unwanted tone (50 Hz)
[nord, Wn] = buttord(fp/(fs/2), fst/(fs/2), 1, 35);   % <=1 dB loss in passband, >=35 dB in stopband
[b, a] = butter(nord, Wn);      % Wn is NORMALISED: 1 = Nyquist = fs/2 (200 Hz), not 1 = fs
yf = filter(b, a, y);           % causal IIR filtering
filter_type = 'IIR';  order = nord;     % order = 3

Yf = fft(yf(1:N));
figure(3); plot(f(1:N/2), A, f(1:N/2), 2*abs(Yf(1:N/2))/N); grid on
legend('before','after'); xlabel('Frequency (Hz)'); ylabel('Amplitude'); title('Before and after filtering');

% Check the specification: gain at each peak over the last half of the record (integer cycles)
seg = N/2+1:N;  nn = (N/2:N-1)';
ampf = @(x, fk) abs(sum(x(:) .* exp(-1j*2*pi*fk*nn/fs)))*2/numel(x);
for fk = f_peaks
    fprintf('Gain at %3d Hz: %6.1f dB\n', fk, 20*log10(ampf(yf(seg), fk)/ampf(y(seg), fk)));
end

%% Option B - FIR alternative (linear phase), same specification
% Hamming-window FIR: about 3.3*fs/(fst-fp) taps -> order 34 (35 taps)
Mfir = 34;
hf   = fir1(Mfir, ((fp+fst)/2)/(fs/2));     % low-pass, Hamming window by default; cutoff between the two edges
yfir = filter(hf, 1, y);                    % delay = Mfir/2 = 17 samples = 42.5 ms; every frequency delayed equally
